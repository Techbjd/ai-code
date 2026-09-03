"""GNN models using PyTorch Geometric (PyG) - public API.

Re-exports all model classes and provides the ``build_pyg_model`` factory,
``EnrichedPyGDataset``, and ``train_gnn_pyg`` / ``predict_gnn_pyg`` helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from vegfr2.features import mol_to_graph_with_fps
from vegfr2.types import GraphBatch

# Re-export model classes from their own files
from vegfr2.models.gcn import GCN_PyG
from vegfr2.models.gat import GAT_PyG
from vegfr2.models.gatv2 import GATv2_PyG
from vegfr2.models.mpnn import MPNN_PyG
from vegfr2.models._base import save_checkpoint, load_checkpoint

__all__ = [
    "GCN_PyG",
    "GAT_PyG",
    "GATv2_PyG",
    "MPNN_PyG",
    "build_pyg_model",
    "EnrichedPyGDataset",
    "train_gnn_pyg",
    "predict_gnn_pyg",
    "save_checkpoint",
    "load_checkpoint",
]


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def build_pyg_model(
    name: str,
    in_dim: int = 2246,
    hidden: int = 64,
    layers: int = 3,
    heads: int = 4,
    out_dim: int = 1,
    edge_dim: int = 11,
    dropout: float = 0.3,
    jk: bool = True,
    pooling: str = "concat",
    towers: int = 4,
    pre_layers: int = 1,
    post_layers: int = 1,
    concat: bool = True,
    fp_type: str = "none",
    fp_dim: int = 0,
) -> nn.Module:
    """Instantiate a PyG model by name.

    Args:
        name: Model name. Enriched models: ``gcn, gat, gatv2, mpnn, gin, pna, graph_transformer``.
              Fused variants: ``{gnn}_{fp}`` where gnn ∈ {gcn,gat,gatv2,gin,mpnn}
              and fp ∈ {graph_only,morgan,maccs,both}. Example: ``gin_morgan``.
        in_dim: Input node feature dimension (2246 = enriched, 32 = plain).
        hidden: Hidden dimension.
        layers: Number of layers.
        heads: Number of attention heads.
        out_dim: Output dimension.
        edge_dim: Edge feature dimension (used by MPNN, GraphTransformer).
        dropout: Dropout rate.
        jk: Jumping Knowledge for GIN.
        pooling: Readout strategy for GIN.
        towers: Towers for PNA.
        pre_layers: Pre-layers for PNA.
        post_layers: Post-layers for PNA.
        concat: Concat for GraphTransformer.
        fp_type: Fingerprint type for fused variants ("none", "morgan", "maccs", "both").
        fp_dim: Fingerprint dimension for fused variants.

    Returns:
        An ``nn.Module`` ready for training.
    """
    name = name.lower()

    # Fused variant models (e.g., "gin_morgan", "gat_graph_only")
    from vegfr2.models.fused_variants import FusedVariant
    _FP_DIMS = {"morgan": 2048, "maccs": 166, "both": 2214, "none": 0, "graph_only": 0}
    _FP_TYPES = {"morgan": "morgan", "maccs": "maccs", "both": "both", "none": "none", "graph_only": "none"}
    for _gnn in ["gcn", "gat", "gatv2", "gin", "mpnn"]:
        for _fp_key in _FP_DIMS:
            if name == f"{_gnn}_{_fp_key}":
                return FusedVariant(
                    gnn_type=_gnn,
                    in_dim=in_dim,
                    fp_type=_FP_TYPES[_fp_key],
                    fp_dim=_FP_DIMS[_fp_key],
                    hidden=hidden,
                    layers=layers,
                    heads=heads,
                    edge_dim=edge_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                )

    # Enriched models (legacy, fingerprints baked into node features)
    if name == "gcn":
        return GCN_PyG(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim, dropout=dropout)
    if name == "gat":
        return GAT_PyG(in_dim=in_dim, hidden=hidden, layers=layers, heads=heads, out_dim=out_dim, dropout=dropout)
    if name == "gatv2":
        return GATv2_PyG(in_dim=in_dim, hidden=hidden, layers=layers, heads=heads, out_dim=out_dim, dropout=dropout)
    if name == "mpnn":
        return MPNN_PyG(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim, edge_dim=edge_dim, dropout=dropout)
    if name == "gin":
        from vegfr2.models.gin import GIN_PyG
        return GIN_PyG(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim, dropout=dropout, jk=jk, pooling=pooling)
    if name == "pna":
        from vegfr2.models.pna import PNA_PyG
        return PNA_PyG(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim, dropout=dropout, towers=towers, pre_layers=pre_layers, post_layers=post_layers)
    if name == "graph_transformer":
        from vegfr2.models.graph_transformer import GraphTransformer_PyG
        return GraphTransformer_PyG(in_dim=in_dim, hidden=hidden, layers=layers, heads=heads, out_dim=out_dim, dropout=dropout, edge_dim=edge_dim, concat=concat)
    if name == "attentive_fp":
        from vegfr2.models.attentive_fp import AttentiveFP
        return AttentiveFP(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim, dropout=dropout)

    # Check MODEL_REGISTRY as fallback
    from vegfr2.models import MODEL_REGISTRY
    if name in MODEL_REGISTRY:
        factory = MODEL_REGISTRY[name]
        if isinstance(factory, type):
            return factory(in_dim=in_dim, hidden=hidden, layers=layers, heads=heads, out_dim=out_dim, edge_dim=edge_dim, dropout=dropout)
        # partial object from fused_variants
        return factory(in_dim=in_dim, hidden=hidden, layers=layers, heads=heads, out_dim=out_dim, edge_dim=edge_dim, dropout=dropout)

    raise ValueError(f"Unknown model: {name}. Available enriched: gcn, gat, gatv2, mpnn, gin, pna, graph_transformer. Available variants: gin_graph_only, gin_morgan, gin_maccs, gin_both, gat_graph_only, ...")


# ------------------------------------------------------------------
# Dataset (always enriched)
# ------------------------------------------------------------------

class EnrichedPyGDataset(torch.utils.data.Dataset):
    """Dataset that injects Morgan+MACCS fingerprints into graph nodes.

    Each atom node gets: [atom_features(32) | morgan(2048) | maccs(166)] = 2246 dims.
    """

    def __init__(
        self,
        smiles: list[str],
        labels: list[int],
        morgan_radius: int = 2,
        morgan_n_bits: int = 2048,
        maccs_n_bits: int = 166,
    ) -> None:
        self.data_list = []
        for s, y in zip(smiles, labels):
            g = mol_to_graph_with_fps(
                s,
                use_morgan=True,
                use_maccs=True,
                morgan_radius=morgan_radius,
                morgan_n_bits=morgan_n_bits,
                maccs_n_bits=maccs_n_bits,
            )
            data = Data(
                x=g["node_feats"],
                edge_index=g["edge_index"],
                edge_attr=g["edge_feats"],
                y=torch.tensor([y], dtype=torch.float32),
            )
            self.data_list.append(data)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]


# ------------------------------------------------------------------
# Forward helper (handles all model types)
# ------------------------------------------------------------------

def graph_forward(model: nn.Module, batch) -> Tensor:
    """Unified forward pass for all GNN model types.

    Handles:
    - Standard PyG models: model(x, edge_index, batch)
    - MPNN: model(x, edge_index, edge_attr, batch)
    - GraphTransformer: model(x, edge_index, batch, edge_attr)
    - AttentiveFP: model(x, edge_index, batch, edge_attr)
    - Fused models: model(x, edge_index, batch, fingerprint=fp)

    Args:
        model: Any GNN model
        batch: PyG Data/Batch object

    Returns:
        Model output (logits)
    """
    # Fused models with fingerprint branch
    if hasattr(model, 'fp_proj') and model.fp_proj is not None:
        fp = getattr(batch, 'fingerprint', None)
        if fp is None:
            fp = getattr(batch, 'morgan_fp', None)
        if fp is None:
            fp = getattr(batch, 'maccs_fp', None)
        if fp is not None:
            return model(batch.x, batch.edge_index, batch.batch, fingerprint=fp)
        return model(batch.x, batch.edge_index, batch.batch)

    # AttentiveFP needs edge_attr for neighbor_fc
    model_name = type(model).__name__
    if model_name == 'AttentiveFP':
        edge_attr = getattr(batch, 'edge_attr', None)
        return model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)

    # MPNN needs edge_attr
    if model_name == 'MPNN_PyG':
        edge_attr = getattr(batch, 'edge_attr', None)
        return model(batch.x, batch.edge_index, edge_attr, batch.batch)

    # GraphTransformer needs edge_attr
    if model_name == 'GraphTransformer_PyG':
        edge_attr = getattr(batch, 'edge_attr', None)
        return model(batch.x, batch.edge_index, batch.batch, edge_attr)

    # Standard models: model(x, edge_index, batch)
    return model(batch.x, batch.edge_index, batch.batch)


# ------------------------------------------------------------------
# Training / Prediction
# ------------------------------------------------------------------

def train_gnn_pyg(
    name: str,
    train_smiles: list[str],
    train_labels: list[int],
    val_smiles: list[str] | None = None,
    val_labels: list[int] | None = None,
    hidden: int = 64,
    layers: int = 3,
    heads: int = 4,
    lr: float = 0.001,
    batch_size: int = 128,
    epochs: int = 200,
    patience: int = 15,
    seed: int = 42,
    device: str | torch.device = "cuda",
) -> nn.Module:
    """Train a PyG GNN with enriched graphs (always).

    Args:
        name: Model name (see ``build_pyg_model``).
        train_smiles: Training SMILES strings.
        train_labels: Training binary labels (0/1).
        val_smiles: Validation SMILES (optional, for early stopping).
        val_labels: Validation labels.
        hidden: Hidden dimension.
        layers: Number of layers.
        heads: Attention heads.
        lr: Learning rate.
        batch_size: Batch size.
        epochs: Maximum training epochs.
        patience: Early-stopping patience.
        seed: Random seed.
        device: Target device.

    Returns:
        Trained ``nn.Module``.
    """
    torch.manual_seed(seed)
    device = torch.device(device)

    train_ds = EnrichedPyGDataset(train_smiles, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_smiles is not None:
        val_ds = EnrichedPyGDataset(val_smiles, val_labels)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_pyg_model(name, in_dim=2246, hidden=hidden, layers=layers, heads=heads, edge_dim=11).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            logits = graph_forward(model, batch)
            loss = loss_fn(logits.squeeze(), batch.y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if val_loader is not None:
            model.eval()
            val_probs: list[float] = []
            val_true: list[int] = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    logits = graph_forward(model, batch)
                    val_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
                    val_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

            from vegfr2.metrics import classification_metrics
            metrics = classification_metrics(val_true, val_probs)
            val_auc = metrics.get("auc") or 0.0

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_gnn_pyg(
    model: nn.Module,
    smiles: list[str],
    batch_size: int = 256,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """Predict probabilities for SMILES using a trained PyG model.

    Args:
        model: Trained model.
        smiles: SMILES strings to predict.
        batch_size: Inference batch size.
        device: Target device.

    Returns:
        Array of probabilities, shape ``(len(smiles),)``.
    """
    device = torch.device(device)
    model.eval()

    ds = EnrichedPyGDataset(smiles, [0] * len(smiles))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    probs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = graph_forward(model, batch)
            probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())

    return np.array(probs)


# ------------------------------------------------------------------
# Fused Variant Training / Prediction
# ------------------------------------------------------------------

def train_fused_variant(
    name: str,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    hidden: int = 64,
    layers: int = 3,
    heads: int = 4,
    lr: float = 0.001,
    epochs: int = 200,
    patience: int = 15,
    seed: int = 42,
    device: str | torch.device = "cuda",
) -> nn.Module:
    """Train a fused variant model (graph + optional fingerprint branch).

    Args:
        name: Model name (e.g., "gin_morgan", "gat_graph_only").
        train_loader: DataLoader yielding PyG batches with x, edge_index, batch,
                      and optionally morgan_fp/maccs_fp/fingerprint.
        val_loader: Validation DataLoader (optional).
        hidden: Hidden dimension.
        layers: Number of layers.
        heads: Attention heads.
        lr: Learning rate.
        epochs: Maximum training epochs.
        patience: Early-stopping patience.
        seed: Random seed.
        device: Target device.

    Returns:
        Trained model.
    """
    torch.manual_seed(seed)
    device = torch.device(device)

    model = build_pyg_model(name, in_dim=32, hidden=hidden, layers=layers, heads=heads, edge_dim=11).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    wait = 0

    def _forward(model, batch):
        """Handle forward pass for both graph-only and fused models."""
        if hasattr(model, 'fp_proj') and model.fp_proj is not None:
            # Fused model — needs fingerprint
            fp = getattr(batch, 'fingerprint', None)
            if fp is None:
                fp = getattr(batch, 'morgan_fp', None)
            if fp is None:
                fp = getattr(batch, 'maccs_fp', None)
            if fp is not None:
                return model(batch.x, batch.edge_index, batch.batch, fingerprint=fp)
            return model(batch.x, batch.edge_index, batch.batch)
        return graph_forward(model, batch)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            logits = _forward(model, batch)
            loss = loss_fn(logits.squeeze(), batch.y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if val_loader is not None:
            model.eval()
            val_probs: list[float] = []
            val_true: list[int] = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    logits = _forward(model, batch)
                    val_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
                    val_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

            from vegfr2.metrics import classification_metrics
            metrics = classification_metrics(val_true, val_probs)
            val_auc = metrics.get("auc") or 0.0

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_fused_variant(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """Predict probabilities using a trained fused variant model.

    Args:
        model: Trained model.
        loader: DataLoader yielding PyG batches.
        device: Target device.

    Returns:
        Array of probabilities.
    """
    device = torch.device(device)
    model.eval()

    def _forward(model, batch):
        if hasattr(model, 'fp_proj') and model.fp_proj is not None:
            fp = getattr(batch, 'fingerprint', None)
            if fp is None:
                fp = getattr(batch, 'morgan_fp', None)
            if fp is None:
                fp = getattr(batch, 'maccs_fp', None)
            if fp is not None:
                return model(batch.x, batch.edge_index, batch.batch, fingerprint=fp)
            return model(batch.x, batch.edge_index, batch.batch)
        return graph_forward(model, batch)

    probs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = _forward(model, batch)
            probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())

    return np.array(probs)
