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
) -> nn.Module:
    """Instantiate a PyG model by name.

    Args:
        name: One of ``gcn, gat, gatv2, mpnn, gin, pna, graph_transformer``.
        in_dim: Input node feature dimension (2246 = enriched by default).
        hidden: Hidden dimension.
        layers: Number of layers.
        heads: Number of attention heads (used by GAT, GATv2, GIN, GraphTransformer).
        out_dim: Output dimension.
        edge_dim: Edge feature dimension (used by MPNN, GraphTransformer).
        dropout: Dropout rate.

    Returns:
        An ``nn.Module`` ready for training.

    Raises:
        ValueError: If ``name`` is not recognised.
    """
    name = name.lower()
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
        return GIN_PyG(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim, dropout=dropout)
    if name == "pna":
        from vegfr2.models.pna import PNA_PyG
        return PNA_PyG(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim, dropout=dropout)
    if name == "graph_transformer":
        from vegfr2.models.graph_transformer import GraphTransformer_PyG
        return GraphTransformer_PyG(in_dim=in_dim, hidden=hidden, layers=layers, heads=heads, out_dim=out_dim, dropout=dropout, edge_dim=edge_dim)
    raise ValueError(f"Unknown model: {name}. Available: gcn, gat, gatv2, mpnn, gin, pna, graph_transformer")


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
            logits = model(batch.x, batch.edge_index, batch.batch)
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
                    logits = model(batch.x, batch.edge_index, batch.batch)
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
            logits = model(batch.x, batch.edge_index, batch.batch)
            probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())

    return np.array(probs)
