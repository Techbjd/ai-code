"""GNN models using PyTorch Geometric (PyG) - public API.

Re-exports all model classes and provides the ``build_pyg_model`` factory,
``PlainPyGDataset``, and ``train_gnn_pyg`` / ``predict_gnn_pyg`` helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from vegfr2.features import mol_to_graph
from vegfr2.types import GraphBatch

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
    "PlainPyGDataset",
    "train_gnn_pyg",
    "predict_gnn_pyg",
    "save_checkpoint",
    "load_checkpoint",
]


def build_pyg_model(
    name: str,
    in_dim: int = 45,
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
    name = name.lower()

    from vegfr2.models.fused_variants import FusedVariant
    _FP_DIMS = {"morgan": 2048, "none": 0, "graph_only": 0}
    _FP_TYPES = {"morgan": "morgan", "none": "none", "graph_only": "none"}
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

    from vegfr2.models import MODEL_REGISTRY
    if name in MODEL_REGISTRY:
        factory = MODEL_REGISTRY[name]
        if isinstance(factory, type):
            return factory(in_dim=in_dim, hidden=hidden, layers=layers, heads=heads, out_dim=out_dim, edge_dim=edge_dim, dropout=dropout)
        return factory(in_dim=in_dim, hidden=hidden, layers=layers, heads=heads, out_dim=out_dim, edge_dim=edge_dim, dropout=dropout)

    raise ValueError(f"Unknown model: {name}. Available: gcn, gat, gatv2, mpnn, gin, pna, graph_transformer, attentive_fp")


class PlainPyGDataset(torch.utils.data.Dataset):
    """Dataset with plain graphs (32-dim atom features)."""

    def __init__(self, smiles: list[str], labels: list[int]) -> None:
        self.data_list = []
        for s, y in zip(smiles, labels):
            g = mol_to_graph(s)
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


def graph_forward(model: nn.Module, batch) -> Tensor:
    if hasattr(model, 'fp_proj') and model.fp_proj is not None:
        fp = getattr(batch, 'fingerprint', None)
        if fp is None:
            fp = getattr(batch, 'morgan_fp', None)
        if fp is not None:
            return model(batch.x, batch.edge_index, batch.batch, fingerprint=fp)
        return model(batch.x, batch.edge_index, batch.batch)

    model_name = type(model).__name__
    if model_name == 'AttentiveFP':
        edge_attr = getattr(batch, 'edge_attr', None)
        return model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)

    if model_name == 'MPNN_PyG':
        edge_attr = getattr(batch, 'edge_attr', None)
        return model(batch.x, batch.edge_index, edge_attr, batch.batch)

    if model_name == 'GraphTransformer_PyG':
        edge_attr = getattr(batch, 'edge_attr', None)
        return model(batch.x, batch.edge_index, batch.batch, edge_attr)

    return model(batch.x, batch.edge_index, batch.batch)


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
    torch.manual_seed(seed)
    device = torch.device(device)

    train_ds = PlainPyGDataset(train_smiles, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_smiles is not None:
        val_ds = PlainPyGDataset(val_smiles, val_labels)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_pyg_model(name, in_dim=45, hidden=hidden, layers=layers, heads=heads, edge_dim=11).to(device)
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
    device = torch.device(device)
    model.eval()

    ds = PlainPyGDataset(smiles, [0] * len(smiles))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    probs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = graph_forward(model, batch)
            probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())

    return np.array(probs)


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
    torch.manual_seed(seed)
    device = torch.device(device)

    model = build_pyg_model(name, in_dim=45, hidden=hidden, layers=layers, heads=heads, edge_dim=11).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    wait = 0

    def _forward(model, batch):
        if hasattr(model, 'fp_proj') and model.fp_proj is not None:
            fp = getattr(batch, 'fingerprint', None)
            if fp is None:
                fp = getattr(batch, 'morgan_fp', None)
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
    device = torch.device(device)
    model.eval()

    def _forward(model, batch):
        if hasattr(model, 'fp_proj') and model.fp_proj is not None:
            fp = getattr(batch, 'fingerprint', None)
            if fp is None:
                fp = getattr(batch, 'morgan_fp', None)
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
