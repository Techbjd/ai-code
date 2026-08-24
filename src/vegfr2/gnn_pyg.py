"""GNN models using PyTorch Geometric (PyG) - alternative to custom implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, NNConv, global_mean_pool

from vegfr2.features import mol_to_graph, mol_to_graph_with_fps
from vegfr2.types import GraphBatch


class GCN_PyG(nn.Module):
    def __init__(self, in_dim: int = 32, hidden: int = 64, layers: int = 3, out_dim: int = 1, dropout: float = 0.3):
        super().__init__()
        self.init_kwargs = {"in_dim": in_dim, "hidden": hidden, "layers": layers, "out_dim": out_dim, "dropout": dropout}
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        for _ in range(layers):
            self.norms.append(nn.LayerNorm(hidden))
            self.dropouts.append(nn.Dropout(dropout))
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        for i, (conv, norm, drop) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = drop(x)
        x = global_mean_pool(x, batch)
        return self.output(x)


class GAT_PyG(nn.Module):
    def __init__(self, in_dim: int = 32, hidden: int = 64, layers: int = 3, heads: int = 4, out_dim: int = 1, dropout: float = 0.3):
        super().__init__()
        self.init_kwargs = {"in_dim": in_dim, "hidden": hidden, "layers": layers, "heads": heads, "out_dim": out_dim, "dropout": dropout}
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout))
        for _ in range(layers - 2):
            self.convs.append(GATConv(hidden, hidden // heads, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden, hidden, heads=1, dropout=dropout))
        for _ in range(layers):
            self.norms.append(nn.LayerNorm(hidden))
            self.dropouts.append(nn.Dropout(dropout))
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        for i, (conv, norm, drop) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = drop(x)
        x = global_mean_pool(x, batch)
        return self.output(x)


class MPNN_PyG(nn.Module):
    def __init__(self, in_dim: int = 32, hidden: int = 64, layers: int = 3, out_dim: int = 1, edge_dim: int = 11, dropout: float = 0.3):
        super().__init__()
        self.init_kwargs = {"in_dim": in_dim, "hidden": hidden, "layers": layers, "out_dim": out_dim, "edge_dim": edge_dim, "dropout": dropout}
        self.node_emb = nn.Linear(in_dim, hidden)
        self.edge_nets = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(layers):
            edge_net = nn.Sequential(
                nn.Linear(edge_dim, hidden * hidden),
                nn.ReLU(),
                nn.Linear(hidden * hidden, hidden * hidden)
            )
            self.edge_nets.append(edge_net)
            self.convs.append(NNConv(hidden, hidden, edge_net, aggr="add"))
            self.norms.append(nn.LayerNorm(hidden))
            self.dropouts.append(nn.Dropout(dropout))
        self.gru = nn.GRUCell(hidden, hidden)
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor) -> Tensor:
        x = F.relu(self.node_emb(x))
        h = x
        for edge_net, conv, norm, drop in zip(self.edge_nets, self.convs, self.norms, self.dropouts):
            m = conv(h, edge_index, edge_attr)
            m = norm(m)
            m = drop(m)
            h = self.gru(m, h)
        h = global_mean_pool(h, batch)
        return self.output(h)


def build_pyg_model(
    name: str,
    in_dim: int = 32,
    hidden: int = 64,
    layers: int = 3,
    heads: int = 4,
    out_dim: int = 1,
    edge_dim: int = 11,
) -> nn.Module:
    name = name.lower()
    if name == "gcn":
        return GCN_PyG(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim)
    if name == "gat":
        return GAT_PyG(in_dim=in_dim, hidden=hidden, layers=layers, heads=heads, out_dim=out_dim)
    if name == "mpnn":
        return MPNN_PyG(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim, edge_dim=edge_dim)
    raise ValueError(f"Unknown model: {name}. Available: gcn, gat, mpnn")


class PyGDataset(torch.utils.data.Dataset):
    def __init__(self, smiles: list[str], labels: list[int]):
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


class EnrichedPyGDataset(torch.utils.data.Dataset):
    """Dataset that injects Morgan+MACCS fingerprints into graph nodes.
    
    Each atom node gets: [atom_features(32) | morgan(2048) | maccs(166)] = 2246 dims
    This gives the GNN access to fingerprint knowledge during message passing.
    """
    def __init__(
        self,
        smiles: list[str],
        labels: list[int],
        use_morgan: bool = True,
        use_maccs: bool = True,
        morgan_radius: int = 2,
        morgan_n_bits: int = 2048,
        maccs_n_bits: int = 166,
    ):
        self.data_list = []
        for s, y in zip(smiles, labels):
            g = mol_to_graph_with_fps(
                s,
                use_morgan=use_morgan,
                use_maccs=use_maccs,
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

    train_ds = PyGDataset(train_smiles, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_smiles is not None:
        val_ds = PyGDataset(val_smiles, val_labels)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_pyg_model(name, in_dim=32, hidden=hidden, layers=layers, heads=heads, edge_dim=11).to(device)
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
            val_probs = []
            val_true = []
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
    import numpy as np
    device = torch.device(device)
    model.eval()

    ds = PyGDataset(smiles, [0] * len(smiles))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())

    return np.array(probs)


def save_checkpoint(model: nn.Module, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "init_kwargs": model.init_kwargs}, path)
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu", name: str = "gcn") -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    model = build_pyg_model(name, **ckpt["init_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model