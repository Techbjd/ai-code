"""Pure-PyTorch message-passing networks (GCN, GAT, MPNN) for molecular graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.utils.data

from vegfr2.features import collate_graphs, mol_to_graph
from vegfr2.types import GraphBatch


MODEL_NAMES = {"gcn": "GCN", "gat": "GAT", "mpnn": "MPNN"}
REV_MODEL_NAMES = {v: k for k, v in MODEL_NAMES.items()}


def _mean_pool(h: Tensor, node_batch: Tensor, num_graphs: int) -> Tensor:
    pooled = torch.zeros(num_graphs, h.shape[1], dtype=h.dtype, device=h.device)
    pooled.index_add_(0, node_batch, h)
    counts = torch.bincount(node_batch, minlength=num_graphs).clamp(min=1).unsqueeze(1)
    return pooled / counts


class GCN(nn.Module):
    def __init__(self, in_dim: int = 28, hidden: int = 64, layers: int = 3, out_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.init_kwargs = {"in_dim": in_dim, "hidden": hidden, "layers": layers, "out_dim": out_dim, "dropout": dropout}
        self.input = nn.Linear(in_dim, hidden)
        self.linears = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(layers)])
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, batch_dict: GraphBatch, device: str | torch.device = "cpu") -> Tensor:
        x = batch_dict["node_feats"].to(device)
        edge_index = batch_dict["edge_index"].to(device)
        node_batch = batch_dict["node_batch"].to(device)
        num_graphs = batch_dict["num_graphs"]

        h = F.relu(self.input(x))
        num_nodes = h.shape[0]
        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(num_nodes, device=device).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float)) + 1.0
        norm = deg.rsqrt()

        for lin, ln, drop in zip(self.linears, self.norms, self.dropouts):
            m = h[src] * norm[src].unsqueeze(1) * norm[dst].unsqueeze(1)
            agg = torch.zeros_like(h).index_add_(0, dst, m)
            h = F.relu(drop(ln(lin(h + agg))))

        pooled = _mean_pool(h, node_batch, num_graphs)
        return self.output(pooled)


class GAT(nn.Module):
    def __init__(self, in_dim: int = 28, hidden: int = 64, layers: int = 3, heads: int = 4, out_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.init_kwargs = {"in_dim": in_dim, "hidden": hidden, "layers": layers, "heads": heads, "out_dim": out_dim, "dropout": dropout}
        assert hidden % heads == 0, "hidden must be divisible by heads"
        self.head_dim = hidden // heads
        self.num_layers = layers
        self.heads = heads

        self.input = nn.Linear(in_dim, hidden)
        self.layer_linears = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.attn_src = nn.ModuleList([nn.Linear(hidden, 1, bias=False) for _ in range(layers)])
        self.attn_dst = nn.ModuleList([nn.Linear(hidden, 1, bias=False) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(layers)])
        self.out_proj = nn.Linear(self.head_dim, hidden)
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, batch_dict: GraphBatch, device: str | torch.device = "cpu") -> Tensor:
        x = batch_dict["node_feats"].to(device)
        edge_index = batch_dict["edge_index"].to(device)
        node_batch = batch_dict["node_batch"].to(device)
        num_graphs = batch_dict["num_graphs"]

        h = self.input(x)
        src, dst = edge_index[0], edge_index[1]
        num_nodes = h.shape[0]

        for i in range(self.num_layers):
            h_lin = self.layer_linears[i](h)
            h_heads = h_lin.view(num_nodes, self.heads, self.head_dim)

            e_src = self.attn_src[i](h).squeeze(-1)
            e_dst = self.attn_dst[i](h).squeeze(-1)
            scores = F.leaky_relu(e_src[src] + e_dst[dst], negative_slope=0.2)

            max_score = torch.zeros(num_nodes, device=device).index_reduce_(0, dst, scores, "amax", include_self=False)
            max_score = max_score[dst]
            exp_scores = (scores - max_score).exp()
            denom = torch.zeros(num_nodes, device=device).index_add_(0, dst, exp_scores)
            alpha = exp_scores / denom[dst].clamp_min(1e-12)

            msg = alpha[:, None, None] * h_heads[src]
            agg = torch.zeros_like(h_heads).index_add_(0, dst, msg)

            if i == self.num_layers - 1:
                h = self.out_proj(agg.mean(dim=1))
            else:
                h = agg.view(num_nodes, -1)
                h = F.elu(self.dropouts[i](self.norms[i](h)))

        pooled = _mean_pool(h, node_batch, num_graphs)
        return self.output(pooled)


class MPNN(nn.Module):
    def __init__(self, in_dim: int = 32, hidden: int = 64, layers: int = 3, out_dim: int = 1, edge_dim: int = 11, dropout: float = 0.2):
        super().__init__()
        self.init_kwargs = {"in_dim": in_dim, "hidden": hidden, "layers": layers, "out_dim": out_dim, "edge_dim": edge_dim, "dropout": dropout}
        self.input = nn.Linear(in_dim, hidden)
        self.edge_mlps = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)) for _ in range(layers)]
        )
        self.grus = nn.ModuleList([nn.GRUCell(hidden, hidden) for _ in range(layers)])
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, batch_dict: GraphBatch, device: str | torch.device = "cpu") -> Tensor:
        x = batch_dict["node_feats"].to(device)
        edge_index = batch_dict["edge_index"].to(device)
        edge_feats = batch_dict["edge_feats"].to(device)
        node_batch = batch_dict["node_batch"].to(device)
        num_graphs = batch_dict["num_graphs"]

        h = F.relu(self.input(x))
        src, dst = edge_index[0], edge_index[1]
        num_nodes = h.shape[0]

        for edge_mlp, gru in zip(self.edge_mlps, self.grus):
            msg = edge_mlp(torch.cat([h[src], h[dst], edge_feats], dim=-1))
            agg = torch.zeros_like(h).index_add_(0, dst, msg)
            h = gru(agg, h)

        pooled = _mean_pool(h, node_batch, num_graphs)
        return self.output(pooled)


def build_model(
    name: str,
    in_dim: int = 32,
    hidden: int = 64,
    layers: int = 3,
    heads: int = 4,
    out_dim: int = 1,
    edge_dim: int = 11,
    dropout: float = 0.2,
) -> nn.Module:
    name = name.lower()
    if name == "gcn":
        return GCN(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim, dropout=dropout)
    if name == "gat":
        return GAT(in_dim=in_dim, hidden=hidden, layers=layers, heads=heads, out_dim=out_dim, dropout=dropout)
    if name == "mpnn":
        return MPNN(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim, edge_dim=edge_dim, dropout=dropout)
    raise ValueError(f"Unknown GNN model: {name}. Available: gcn, gat, mpnn")


def save_checkpoint(model: nn.Module, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_name = REV_MODEL_NAMES.get(type(model).__name__, type(model).__name__.lower())
    torch.save(
        {"model_state": model.state_dict(), "init_kwargs": model.init_kwargs, "model_name": model_name}, path
    )
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    model_name = ckpt["model_name"]
    model = build_model(model_name, **ckpt["init_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, smiles: list[str], labels: list[int]):
        self.graphs = [mol_to_graph(s) for s in smiles]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return self.graphs[idx], self.labels[idx]


def train_gnn_model(
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
    """Train a GNN model (GCN, GAT, MPNN) on SMILES data.
    
    Args:
        name: Model name ('gcn', 'gat', 'mpnn')
        train_smiles: List of training SMILES
        train_labels: List of training labels (0/1)
        val_smiles: Optional validation SMILES
        val_labels: Optional validation labels
        hidden: Hidden dimension
        layers: Number of layers
        heads: Number of attention heads (GAT only)
        lr: Learning rate
        batch_size: Batch size
        epochs: Max epochs
        patience: Early stopping patience
        seed: Random seed
        device: Device to train on
        
    Returns:
        Trained model
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    
    train_ds = GraphDataset(train_smiles, train_labels)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs)
    
    if val_smiles is not None:
        val_ds = GraphDataset(val_smiles, val_labels)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    else:
        val_loader = None
    
    model = build_model(name, in_dim=32, hidden=hidden, layers=layers, heads=heads, edge_dim=11).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    
    best_auc = -1.0
    best_state = None
    wait = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch, device)
            loss = loss_fn(logits.squeeze(), batch["labels"].squeeze())
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        if val_loader is not None:
            model.eval()
            val_probs = []
            val_true = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    logits = model(batch, device)
                    val_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
                    val_true.extend(batch["labels"].squeeze().cpu().numpy().astype(int))
            
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


def predict_gnn_model(model: nn.Module, smiles: list[str], batch_size: int = 256, device: str | torch.device = "cuda") -> np.ndarray:
    """Predict probabilities for SMILES using trained GNN.
    
    Args:
        model: Trained GNN model
        smiles: List of SMILES strings
        batch_size: Batch size for inference
        device: Device for inference
        
    Returns:
        Array of probabilities (0-1)
    """
    import numpy as np
    device = torch.device(device)
    model.eval()
    
    ds = GraphDataset(smiles, [0] * len(smiles))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch, device)
            probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
    
    return np.array(probs)