"""Pure-PyTorch message-passing networks (GCN, GAT, MPNN) for molecular graphs."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from vegfr2.types import GraphBatch


MODEL_NAMES = {"gcn": "GCN", "gat": "GAT", "mpnn": "MPNN"}
REV_MODEL_NAMES = {v: k for k, v in MODEL_NAMES.items()}


def _mean_pool(h: Tensor, node_batch: Tensor, num_graphs: int) -> Tensor:
    pooled = torch.zeros(num_graphs, h.shape[1], dtype=h.dtype, device=h.device)
    pooled.index_add_(0, node_batch, h)
    counts = torch.bincount(node_batch, minlength=num_graphs).clamp(min=1).unsqueeze(1)
    return pooled / counts


class GCN(nn.Module):
    def __init__(self, in_dim: int = 28, hidden: int = 64, layers: int = 3, out_dim: int = 1):
        super().__init__()
        self.init_kwargs = {"in_dim": in_dim, "hidden": hidden, "layers": layers, "out_dim": out_dim}
        self.input = nn.Linear(in_dim, hidden)
        self.linears = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
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

        for lin, ln in zip(self.linears, self.norms):
            m = h[src] * norm[src].unsqueeze(1) * norm[dst].unsqueeze(1)
            agg = torch.zeros_like(h).index_add_(0, dst, m)
            h = F.relu(ln(lin(h + agg)))

        pooled = _mean_pool(h, node_batch, num_graphs)
        return self.output(pooled)


class GAT(nn.Module):
    def __init__(self, in_dim: int = 28, hidden: int = 64, layers: int = 3, heads: int = 4, out_dim: int = 1):
        super().__init__()
        self.init_kwargs = {"in_dim": in_dim, "hidden": hidden, "layers": layers, "heads": heads, "out_dim": out_dim}
        assert hidden % heads == 0, "hidden must be divisible by heads"
        self.head_dim = hidden // heads
        self.num_layers = layers
        self.heads = heads

        self.input = nn.Linear(in_dim, hidden)
        self.layer_linears = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.attn_src = nn.ModuleList([nn.Linear(hidden, 1, bias=False) for _ in range(layers)])
        self.attn_dst = nn.ModuleList([nn.Linear(hidden, 1, bias=False) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
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
                h = F.elu(self.norms[i](h))

        pooled = _mean_pool(h, node_batch, num_graphs)
        return self.output(pooled)


class MPNN(nn.Module):
    def __init__(self, in_dim: int = 32, hidden: int = 64, layers: int = 3, out_dim: int = 1, edge_dim: int = 11):
        super().__init__()
        self.init_kwargs = {"in_dim": in_dim, "hidden": hidden, "layers": layers, "out_dim": out_dim, "edge_dim": edge_dim}
        self.input = nn.Linear(in_dim, hidden)
        self.edge_mlps = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(layers)]
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
) -> nn.Module:
    name = name.lower()
    if name == "gcn":
        return GCN(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim)
    if name == "gat":
        return GAT(in_dim=in_dim, hidden=hidden, layers=layers, heads=heads, out_dim=out_dim)
    if name == "mpnn":
        return MPNN(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim, edge_dim=edge_dim)
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