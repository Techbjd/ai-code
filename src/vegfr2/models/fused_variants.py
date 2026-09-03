"""Fused Variants - Generalized GNN + Fingerprint architecture.

Any GNN backbone + optional fingerprint branch. Fingerprints are fused
after GNN pooling, not baked into node features.

Architecture:
    Graph branch:  x=[N,32] → GNN layers → pooling → [B, hidden]
    FP branch:     fp=[B, fp_dim] → Linear → ReLU → Linear → [B, hidden]
    Fusion:        concat([graph_emb, fp_emb]) → classifier → [B, 1]

Supported backbones: gin, gcn, gat, gatv2, mpnn
Fingerprint types:   none, morgan, maccs, both
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GATv2Conv,
    GINConv,
    NNConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)


# ---------------------------------------------------------------------------
# GNN backbone extractors (return node embeddings before pooling)
# ---------------------------------------------------------------------------

class _GCNBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        for _ in range(layers):
            self.norms.append(nn.LayerNorm(hidden))
            self.dropouts.append(nn.Dropout(dropout))
        self.hidden = hidden

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        for i, (conv, norm, drop) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = drop(x)
        return global_mean_pool(x, batch)


class _GATBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, heads: int, dropout: float):
        super().__init__()
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
        self.hidden = hidden

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        for i, (conv, norm, drop) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = drop(x)
        return global_mean_pool(x, batch)


class _GATv2Backbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.convs.append(GATv2Conv(in_dim, hidden // heads, heads=heads, dropout=dropout))
        for _ in range(layers - 2):
            self.convs.append(GATv2Conv(hidden, hidden // heads, heads=heads, dropout=dropout))
        self.convs.append(GATv2Conv(hidden, hidden, heads=1, dropout=dropout))
        for _ in range(layers):
            self.norms.append(nn.LayerNorm(hidden))
            self.dropouts.append(nn.Dropout(dropout))
        self.hidden = hidden

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        for i, (conv, norm, drop) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = drop(x)
        return global_mean_pool(x, batch)


class _GINBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.node_emb = nn.Linear(in_dim, hidden)
        self.gin_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
            )
            self.gin_convs.append(GINConv(mlp, train_eps=True))
            self.bn.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        self.hidden = hidden

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        x = F.relu(self.node_emb(x))
        for i, (conv, bn) in enumerate(zip(self.gin_convs, self.bn)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.gin_convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)


class _MPNNBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, edge_dim: int, dropout: float):
        super().__init__()
        self.node_emb = nn.Linear(in_dim, hidden)
        self.edge_nets = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(layers):
            edge_net = nn.Sequential(
                nn.Linear(edge_dim, hidden * hidden),
                nn.ReLU(),
                nn.Linear(hidden * hidden, hidden * hidden),
            )
            self.edge_nets.append(edge_net)
            self.convs.append(NNConv(hidden, hidden, edge_net, aggr="add"))
            self.norms.append(nn.LayerNorm(hidden))
            self.dropouts.append(nn.Dropout(dropout))
        self.gru = nn.GRUCell(hidden, hidden)
        self.hidden = hidden

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        x = F.relu(self.node_emb(x))
        h = x
        for edge_net, conv, norm, drop in zip(self.edge_nets, self.convs, self.norms, self.dropouts):
            m = conv(h, edge_index, edge_attr)
            m = norm(m)
            m = drop(m)
            h = self.gru(m, h)
        return global_mean_pool(h, batch)


# ---------------------------------------------------------------------------
# Main fused variant model
# ---------------------------------------------------------------------------

BackboneMap = {
    "gcn": _GCNBackbone,
    "gat": _GATBackbone,
    "gatv2": _GATv2Backbone,
    "gin": _GINBackbone,
    "mpnn": _MPNNBackbone,
}


class FusedVariant(nn.Module):
    """GNN backbone + optional fingerprint branch.

    Args:
        gnn_type: GNN backbone ("gcn", "gat", "gatv2", "gin", "mpnn")
        in_dim: Node feature dimension (32 for graph-only)
        fp_type: Fingerprint type ("none", "morgan", "maccs", "both")
        fp_dim: Fingerprint dimension (0 if fp_type="none")
        hidden: Hidden dimension
        layers: Number of GNN layers
        heads: Attention heads (for GAT/GATv2)
        edge_dim: Edge feature dimension (for MPNN)
        out_dim: Output dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        gnn_type: str = "gin",
        in_dim: int = 32,
        fp_type: Literal["none", "morgan", "maccs", "both"] = "none",
        fp_dim: int = 0,
        hidden: int = 64,
        layers: int = 3,
        heads: int = 4,
        edge_dim: int = 11,
        out_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.fp_type = fp_type
        self.fp_dim = fp_dim
        self.hidden = hidden

        self.init_kwargs = {
            "gnn_type": gnn_type,
            "in_dim": in_dim,
            "fp_type": fp_type,
            "fp_dim": fp_dim,
            "hidden": hidden,
            "layers": layers,
            "heads": heads,
            "edge_dim": edge_dim,
            "out_dim": out_dim,
            "dropout": dropout,
        }

        # Build GNN backbone
        if gnn_type not in BackboneMap:
            raise ValueError(f"Unknown gnn_type: {gnn_type}. Choose from: {list(BackboneMap.keys())}")

        BackboneCls = BackboneMap[gnn_type]
        if gnn_type == "mpnn":
            self.backbone = BackboneCls(in_dim, hidden, layers, edge_dim, dropout)
        elif gnn_type in ("gat", "gatv2"):
            self.backbone = BackboneCls(in_dim, hidden, layers, heads, dropout)
        else:
            self.backbone = BackboneCls(in_dim, hidden, layers, dropout)

        gnn_out_dim = self.backbone.hidden

        # Fingerprint projection (optional)
        if fp_type != "none" and fp_dim > 0:
            self.fp_proj = nn.Sequential(
                nn.Linear(fp_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
            )
            classifier_in = gnn_out_dim + hidden
        else:
            self.fp_proj = None
            classifier_in = gnn_out_dim

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Tensor | None = None,
        fingerprint: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Node features [N_nodes, in_dim]
            edge_index: Graph connectivity [2, E]
            batch: Batch assignment [N_nodes]
            edge_attr: Edge features [E, edge_dim] (used by MPNN)
            fingerprint: Molecular fingerprints [B, fp_dim] (if fp_type != "none")

        Returns:
            logits: [B, out_dim]
        """
        # Graph branch
        g_emb = self.backbone(x, edge_index, batch, edge_attr)  # [B, hidden]

        # Fingerprint branch (optional)
        if self.fp_proj is not None and fingerprint is not None:
            fp_emb = self.fp_proj(fingerprint)  # [B, hidden]
            fused = torch.cat([g_emb, fp_emb], dim=-1)  # [B, hidden*2]
        else:
            fused = g_emb

        return self.classifier(fused)

    def get_embeddings(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Tensor | None = None,
        fingerprint: Tensor | None = None,
    ) -> Tensor:
        """Get fused embeddings (before classifier)."""
        g_emb = self.backbone(x, edge_index, batch, edge_attr)
        if self.fp_proj is not None and fingerprint is not None:
            fp_emb = self.fp_proj(fingerprint)
            return torch.cat([g_emb, fp_emb], dim=-1)
        return g_emb
