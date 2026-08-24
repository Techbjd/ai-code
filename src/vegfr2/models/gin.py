"""GIN (Graph Isomorphism Network) - Provably most expressive message-passing GNN.

GIN uses:
- Learnable epsilon parameter (controls weight of central node)
- MLP aggregation instead of linear transforms
- Mean/Max pooling readout

Reference: Xu et al., "How Powerful are Graph Neural Networks?" (ICLR 2019)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, global_max_pool


class GIN_PyG(nn.Module):
    """Graph Isomorphism Network using PyTorch Geometric.

    GIN is provably the most expressive message-passing GNN under the
    Weisfeiler-Lehman 1-test. It uses:
    - MLP for message transformation (not linear)
    - Learnable epsilon for self-loop weight
    - Multiple readout strategies (mean + max + add)

    Args:
        in_dim: Input node feature dimension
        hidden: Hidden dimension
        layers: Number of GIN layers
        out_dim: Output dimension (1 for binary classification)
        dropout: Dropout rate
        jk: Jumping Knowledge - concatenate all layer outputs (True) or not
        pooling: Readout strategy - "concat" (mean+max+add), "mean", "max"
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden: int = 128,
        layers: int = 3,
        out_dim: int = 1,
        dropout: float = 0.3,
        jk: bool = True,
        pooling: str = "concat",
    ):
        super().__init__()
        self.init_kwargs = {
            "in_dim": in_dim,
            "hidden": hidden,
            "layers": layers,
            "out_dim": out_dim,
            "dropout": dropout,
            "jk": jk,
            "pooling": pooling,
        }
        self.num_layers = layers
        self.jk = jk
        self.pooling = pooling
        self.dropout = dropout

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

        if jk:
            self.jk_linear = nn.Linear(hidden * layers, hidden)

        if pooling == "concat":
            out_pool_dim = hidden * 3  # mean + max + add
        else:
            out_pool_dim = hidden

        self.output = nn.Linear(out_pool_dim, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = F.relu(self.node_emb(x))
        layer_outputs = []

        for i, (conv, bn) in enumerate(zip(self.gin_convs, self.bn)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)

        if self.jk:
            x = self.jk_linear(torch.cat(layer_outputs, dim=-1))

        if self.pooling == "concat":
            g_mean = global_mean_pool(x, batch)
            g_max = global_max_pool(x, batch)
            g_add = global_add_pool(x, batch)
            x = torch.cat([g_mean, g_max, g_add], dim=-1)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)

        return self.output(x)
