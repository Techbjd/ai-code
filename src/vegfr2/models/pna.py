"""PNA (Principal Neighbourhood Aggregation) - Multi-aggregator GNN.

PNA uses multiple aggregators (mean, max, min, std) and multiple scalers
(identity, amplification, attenuation) to capture different graph properties.

Reference: Corso et al., "Principal Neighbourhood Aggregation for Graph Nets" (NeurIPS 2020)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, global_mean_pool, global_max_pool, global_add_pool


AGGREGATORS = ["mean", "min", "max", "std"]
SCALERS = ["identity", "amplification", "attenuation"]


class PNA_PyG(nn.Module):
    """Principal Neighbourhood Aggregation network.

    PNA combines multiple aggregation functions to capture diverse
    neighbourhood patterns. Each layer applies:
    - Multiple aggregators (mean, min, max, std)
    - Multiple scalers (identity, amplification, attenuation)
    - Linear transformation + residual connection

    Args:
        in_dim: Input node feature dimension
        hidden: Hidden dimension
        layers: Number of PNA layers
        out_dim: Output dimension (1 for binary classification)
        dropout: Dropout rate
        towers: Number of parallel towers (split hidden dim)
        pre_layers: Number of pre-transform MLP layers
        post_layers: Number of post-transform MLP layers
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden: int = 128,
        layers: int = 3,
        out_dim: int = 1,
        dropout: float = 0.3,
        towers: int = 4,
        pre_layers: int = 1,
        post_layers: int = 1,
    ):
        super().__init__()
        self.init_kwargs = {
            "in_dim": in_dim,
            "hidden": hidden,
            "layers": layers,
            "out_dim": out_dim,
            "dropout": dropout,
            "towers": towers,
            "pre_layers": pre_layers,
            "post_layers": post_layers,
        }
        self.num_layers = layers
        self.dropout = dropout

        self.node_emb = nn.Linear(in_dim, hidden)

        # Default degree histogram for PNA (computed over typical molecular graphs)
        # In practice, compute from dataset. For molecular graphs: avg degree ~2-4
        avg_degree = 4
        num_bins = 10
        deg = torch.ones(num_bins) / num_bins
        deg[0] = 0.1
        deg[1] = 0.2
        deg[2] = 0.3
        deg[3] = 0.2
        deg[4] = 0.1

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(layers):
            self.convs.append(
                PNAConv(
                    in_channels=hidden,
                    out_channels=hidden,
                    aggregators=AGGREGATORS,
                    scalers=SCALERS,
                    towers=towers,
                    pre_layers=pre_layers,
                    post_layers=post_layers,
                    divide_input=False,
                    deg=deg,
                )
            )
            self.norms.append(nn.LayerNorm(hidden))
            self.dropouts.append(nn.Dropout(dropout))

        self.output = nn.Linear(hidden, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = F.relu(self.node_emb(x))

        for i, (conv, norm, drop) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = drop(x)
            x = x + residual  # residual connection

        x = global_mean_pool(x, batch)
        return self.output(x)
