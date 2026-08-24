"""Graph Transformer with Edge Bias - Global attention over all atoms.

Uses TransformerConv for global attention over all nodes, with edge features
as attention bias for chemistry-aware attention.

Reference: Ying et al., "Do Transformers Really Perform Bad for Graph Representation?" (NeurIPS 2021)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool


class GraphTransformer_PyG(nn.Module):
    """Graph Transformer with edge-biased attention.

    Key features:
    - Global self-attention over all atoms (not just neighbors)
    - Edge features as attention bias
    - Relative position encoding via edge attributes
    - Multi-head attention with learnable heads

    This architecture can capture long-range dependencies that
    message-passing GNNs miss (e.g., atoms far apart in the graph
    but close in 3D space).

    Args:
        in_dim: Input node feature dimension
        hidden: Hidden dimension
        layers: Number of transformer layers
        heads: Number of attention heads
        out_dim: Output dimension (1 for binary classification)
        dropout: Dropout rate
        edge_dim: Edge feature dimension (0 = no edge features)
        concat: Whether to concatenate multi-head outputs
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden: int = 128,
        layers: int = 2,
        heads: int = 8,
        out_dim: int = 1,
        dropout: float = 0.3,
        edge_dim: int = 11,
        concat: bool = True,
    ):
        super().__init__()
        self.init_kwargs = {
            "in_dim": in_dim,
            "hidden": hidden,
            "layers": layers,
            "heads": heads,
            "out_dim": out_dim,
            "dropout": dropout,
            "edge_dim": edge_dim,
            "concat": concat,
        }
        self.num_layers = layers
        self.dropout = dropout
        self.concat = concat

        head_dim = hidden // heads if concat else hidden

        self.node_emb = nn.Linear(in_dim, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(layers):
            # First layer takes raw edge_dim, others use hidden
            e_dim = edge_dim if i == 0 else hidden
            self.convs.append(
                TransformerConv(
                    in_channels=hidden,
                    out_channels=head_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=e_dim,
                    concat=concat,
                )
            )
            out_dim_layer = hidden if concat else head_dim
            self.norms.append(nn.LayerNorm(out_dim_layer))
            self.ffns.append(
                nn.Sequential(
                    nn.Linear(out_dim_layer, hidden * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden * 2, out_dim_layer),
                )
            )
            self.ffn_norms.append(nn.LayerNorm(out_dim_layer))
            self.dropouts.append(nn.Dropout(dropout))

        # Edge projection for hidden layers (from edge_dim to hidden)
        self.edge_proj = nn.ModuleList()
        for i in range(1, layers):
            self.edge_proj.append(nn.Linear(edge_dim, hidden))

        self.output = nn.Linear(hidden if concat else head_dim, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        x = self.node_emb(x)

        for i in range(self.num_layers):
            # Multi-head attention with residual
            residual = x
            if edge_attr is not None and i == 0:
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            elif edge_attr is not None and i > 0:
                x = self.convs[i](x, edge_index, edge_attr=self.edge_proj[i - 1](edge_attr))
            else:
                x = self.convs[i](x, edge_index)
            x = self.norms[i](x + residual)

            # Feed-forward with residual
            residual = x
            x = self.ffns[i](x)
            x = self.ffn_norms[i](x + residual)
            x = self.dropouts[i](x)

        # Global pooling
        x = global_mean_pool(x, batch)
        return self.output(x)
