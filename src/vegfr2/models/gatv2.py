"""GATv2 (Graph Attention Network v2) using PyTorch Geometric.

Key difference from GAT:
    GAT:  e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
    GATv2: e_ij = a^T LeakyReLU(Wh_i || Wh_j)

This makes GATv2 strictly more expressive than GAT.
Reference: Brody et al., "How Attentive are Graph Attention Networks?" (ICLR 2022)
"""

from __future__ import annotations

from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATv2_PyG(nn.Module):
    """Dynamic graph attention network with LayerNorm and dropout.

    Args:
        in_dim: Input node feature dimension.
        hidden: Hidden dimension (must be divisible by ``heads``).
        layers: Number of GATv2 layers.
        heads: Number of attention heads.
        out_dim: Output dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden: int = 64,
        layers: int = 3,
        heads: int = 4,
        out_dim: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.init_kwargs = {
            "in_dim": in_dim,
            "hidden": hidden,
            "layers": layers,
            "heads": heads,
            "out_dim": out_dim,
            "dropout": dropout,
        }
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

        self.output = nn.Linear(hidden, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        for i, (conv, norm, drop) in enumerate(
            zip(self.convs, self.norms, self.dropouts)
        ):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = drop(x)
        x = global_mean_pool(x, batch)
        return self.output(x)
