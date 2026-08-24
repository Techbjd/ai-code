"""GCN (Graph Convolutional Network) using PyTorch Geometric."""

from __future__ import annotations

from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN_PyG(nn.Module):
    """Graph Convolutional Network with LayerNorm and dropout.

    Args:
        in_dim: Input node feature dimension.
        hidden: Hidden dimension.
        layers: Number of GCN layers.
        out_dim: Output dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden: int = 64,
        layers: int = 3,
        out_dim: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.init_kwargs = {
            "in_dim": in_dim,
            "hidden": hidden,
            "layers": layers,
            "out_dim": out_dim,
            "dropout": dropout,
        }
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
        for i, (conv, norm, drop) in enumerate(
            zip(self.convs, self.norms, self.dropouts)
        ):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = drop(x)
        x = global_mean_pool(x, batch)
        return self.output(x)
