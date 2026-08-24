"""MPNN (Message Passing Neural Network) using PyTorch Geometric.

Reference: Gilmer et al., "Neural Message Passing for Quantum Chemistry" (ICML 2017)
"""

from __future__ import annotations

from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool


class MPNN_PyG(nn.Module):
    """Message passing neural network with edge-MLP and GRU update.

    Args:
        in_dim: Input node feature dimension.
        hidden: Hidden dimension.
        layers: Number of MPNN layers.
        out_dim: Output dimension.
        edge_dim: Edge feature dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden: int = 64,
        layers: int = 3,
        out_dim: int = 1,
        edge_dim: int = 11,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.init_kwargs = {
            "in_dim": in_dim,
            "hidden": hidden,
            "layers": layers,
            "out_dim": out_dim,
            "edge_dim": edge_dim,
            "dropout": dropout,
        }
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
        self.output = nn.Linear(hidden, out_dim)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor
    ) -> Tensor:
        x = F.relu(self.node_emb(x))
        h = x
        for edge_net, conv, norm, drop in zip(
            self.edge_nets, self.convs, self.norms, self.dropouts
        ):
            m = conv(h, edge_index, edge_attr)
            m = norm(m)
            m = drop(m)
            h = self.gru(m, h)
        h = global_mean_pool(h, batch)
        return self.output(h)
