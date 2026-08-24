"""Type definitions for graph batches."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor


class GraphBatch(TypedDict):
    """Batched molecular graph from collate_graphs."""
    node_feats: Tensor          # [N, 32] node features
    edge_index: Tensor          # [2, E] edge connections
    edge_feats: Tensor          # [E, 11] edge features
    node_batch: Tensor          # [N] graph index per node
    labels: Tensor              # [B, 1] labels per graph
    num_graphs: int             # number of graphs in batch


class GraphSample(TypedDict):
    """Single molecular graph from mol_to_graph."""
    node_feats: Tensor          # [n_nodes, 32]
    edge_index: Tensor          # [2, n_edges]
    edge_feats: Tensor          # [n_edges, 11]
    num_nodes: int              # number of nodes in this graph


# Type alias for DataLoader batch
GraphDataLoader = torch.utils.data.DataLoader[GraphBatch]