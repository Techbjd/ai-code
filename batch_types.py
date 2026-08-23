from typing import TypedDict
import torch
from torch import Tensor


class GraphBatch(TypedDict):
    x: Tensor                    # [N, 28] node features
    edge_index: Tensor           # [2, E] edge connections
    edge_attr: Tensor            # [E, ...] edge features (optional)
    batch: Tensor                # [N] graph index per node
    labels: Tensor               # [B] labels per graph
    num_graphs: int              # number of graphs in batch


# Usage
def move_to_device(batch: GraphBatch, device: torch.device) -> GraphBatch:
    return {k: v.to(device) for k, v in batch.items()}


# Or with dataclass (more runtime safety)
from dataclasses import dataclass

@dataclass
class GraphBatchDC:
    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    batch: Tensor
    labels: Tensor
    num_graphs: int

    def to(self, device: torch.device) -> "GraphBatchDC":
        return GraphBatchDC(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            batch=self.batch.to(device),
            labels=self.labels.to(device),
            num_graphs=self.num_graphs,
        )