"""Dataset classes for GNN model input types.

These load from pre-computed data (saved by VEGFR2Pipeline.process_split_plain()),
so no RDKit calls happen at runtime. Each Dataset produces PyG Data objects
with the correct fields for its model variant.

Dataset classes:
    GraphOnlyDataset  -> x=[N,32], edge_index, edge_attr, y
    MorganFPDataset   -> x=[N,32], edge_index, edge_attr, y, morgan_fp=[2048]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class GraphOnlyDataset(torch.utils.data.Dataset):
    """Plain graphs (32-dim atom features). No fingerprints."""

    def __init__(self, split_dir: str | Path) -> None:
        split_dir = Path(split_dir)
        node_feats = torch.load(split_dir / "node_feats_plain.pt", weights_only=True)
        edge_index = torch.load(split_dir / "edge_index.pt", weights_only=True)
        edge_feats = torch.load(split_dir / "edge_feats.pt", weights_only=True)
        labels = torch.load(split_dir / "labels.pt", weights_only=True)

        with open(split_dir / "metadata.json") as f:
            metadata = json.load(f)
        node_counts = metadata["node_counts"]

        self.data_list = []
        node_offset = 0
        for i, n_nodes in enumerate(node_counts):
            edge_mask = (
                (edge_index[0] >= node_offset) & (edge_index[0] < node_offset + n_nodes)
            )
            graph = Data(
                x=node_feats[node_offset:node_offset + n_nodes],
                edge_index=edge_index[:, edge_mask] - node_offset,
                edge_attr=edge_feats[edge_mask],
                y=labels[i].unsqueeze(0),
            )
            self.data_list.append(graph)
            node_offset += n_nodes

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]


class MorganFPDataset(torch.utils.data.Dataset):
    """Plain graphs + pre-computed Morgan fingerprints."""

    def __init__(self, split_dir: str | Path) -> None:
        split_dir = Path(split_dir)
        node_feats = torch.load(split_dir / "node_feats_plain.pt", weights_only=True)
        edge_index = torch.load(split_dir / "edge_index.pt", weights_only=True)
        edge_feats = torch.load(split_dir / "edge_feats.pt", weights_only=True)
        labels = torch.load(split_dir / "labels.pt", weights_only=True)
        morgan_fps = torch.tensor(np.load(split_dir / "morgan_fps.npy"), dtype=torch.float32)

        with open(split_dir / "metadata.json") as f:
            metadata = json.load(f)
        node_counts = metadata["node_counts"]

        self.data_list = []
        node_offset = 0
        for i, n_nodes in enumerate(node_counts):
            edge_mask = (
                (edge_index[0] >= node_offset) & (edge_index[0] < node_offset + n_nodes)
            )
            graph = Data(
                x=node_feats[node_offset:node_offset + n_nodes],
                edge_index=edge_index[:, edge_mask] - node_offset,
                edge_attr=edge_feats[edge_mask],
                y=labels[i].unsqueeze(0),
                morgan_fp=morgan_fps[i].unsqueeze(0),
            )
            self.data_list.append(graph)
            node_offset += n_nodes

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]


FP_TYPE_DATASET_MAP: dict[str, type] = {
    "none": GraphOnlyDataset,
    "morgan": MorganFPDataset,
}


def get_dataset_class(fp_type: str) -> type:
    if fp_type not in FP_TYPE_DATASET_MAP:
        raise ValueError(f"Unknown fp_type: {fp_type}. Choose from: {list(FP_TYPE_DATASET_MAP.keys())}")
    return FP_TYPE_DATASET_MAP[fp_type]
