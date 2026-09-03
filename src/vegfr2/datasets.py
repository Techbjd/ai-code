"""Clean Dataset classes for each model input type.

These load from pre-computed data (saved by VEGFR2Pipeline.process_split_plain()),
so no RDKit calls happen at runtime. Each Dataset produces PyG Data objects
with the correct fields for its model variant.

Dataset classes:
    GraphOnlyDataset  → x=[N,32], edge_index, edge_attr, y
    MorganFPDataset   → x=[N,32], edge_index, edge_attr, y, morgan_fp=[2048]
    MACCSFPDataset    → x=[N,32], edge_index, edge_attr, y, maccs_fp=[166]
    BothFPDataset     → x=[N,32], edge_index, edge_attr, y, fingerprint=[2214]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


class GraphOnlyDataset(torch.utils.data.Dataset):
    """Plain graphs (32-dim atom features). No fingerprints.

    Loads from:
        split_dir/node_feats_plain.pt  [total_nodes, 32]
        split_dir/edge_index.pt        [2, total_edges]
        split_dir/edge_feats.pt        [total_edges, 11]
        split_dir/labels.pt            [n_molecules]
        split_dir/metadata.json        (node_counts)
    """

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
    """Plain graphs + pre-computed Morgan fingerprints.

    Loads from:
        split_dir/node_feats_plain.pt  [total_nodes, 32]
        split_dir/edge_index.pt        [2, total_edges]
        split_dir/edge_feats.pt        [total_edges, 11]
        split_dir/labels.pt            [n_molecules]
        split_dir/morgan_fps.npy       [n_molecules, 2048]
        split_dir/metadata.json
    """

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
                morgan_fp=morgan_fps[i].unsqueeze(0),  # [1, fp_dim] so DataLoader stacks to [B, fp_dim]
            )
            self.data_list.append(graph)
            node_offset += n_nodes

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]


class MACCSFPDataset(torch.utils.data.Dataset):
    """Plain graphs + pre-computed MACCS fingerprints.

    Loads from:
        split_dir/node_feats_plain.pt  [total_nodes, 32]
        split_dir/edge_index.pt        [2, total_edges]
        split_dir/edge_feats.pt        [total_edges, 11]
        split_dir/labels.pt            [n_molecules]
        split_dir/maccs_fps.npy        [n_molecules, 166]
        split_dir/metadata.json
    """

    def __init__(self, split_dir: str | Path) -> None:
        split_dir = Path(split_dir)
        node_feats = torch.load(split_dir / "node_feats_plain.pt", weights_only=True)
        edge_index = torch.load(split_dir / "edge_index.pt", weights_only=True)
        edge_feats = torch.load(split_dir / "edge_feats.pt", weights_only=True)
        labels = torch.load(split_dir / "labels.pt", weights_only=True)
        maccs_fps = torch.tensor(np.load(split_dir / "maccs_fps.npy"), dtype=torch.float32)

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
                maccs_fp=maccs_fps[i].unsqueeze(0),  # [1, fp_dim] so DataLoader stacks to [B, fp_dim]
            )
            self.data_list.append(graph)
            node_offset += n_nodes

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]


class BothFPDataset(torch.utils.data.Dataset):
    """Plain graphs + pre-computed Morgan + MACCS fingerprints.

    Loads from:
        split_dir/node_feats_plain.pt  [total_nodes, 32]
        split_dir/edge_index.pt        [2, total_edges]
        split_dir/edge_feats.pt        [total_edges, 11]
        split_dir/labels.pt            [n_molecules]
        split_dir/morgan_fps.npy       [n_molecules, 2048]
        split_dir/maccs_fps.npy        [n_molecules, 166]
        split_dir/metadata.json
    """

    def __init__(self, split_dir: str | Path) -> None:
        split_dir = Path(split_dir)
        node_feats = torch.load(split_dir / "node_feats_plain.pt", weights_only=True)
        edge_index = torch.load(split_dir / "edge_index.pt", weights_only=True)
        edge_feats = torch.load(split_dir / "edge_feats.pt", weights_only=True)
        labels = torch.load(split_dir / "labels.pt", weights_only=True)
        morgan_fps = torch.tensor(np.load(split_dir / "morgan_fps.npy"), dtype=torch.float32)
        maccs_fps = torch.tensor(np.load(split_dir / "maccs_fps.npy"), dtype=torch.float32)

        # Combined fingerprint
        combined_fps = torch.cat([morgan_fps, maccs_fps], dim=-1)  # [B, 2214]

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
                fingerprint=combined_fps[i].unsqueeze(0),  # [1, fp_dim] so DataLoader stacks to [B, fp_dim]
            )
            self.data_list.append(graph)
            node_offset += n_nodes

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]


# ---------------------------------------------------------------------------
# Convenience mapping: fp_type string → Dataset class
# ---------------------------------------------------------------------------
FP_TYPE_DATASET_MAP: dict[str, type] = {
    "none": GraphOnlyDataset,
    "morgan": MorganFPDataset,
    "maccs": MACCSFPDataset,
    "both": BothFPDataset,
}


def get_dataset_class(fp_type: str) -> type:
    """Get Dataset class by fingerprint type name.

    Args:
        fp_type: One of "none", "morgan", "maccs", "both"

    Returns:
        The corresponding Dataset class
    """
    if fp_type not in FP_TYPE_DATASET_MAP:
        raise ValueError(f"Unknown fp_type: {fp_type}. Choose from: {list(FP_TYPE_DATASET_MAP.keys())}")
    return FP_TYPE_DATASET_MAP[fp_type]
