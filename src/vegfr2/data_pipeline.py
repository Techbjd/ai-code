"""
Data Pipeline: SMILES -> Plain Graphs -> Ready-to-train PyTorch data.

This module converts raw SMILES data into preprocessed plain graph
datasets (32-dim atom features) following the paper's method.

Pipeline:
    1. Load raw CSV (SMILES + IC50)
    2. Preprocess (validate, deduplicate, label, split)
    3. Convert each molecule to plain graph (32-dim atom features)
    4. Save as PyTorch tensors + cached Morgan fingerprints

Usage:
    from vegfr2.data_pipeline import VEGFR2Pipeline

    pipeline = VEGFR2Pipeline()
    pipeline.run("data/raw/chembl_vegfr2.csv", output_dir="data/processed")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from vegfr2.data import load_csv, preprocess, split
from vegfr2.features import (
    mol_to_graph,
    smiles_to_morgan,
    clear_fp_cache,
    ATOM_FEAT_DIM,
    BOND_FEAT_DIM,
)


class VEGFR2Pipeline:
    """Full data pipeline: SMILES -> plain graphs -> PyTorch tensors.

    Args:
        morgan_radius: Morgan fingerprint radius
        morgan_n_bits: Morgan fingerprint bits
        seed: Random seed for splitting
    """

    def __init__(
        self,
        morgan_radius: int = 2,
        morgan_n_bits: int = 2048,
        seed: int = 42,
    ):
        self.morgan_radius = morgan_radius
        self.morgan_n_bits = morgan_n_bits
        self.seed = seed
        self.node_dim = ATOM_FEAT_DIM  # 32

    def process_split(
        self,
        df: pd.DataFrame,
        split_name: str,
        output_dir: Path,
    ) -> dict:
        """Process one split saving plain graphs (32-dim) + cached fingerprints.

        Saves:
        - node_feats_plain.pt: [total_nodes, 32]
        - morgan_fps.npy: [n_molecules, 2048]
        - edge_index.pt, edge_feats.pt, labels.pt, node_batch.pt
        - smiles.json: list of SMILES
        """
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        smiles_list = df["smiles"].tolist()
        labels = df["active"].astype(int).tolist()
        n_molecules = len(smiles_list)

        all_node_feats = []
        all_edge_index = []
        all_edge_feats = []
        all_labels = []
        all_morgan_fps = []
        all_smiles = []
        node_counts = []

        failed = 0
        for i, (smiles, label) in enumerate(zip(smiles_list, labels)):
            try:
                graph = mol_to_graph(smiles)
                morgan_fp = torch.tensor(
                    smiles_to_morgan(smiles, radius=self.morgan_radius, n_bits=self.morgan_n_bits),
                    dtype=torch.float32,
                )

                offset = sum(node_counts)
                edge_index = graph["edge_index"] + offset

                all_node_feats.append(graph["node_feats"])
                all_edge_index.append(edge_index)
                all_edge_feats.append(graph["edge_feats"])
                all_labels.append(label)
                all_morgan_fps.append(morgan_fp)
                all_smiles.append(smiles)
                node_counts.append(graph["num_nodes"])

            except Exception:
                failed += 1
                continue

            if (i + 1) % 1000 == 0:
                print(f"    [{split_name}] {i + 1}/{n_molecules} processed...")

        node_feats = torch.cat(all_node_feats, dim=0)
        edge_index = torch.cat(all_edge_index, dim=1)
        edge_feats = torch.cat(all_edge_feats, dim=0)
        labels = torch.tensor(all_labels, dtype=torch.float32)
        morgan_fps = torch.stack(all_morgan_fps)

        node_batch = torch.repeat_interleave(
            torch.arange(len(all_labels)), torch.tensor(node_counts)
        )

        torch.save(node_feats, split_dir / "node_feats_plain.pt")
        torch.save(edge_index, split_dir / "edge_index.pt")
        torch.save(edge_feats, split_dir / "edge_feats.pt")
        torch.save(labels, split_dir / "labels.pt")
        torch.save(node_batch, split_dir / "node_batch.pt")
        np.save(split_dir / "morgan_fps.npy", morgan_fps.numpy())

        with open(split_dir / "smiles.json", "w") as f:
            json.dump(all_smiles, f)

        metadata = {
            "split": split_name,
            "n_molecules": len(all_labels),
            "n_nodes": int(node_feats.shape[0]),
            "n_edges": int(edge_index.shape[1]),
            "node_dim": ATOM_FEAT_DIM,
            "morgan_dim": self.morgan_n_bits,
            "n_active": int(sum(all_labels)),
            "n_inactive": len(all_labels) - int(sum(all_labels)),
            "failed": failed,
            "node_counts": node_counts,
        }
        with open(split_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def run(self, raw_csv: str | Path, output_dir: str | Path = "data/processed") -> dict:
        """Run full pipeline: raw CSV -> processed data."""
        output_dir = Path(output_dir)
        print("=" * 60)
        print("VEGFR2 DATA PIPELINE")
        print("=" * 60)

        print("\n[Step 1] Loading and preprocessing raw data...")
        df = load_csv(raw_csv)
        df = preprocess(df)
        print(f"  Total molecules: {len(df)}")
        print(f"  Active: {df['active'].sum()} ({df['active'].mean():.1%})")
        print(f"  Inactive: {(~df['active'].astype(bool)).sum()}")

        print("\n[Step 2] Splitting into train/val/test...")
        train_df, val_df, test_df = split(df, seed=self.seed)
        print(f"  Train: {len(train_df)}")
        print(f"  Val:   {len(val_df)}")
        print(f"  Test:  {len(test_df)}")

        results = {}
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            print(f"\n[Step 3] Processing {split_name} split ({len(split_df)} molecules)...")
            print(f"  Converting SMILES -> plain graphs (32-dim) + Morgan FPs...")
            stats = self.process_split(split_df, split_name, output_dir)
            results[split_name] = stats
            print(f"  Done: {stats['n_molecules']} molecules, "
                  f"{stats['n_nodes']} nodes, {stats['n_edges']} edges")
            if stats["failed"] > 0:
                print(f"  Warning: {stats['failed']} molecules failed")

        global_meta = {
            "node_dim": ATOM_FEAT_DIM,
            "morgan_dim": self.morgan_n_bits,
            "morgan_radius": self.morgan_radius,
            "seed": self.seed,
            "splits": results,
        }
        with open(output_dir / "pipeline_metadata.json", "w") as f:
            json.dump(global_meta, f, indent=2)

        total = sum(r["n_molecules"] for r in results.values())
        total_nodes = sum(r["n_nodes"] for r in results.values())

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Total molecules: {total}")
        print(f"  Total nodes:     {total_nodes:,}")
        print(f"  Node dimension:  {ATOM_FEAT_DIM} (atom features only)")
        print(f"  Output:          {output_dir}/")

        clear_fp_cache()
        return results

    def load_split(self, output_dir: str | Path, split_name: str) -> dict:
        """Load a preprocessed split as a dict of tensors."""
        output_dir = Path(output_dir)
        split_dir = output_dir / split_name

        if not split_dir.exists():
            raise FileNotFoundError(f"Split not found: {split_dir}")

        node_feats = torch.load(split_dir / "node_feats_plain.pt", weights_only=True)
        edge_index = torch.load(split_dir / "edge_index.pt", weights_only=True)
        edge_feats = torch.load(split_dir / "edge_feats.pt", weights_only=True)
        labels = torch.load(split_dir / "labels.pt", weights_only=True)
        node_batch = torch.load(split_dir / "node_batch.pt", weights_only=True)
        morgan_fps = torch.tensor(np.load(split_dir / "morgan_fps.npy"), dtype=torch.float32)

        with open(split_dir / "metadata.json") as f:
            metadata = json.load(f)

        return {
            "node_feats": node_feats,
            "edge_index": edge_index,
            "edge_feats": edge_feats,
            "labels": labels,
            "node_batch": node_batch,
            "morgan_fps": morgan_fps,
            "metadata": metadata,
        }

    def to_pyg_dataset(self, output_dir: str | Path, split_name: str) -> list[Data]:
        """Load preprocessed split as list of PyG Data objects."""
        data = self.load_split(output_dir, split_name)
        metadata = data["metadata"]
        node_counts = metadata["node_counts"]
        labels = data["labels"]

        data_list = []
        node_offset = 0

        for i, n_nodes in enumerate(node_counts):
            edge_mask = (data["edge_index"][0] >= node_offset) & \
                        (data["edge_index"][0] < node_offset + n_nodes)

            graph_data = Data(
                x=data["node_feats"][node_offset:node_offset + n_nodes],
                edge_index=data["edge_index"][:, edge_mask] - node_offset,
                edge_attr=data["edge_feats"][edge_mask],
                y=labels[i].unsqueeze(0),
            )
            data_list.append(graph_data)
            node_offset += n_nodes

        return data_list

    def get_morgan_fps(self, output_dir: str | Path, split_name: str) -> np.ndarray:
        """Get Morgan fingerprints as numpy array (for ML models)."""
        data = self.load_split(output_dir, split_name)
        return data["morgan_fps"].numpy()
