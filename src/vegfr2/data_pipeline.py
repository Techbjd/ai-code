"""
Data Pipeline: SMILES → Enriched Graphs → Ready-to-train PyTorch data.

This module converts raw SMILES data into preprocessed, enriched graph
datasets that can be directly loaded by any GNN model.

Pipeline:
    1. Load raw CSV (SMILES + IC50)
    2. Preprocess (validate, deduplicate, label, split)
    3. Convert each molecule to enriched graph:
       [atom(32) + Morgan(2048) + MACCS(166)] = 2246-dim per node
    4. Save as PyTorch tensors (node_feats, edge_index, edge_feats, labels)

Also supports plain graph output (32-dim only) with separate cached
fingerprints for the fused-variant model architecture.

Usage:
    from vegfr2.data_pipeline import VEGFR2Pipeline

    pipeline = VEGFR2Pipeline()
    pipeline.run("data/raw/chembl_vegfr2.csv", output_dir="data/processed")

    # Later, load ready-to-train data:
    train_data = pipeline.load_split("data/processed", "train")
    val_data = pipeline.load_split("data/processed", "val")
    test_data = pipeline.load_split("data/processed", "test")
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
    mol_to_graph_with_fps,
    smiles_to_morgan,
    smiles_to_maccs,
    clear_fp_cache,
    ATOM_FEAT_DIM,
    BOND_FEAT_DIM,
)


class VEGFR2Pipeline:
    """Full data pipeline: SMILES → enriched graphs → PyTorch tensors.

    Args:
        morgan_radius: Morgan fingerprint radius
        morgan_n_bits: Morgan fingerprint bits
        maccs_n_bits: MACCS key bits
        seed: Random seed for splitting
    """

    def __init__(
        self,
        morgan_radius: int = 2,
        morgan_n_bits: int = 2048,
        maccs_n_bits: int = 166,
        seed: int = 42,
    ):
        self.morgan_radius = morgan_radius
        self.morgan_n_bits = morgan_n_bits
        self.maccs_n_bits = maccs_n_bits
        self.seed = seed

        # Computed node feature dimension
        self.node_dim = ATOM_FEAT_DIM + morgan_n_bits + maccs_n_bits  # 32 + 2048 + 166 = 2246

    def smiles_to_enriched_graph(self, smiles: str) -> dict:
        """Convert single SMILES to enriched graph dict.

        Returns dict with:
            node_feats: Tensor [n_nodes, 2246]
            edge_index: Tensor [2, n_edges]
            edge_feats: Tensor [n_edges, 11]
            num_nodes: int
            morgan_fp: Tensor [2048] (molecular-level Morgan fingerprint)
            maccs_fp: Tensor [166] (molecular-level MACCS keys)
        """
        graph = mol_to_graph_with_fps(
            smiles,
            use_morgan=True,
            use_maccs=True,
            morgan_radius=self.morgan_radius,
            morgan_n_bits=self.morgan_n_bits,
            maccs_n_bits=self.maccs_n_bits,
        )

        # Also extract molecular-level fingerprints (for ensemble/ML models)
        morgan_fp = torch.tensor(
            smiles_to_morgan(smiles, radius=self.morgan_radius, n_bits=self.morgan_n_bits),
            dtype=torch.float32,
        )
        maccs_fp = torch.tensor(
            smiles_to_maccs(smiles, n_bits=self.maccs_n_bits),
            dtype=torch.float32,
        )

        return {
            "node_feats": graph["node_feats"],
            "edge_index": graph["edge_index"],
            "edge_feats": graph["edge_feats"],
            "num_nodes": graph["num_nodes"],
            "morgan_fp": morgan_fp,
            "maccs_fp": maccs_fp,
        }

    def process_split(
        self,
        df: pd.DataFrame,
        split_name: str,
        output_dir: Path,
    ) -> dict:
        """Process one split (train/val/test) and save to disk.

        Args:
            df: DataFrame with 'smiles' and 'active' columns
            split_name: "train", "val", or "test"
            output_dir: Directory to save processed data

        Returns:
            Stats dict (n_molecules, n_active, n_inactive, etc.)
        """
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        smiles_list = df["smiles"].tolist()
        labels = df["active"].astype(int).tolist()
        n_molecules = len(smiles_list)

        # Process all molecules
        all_node_feats = []
        all_edge_index = []
        all_edge_feats = []
        all_labels = []
        all_morgan_fps = []
        all_maccs_fps = []
        node_counts = []

        failed = 0
        for i, (smiles, label) in enumerate(zip(smiles_list, labels)):
            try:
                graph = self.smiles_to_enriched_graph(smiles)

                # Offset edge_index for batch
                offset = sum(node_counts)
                edge_index = graph["edge_index"] + offset

                all_node_feats.append(graph["node_feats"])
                all_edge_index.append(edge_index)
                all_edge_feats.append(graph["edge_feats"])
                all_labels.append(label)
                all_morgan_fps.append(graph["morgan_fp"])
                all_maccs_fps.append(graph["maccs_fp"])
                node_counts.append(graph["num_nodes"])

            except Exception as e:
                failed += 1
                continue

            # Progress
            if (i + 1) % 1000 == 0:
                print(f"    [{split_name}] {i + 1}/{n_molecules} processed...")

        # Concatenate into batched tensors
        node_feats = torch.cat(all_node_feats, dim=0)
        edge_index = torch.cat(all_edge_index, dim=1)
        edge_feats = torch.cat(all_edge_feats, dim=0)
        labels = torch.tensor(all_labels, dtype=torch.float32)
        morgan_fps = torch.stack(all_morgan_fps)
        maccs_fps = torch.stack(all_maccs_fps)

        # Node-to-graph mapping
        node_batch = torch.repeat_interleave(
            torch.arange(len(all_labels)), torch.tensor(node_counts)
        )

        # Save tensors
        torch.save(node_feats, split_dir / "node_feats.pt")
        torch.save(edge_index, split_dir / "edge_index.pt")
        torch.save(edge_feats, split_dir / "edge_feats.pt")
        torch.save(labels, split_dir / "labels.pt")
        torch.save(node_batch, split_dir / "node_batch.pt")
        torch.save(morgan_fps, split_dir / "morgan_fps.pt")
        torch.save(maccs_fps, split_dir / "maccs_fps.pt")

        # Save metadata
        metadata = {
            "split": split_name,
            "n_molecules": len(all_labels),
            "n_nodes": int(node_feats.shape[0]),
            "n_edges": int(edge_index.shape[1]),
            "node_dim": int(node_feats.shape[1]),
            "morgan_dim": self.morgan_n_bits,
            "maccs_dim": self.maccs_n_bits,
            "n_active": int(sum(all_labels)),
            "n_inactive": len(all_labels) - int(sum(all_labels)),
            "failed": failed,
            "node_counts": node_counts,
        }
        with open(split_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def process_split_plain(
        self,
        df: pd.DataFrame,
        split_name: str,
        output_dir: Path,
    ) -> dict:
        """Process one split saving PLAIN graphs (32-dim) + cached fingerprints.

        Unlike process_split() which bakes fingerprints into node features,
        this saves:
        - node_feats_plain.pt: [total_nodes, 32] (atom features only)
        - node_feats_enriched.pt: [total_nodes, 2246] (kept for backward compat)
        - morgan_fps.npy: [n_molecules, 2048] (pre-computed, no RDKit at load time)
        - maccs_fps.npy: [n_molecules, 166] (pre-computed, no RDKit at load time)
        - smiles.json: list of SMILES (for cache key lookup)
        - edge_index.pt, edge_feats.pt, labels.pt, node_batch.pt

        Args:
            df: DataFrame with 'smiles' and 'active' columns
            split_name: "train", "val", or "test"
            output_dir: Directory to save processed data

        Returns:
            Stats dict
        """
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        smiles_list = df["smiles"].tolist()
        labels = df["active"].astype(int).tolist()
        n_molecules = len(smiles_list)

        all_node_feats_plain = []
        all_node_feats_enriched = []
        all_edge_index = []
        all_edge_feats = []
        all_labels = []
        all_morgan_fps = []
        all_maccs_fps = []
        all_smiles = []
        node_counts = []

        failed = 0
        for i, (smiles, label) in enumerate(zip(smiles_list, labels)):
            try:
                # Plain graph (32-dim)
                graph_plain = mol_to_graph(smiles)
                # Enriched graph (2246-dim) for backward compat
                graph_enriched = mol_to_graph_with_fps(
                    smiles,
                    use_morgan=True,
                    use_maccs=True,
                    morgan_radius=self.morgan_radius,
                    morgan_n_bits=self.morgan_n_bits,
                    maccs_n_bits=self.maccs_n_bits,
                )
                # Fingerprint tensors (cached via features.py)
                morgan_fp = torch.tensor(
                    smiles_to_morgan(smiles, radius=self.morgan_radius, n_bits=self.morgan_n_bits),
                    dtype=torch.float32,
                )
                maccs_fp = torch.tensor(
                    smiles_to_maccs(smiles, n_bits=self.maccs_n_bits),
                    dtype=torch.float32,
                )

                offset = sum(node_counts)
                edge_index_plain = graph_plain["edge_index"] + offset
                edge_index_enriched = graph_enriched["edge_index"] + offset

                all_node_feats_plain.append(graph_plain["node_feats"])
                all_node_feats_enriched.append(graph_enriched["node_feats"])
                all_edge_index.append(edge_index_plain)
                all_edge_feats.append(graph_plain["edge_feats"])
                all_labels.append(label)
                all_morgan_fps.append(morgan_fp)
                all_maccs_fps.append(maccs_fp)
                all_smiles.append(smiles)
                node_counts.append(graph_plain["num_nodes"])

            except Exception:
                failed += 1
                continue

            if (i + 1) % 1000 == 0:
                print(f"    [{split_name}] {i + 1}/{n_molecules} processed...")

        # Concatenate batched tensors
        node_feats_plain = torch.cat(all_node_feats_plain, dim=0)
        node_feats_enriched = torch.cat(all_node_feats_enriched, dim=0)
        edge_index = torch.cat(all_edge_index, dim=1)
        edge_feats = torch.cat(all_edge_feats, dim=0)
        labels = torch.tensor(all_labels, dtype=torch.float32)
        morgan_fps = torch.stack(all_morgan_fps)
        maccs_fps = torch.stack(all_maccs_fps)

        node_batch = torch.repeat_interleave(
            torch.arange(len(all_labels)), torch.tensor(node_counts)
        )

        # Save tensors
        torch.save(node_feats_plain, split_dir / "node_feats_plain.pt")
        torch.save(node_feats_enriched, split_dir / "node_feats_enriched.pt")
        torch.save(edge_index, split_dir / "edge_index.pt")
        torch.save(edge_feats, split_dir / "edge_feats.pt")
        torch.save(labels, split_dir / "labels.pt")
        torch.save(node_batch, split_dir / "node_batch.pt")
        torch.save(morgan_fps, split_dir / "morgan_fps.pt")
        torch.save(maccs_fps, split_dir / "maccs_fps.pt")

        # Save fingerprints as numpy for fast Dataset loading
        np.save(split_dir / "morgan_fps.npy", morgan_fps.numpy())
        np.save(split_dir / "maccs_fps.npy", maccs_fps.numpy())

        # Save SMILES list (for cache key lookup / inference)
        import json as _json
        with open(split_dir / "smiles.json", "w") as f:
            _json.dump(all_smiles, f)

        # Save metadata
        metadata = {
            "split": split_name,
            "n_molecules": len(all_labels),
            "n_nodes": int(node_feats_plain.shape[0]),
            "n_edges": int(edge_index.shape[1]),
            "node_dim_plain": int(node_feats_plain.shape[1]),
            "node_dim_enriched": int(node_feats_enriched.shape[1]),
            "morgan_dim": self.morgan_n_bits,
            "maccs_dim": self.maccs_n_bits,
            "n_active": int(sum(all_labels)),
            "n_inactive": len(all_labels) - int(sum(all_labels)),
            "failed": failed,
            "node_counts": node_counts,
            "has_plain": True,
            "has_smiles": True,
        }
        with open(split_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Free cache after batch processing
        clear_fp_cache()

        return metadata

    def run(self, raw_csv: str | Path, output_dir: str | Path = "data/processed") -> dict:
        """Run full pipeline: raw CSV → processed data.

        Args:
            raw_csv: Path to raw CSV with 'smiles' and 'ic50_nM' columns
            output_dir: Directory to save processed data

        Returns:
            Summary dict with stats for each split
        """
        output_dir = Path(output_dir)
        print("=" * 60)
        print("VEGFR2 DATA PIPELINE")
        print("=" * 60)

        # Step 1: Load and preprocess
        print("\n[Step 1] Loading and preprocessing raw data...")
        df = load_csv(raw_csv)
        df = preprocess(df)
        print(f"  Total molecules: {len(df)}")
        print(f"  Active: {df['active'].sum()} ({df['active'].mean():.1%})")
        print(f"  Inactive: {(~df['active'].astype(bool)).sum()}")

        # Step 2: Split
        print("\n[Step 2] Splitting into train/val/test...")
        train_df, val_df, test_df = split(df, seed=self.seed)
        print(f"  Train: {len(train_df)}")
        print(f"  Val:   {len(val_df)}")
        print(f"  Test:  {len(test_df)}")

        # Step 3: Process each split
        results = {}
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            print(f"\n[Step 3] Processing {split_name} split ({len(split_df)} molecules)...")
            print(f"  Converting SMILES → enriched graphs ({self.node_dim}-dim per node)...")
            print(f"  [atom(32) + Morgan({self.morgan_n_bits}) + MACCS({self.maccs_n_bits})]")

            stats = self.process_split(split_df, split_name, output_dir)
            results[split_name] = stats

            print(f"  Done: {stats['n_molecules']} molecules, "
                  f"{stats['n_nodes']} nodes, {stats['n_edges']} edges")
            if stats["failed"] > 0:
                print(f"  Warning: {stats['failed']} molecules failed")

        # Step 4: Save global metadata
        global_meta = {
            "node_dim": self.node_dim,
            "morgan_dim": self.morgan_n_bits,
            "maccs_dim": self.maccs_n_bits,
            "morgan_radius": self.morgan_radius,
            "seed": self.seed,
            "splits": results,
        }
        with open(output_dir / "pipeline_metadata.json", "w") as f:
            json.dump(global_meta, f, indent=2)

        # Summary
        total = sum(r["n_molecules"] for r in results.values())
        total_nodes = sum(r["n_nodes"] for r in results.values())
        total_edges = sum(r["n_edges"] for r in results.values())

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Total molecules: {total}")
        print(f"  Total nodes:     {total_nodes:,}")
        print(f"  Total edges:     {total_edges:,}")
        print(f"  Node dimension:  {self.node_dim} (32 atom + {self.morgan_n_bits} morgan + {self.maccs_n_bits} maccs)")
        print(f"  Output:          {output_dir}/")
        print(f"  Files:           train/node_feats.pt, edge_index.pt, edge_feats.pt, labels.pt, ...")
        print()
        print("  To load in training:")
        print(f"    pipeline = VEGFR2Pipeline()")
        print(f"    data = pipeline.load_split('{output_dir}', 'train')")

        return results

    def run_plain(self, raw_csv: str | Path, output_dir: str | Path = "data/processed") -> dict:
        """Run pipeline saving PLAIN graphs (32-dim) + cached fingerprints.

        Same as run() but outputs:
        - node_feats_plain.pt (32-dim atom features)
        - node_feats_enriched.pt (2246-dim, for backward compat)
        - morgan_fps.npy / maccs_fps.npy (pre-computed, no RDKit at load time)
        - smiles.json (SMILES list)

        Use this for the fused-variant architecture where fingerprints
        go through a separate branch, not baked into node features.
        """
        output_dir = Path(output_dir)
        print("=" * 60)
        print("VEGFR2 DATA PLAIN (32-dim graphs + cached fingerprints)")
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
            print(f"  Converting SMILES → plain graphs (32-dim) + cached FPs...")
            stats = self.process_split_plain(split_df, split_name, output_dir)
            results[split_name] = stats
            print(f"  Done: {stats['n_molecules']} molecules, "
                  f"{stats['n_nodes']} nodes, {stats['n_edges']} edges")
            if stats["failed"] > 0:
                print(f"  Warning: {stats['failed']} molecules failed")

        global_meta = {
            "mode": "plain",
            "node_dim_plain": ATOM_FEAT_DIM,
            "node_dim_enriched": self.node_dim,
            "morgan_dim": self.morgan_n_bits,
            "maccs_dim": self.maccs_n_bits,
            "morgan_radius": self.morgan_radius,
            "seed": self.seed,
            "splits": results,
        }
        with open(output_dir / "pipeline_metadata.json", "w") as f:
            json.dump(global_meta, f, indent=2)

        total = sum(r["n_molecules"] for r in results.values())
        total_nodes = sum(r["n_nodes"] for r in results.values())

        print("\n" + "=" * 60)
        print("PLAIN PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Total molecules: {total}")
        print(f"  Total nodes:     {total_nodes:,}")
        print(f"  Node dimension:  {ATOM_FEAT_DIM} (atom features only)")
        print(f"  FPs saved as:    morgan_fps.npy, maccs_fps.npy")
        print(f"  Output:          {output_dir}/")

        return results

    def load_split(self, output_dir: str | Path, split_name: str) -> dict:
        """Load a preprocessed split as a dict of tensors.

        Args:
            output_dir: Pipeline output directory
            split_name: "train", "val", or "test"

        Returns:
            Dict with keys: node_feats, edge_index, edge_feats, labels,
                           node_batch, morgan_fps, maccs_fps, metadata
        """
        output_dir = Path(output_dir)
        split_dir = output_dir / split_name

        if not split_dir.exists():
            raise FileNotFoundError(f"Split not found: {split_dir}")

        node_feats = torch.load(split_dir / "node_feats.pt", weights_only=True)
        edge_index = torch.load(split_dir / "edge_index.pt", weights_only=True)
        edge_feats = torch.load(split_dir / "edge_feats.pt", weights_only=True)
        labels = torch.load(split_dir / "labels.pt", weights_only=True)
        node_batch = torch.load(split_dir / "node_batch.pt", weights_only=True)
        morgan_fps = torch.load(split_dir / "morgan_fps.pt", weights_only=True)
        maccs_fps = torch.load(split_dir / "maccs_fps.pt", weights_only=True)

        with open(split_dir / "metadata.json") as f:
            metadata = json.load(f)

        return {
            "node_feats": node_feats,
            "edge_index": edge_index,
            "edge_feats": edge_feats,
            "labels": labels,
            "node_batch": node_batch,
            "morgan_fps": morgan_fps,
            "maccs_fps": maccs_fps,
            "metadata": metadata,
        }

    def to_pyg_dataset(self, output_dir: str | Path, split_name: str) -> list[Data]:
        """Load preprocessed split as list of PyG Data objects.

        Useful for DataLoader with per-graph batching.

        Args:
            output_dir: Pipeline output directory
            split_name: "train", "val", or "test"

        Returns:
            List of torch_geometric.data.Data objects
        """
        data = self.load_split(output_dir, split_name)
        metadata = data["metadata"]
        node_counts = metadata["node_counts"]
        labels = data["labels"]

        # Split batched tensors into individual graphs
        data_list = []
        node_offset = 0
        edge_offset = 0

        for i, n_nodes in enumerate(node_counts):
            # Find edges belonging to this graph
            edge_mask = (data["edge_index"][0] >= node_offset) & \
                        (data["edge_index"][0] < node_offset + n_nodes)

            graph_node_feats = data["node_feats"][node_offset:node_offset + n_nodes]
            graph_edge_index = data["edge_index"][:, edge_mask] - node_offset
            graph_edge_feats = data["edge_feats"][edge_mask]
            graph_label = labels[i]

            graph_data = Data(
                x=graph_node_feats,
                edge_index=graph_edge_index,
                edge_attr=graph_edge_feats,
                y=graph_label.unsqueeze(0),
            )
            data_list.append(graph_data)

            node_offset += n_nodes

        return data_list

    def get_morgan_fps(self, output_dir: str | Path, split_name: str) -> np.ndarray:
        """Get Morgan fingerprints as numpy array (for ML models)."""
        data = self.load_split(output_dir, split_name)
        return data["morgan_fps"].numpy()

    def get_maccs_fps(self, output_dir: str | Path, split_name: str) -> np.ndarray:
        """Get MACCS fingerprints as numpy array (for ML models)."""
        data = self.load_split(output_dir, split_name)
        return data["maccs_fps"].numpy()

    def get_combined_fps(self, output_dir: str | Path, split_name: str) -> np.ndarray:
        """Get combined Morgan + MACCS fingerprints as numpy array."""
        data = self.load_split(output_dir, split_name)
        return np.hstack([data["morgan_fps"].numpy(), data["maccs_fps"].numpy()])
