#!/usr/bin/env python3
"""
Screen large compound libraries (COCONUT, ZINC, etc.) with pre-trained VEGFR2 model.

Usage:
    # Screen COCONUT database
    python scripts/screen_library.py --input data/coconut.csv --output hits.csv

    # With custom threshold
    python scripts/screen_library.py --input data/coconut.csv --output hits.csv --threshold 0.8

    # Using pre-trained model
    python scripts/screen_library.py --input data/coconut.csv --output hits.csv --model checkpoints/finetuned_gin_pretrained.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_pretrained_model(model_path: str, device: torch.device):
    """Load pre-trained/fine-tuned model."""
    from vegfr2.gnn_pyg import build_pyg_model

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model_name = checkpoint.get("model_name", "gin")
    model = build_pyg_model(
        model_name,
        in_dim=2246,
        hidden=128,
        layers=3,
        heads=8,
        dropout=0.3,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    print(f"Loaded {model_name} model from {model_path}")
    return model, model_name


def forward_model(model, model_name, batch):
    """Forward pass for different model types."""
    if model_name == "mpnn":
        return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    elif model_name == "graph_transformer":
        return model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
    else:
        return model(batch.x, batch.edge_index, batch.batch)


def screen_library(
    model,
    model_name: str,
    library_df: pd.DataFrame,
    batch_size: int = 256,
    device: torch.device = torch.device("cuda"),
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Screen a library of SMILES for VEGFR2 activity.

    Args:
        model: Trained GNN model
        model_name: Model type (gin, gcn, etc.)
        library_df: DataFrame with 'smiles' column
        batch_size: Batch size for inference
        device: Device to use
        threshold: Probability threshold for hits

    Returns:
        DataFrame with probability and hit columns
    """
    from vegfr2.features import mol_to_graph_with_fps

    print(f"\nScreening {len(library_df)} molecules...")
    print(f"  Threshold: {threshold}")
    print(f"  Batch size: {batch_size}")

    start_time = time.time()
    probs_all = []
    valid_count = 0
    invalid_count = 0

    # Process in batches
    for i in range(0, len(library_df), batch_size):
        batch_df = library_df.iloc[i : i + batch_size]
        data_list = []

        for s in batch_df["smiles"]:
            try:
                g = mol_to_graph_with_fps(s, use_morgan=True, use_maccs=True)
                data = Data(
                    x=g["node_feats"],
                    edge_index=g["edge_index"],
                    edge_attr=g["edge_feats"],
                    y=torch.tensor([0], dtype=torch.float32),  # dummy
                )
                data_list.append(data)
                valid_count += 1
            except Exception as e:
                probs_all.append(np.nan)
                invalid_count += 1

        if data_list:
            loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    logits = forward_model(model, model_name, batch)
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                    if isinstance(probs, np.floating):
                        probs = [probs]
                    probs_all.extend(probs)

        # Progress update
        processed = min(i + batch_size, len(library_df))
        if processed % 1000 == 0 or processed == len(library_df):
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  Processed {processed}/{len(library_df)} ({rate:.0f} mol/s)")

    elapsed = time.time() - start_time
    print(f"\nScreening complete!")
    print(f"  Valid molecules: {valid_count}")
    print(f"  Invalid molecules: {invalid_count}")
    print(f"  Time: {elapsed:.1f}s ({valid_count/elapsed:.0f} mol/s)")

    # Create output DataFrame
    out = library_df.copy()
    out["probability"] = probs_all
    out["hit"] = out["probability"] >= threshold
    out = out.sort_values("probability", ascending=False)

    n_hits = int(out["hit"].sum())
    print(f"  Hits: {n_hits} (threshold={threshold})")

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Screen compound library with pre-trained VEGFR2 model"
    )
    parser.add_argument(
        "--model",
        default="checkpoints/finetuned_gin_pretrained.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input library CSV (must have 'smiles' column)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output hits CSV",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for hit (default: 0.5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference",
    )
    args = parser.parse_args()

    # Check device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU - using CPU")
    print(f"Device: {device}")

    # Load input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    library_df = pd.read_csv(input_path)
    if "smiles" not in library_df.columns:
        print("Input CSV must contain a 'smiles' column", file=sys.stderr)
        return 1

    print(f"Loaded {len(library_df)} molecules from {input_path}")

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        print("Run colab_pretrain_pipeline.py first to train a model", file=sys.stderr)
        return 1

    model, model_name = load_pretrained_model(str(model_path), device)

    # Screen
    out_df = screen_library(
        model,
        model_name,
        library_df,
        batch_size=args.batch_size,
        device=device,
        threshold=args.threshold,
    )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\nResults saved to: {out_path}")
    print(f"  Top 10 hits:")
    top_hits = out_df.head(10)
    for _, row in top_hits.iterrows():
        print(f"    {row['smiles']:<30} {row['probability']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
