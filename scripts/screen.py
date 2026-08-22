"""Screen an external SMILES library with a trained VEGFR2 model (GPU-only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vegfr2.device import get_device
from vegfr2.features import mol_to_graph, smiles_to_morgan, collate_graphs
from vegfr2.gnn_models import load_checkpoint
from vegfr2.ml_models import load_ml_model, predict_ml_model


def screen_gnn(model_path: str, library_df: pd.DataFrame, batch_size: int, device: torch.device, threshold: float) -> pd.DataFrame:
    model = load_checkpoint(model_path, device=device)
    model.eval()

    probs_all = []
    with torch.no_grad():
        for i in range(0, len(library_df), batch_size):
            batch_df = library_df.iloc[i : i + batch_size]
            graphs = []
            valid_idx = []
            for idx, row in batch_df.iterrows():
                try:
                    g = mol_to_graph(row["smiles"])
                    graphs.append(g)
                    valid_idx.append(idx)
                except ValueError:
                    probs_all.append(np.nan)
            if graphs:
                batch = collate_graphs(graphs, [0] * len(graphs))
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch, device)
                batch_probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                for j, idx in enumerate(valid_idx):
                    probs_all[idx] = batch_probs[j]

    out = library_df.copy()
    out["probability"] = probs_all
    out["hit"] = out["probability"] >= threshold
    return out


def screen_ml(model_path: str, library_df: pd.DataFrame, batch_size: int, threshold: float) -> pd.DataFrame:
    estimator = load_ml_model(model_path)

    radius = 2
    n_bits = 2048

    probs_all = []
    for i in range(0, len(library_df), batch_size):
        batch_df = library_df.iloc[i : i + batch_size]
        fps = np.vstack([smiles_to_morgan(s, radius=radius, n_bits=n_bits) for s in batch_df["smiles"]])
        probs = predict_ml_model(estimator, fps)
        probs_all.extend(probs)

    out = library_df.copy()
    out["probability"] = probs_all
    out["hit"] = out["probability"] >= threshold
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Screen compound library with trained VEGFR2 model")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pt for GNN, .pkl for ML)")
    parser.add_argument("--input", required=True, help="Input library CSV (must have 'smiles' column)")
    parser.add_argument("--output", required=True, help="Output hits CSV")
    parser.add_argument("--threshold", type=float, default=0.9, help="Probability threshold for hit")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference")
    args = parser.parse_args()

    # GPU enforcement at entrypoint
    device = get_device()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    library_df = pd.read_csv(input_path)
    if "smiles" not in library_df.columns:
        print("Input CSV must contain a 'smiles' column", file=sys.stderr)
        return 1

    model_path = args.model
    if model_path.endswith(".pt"):
        out_df = screen_gnn(model_path, library_df, args.batch_size, device, args.threshold)
    elif model_path.endswith(".pkl"):
        out_df = screen_ml(model_path, library_df, args.batch_size, args.threshold)
    else:
        print("Model file must end with .pt (GNN) or .pkl (ML)", file=sys.stderr)
        return 1

    out_df = out_df.sort_values("probability", ascending=False)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    n_hits = int(out_df["hit"].sum())
    print(f"Wrote {len(out_df)} rows, {n_hits} hits >= {args.threshold} to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())