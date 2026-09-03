#!/usr/bin/env python
"""Train fused variant models (graph + optional fingerprint branch).

Usage:
    # First run the plain pipeline to generate cached data:
    python -c "
    from vegfr2.data_pipeline import VEGFR2Pipeline
    VEGFR2Pipeline().run_plain('data/raw/chembl_vegfr2.csv', 'data/processed')
    "

    # Then train variants:
    python scripts/train_variants.py --model gin_morgan --data-dir data/processed
    python scripts/train_variants.py --model gin_graph_only --data-dir data/processed
    python scripts/train_variants.py --all --data-dir data/processed
    python scripts/train_variants.py --all --data-dir data/processed --group gin
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from vegfr2.datasets import get_dataset_class
from vegfr2.gnn_pyg import build_pyg_model, train_fused_variant, predict_fused_variant
from vegfr2.metrics import classification_metrics

# ---------------------------------------------------------------------------
# All available variant models
# ---------------------------------------------------------------------------
ALL_VARIANTS = []
_GNN_TYPES = ["gcn", "gat", "gatv2", "gin", "mpnn"]
_FP_TYPES = ["graph_only", "morgan", "maccs", "both"]
for _gnn in _GNN_TYPES:
    for _fp in _FP_TYPES:
        ALL_VARIANTS.append(f"{_gnn}_{_fp}")

# ---------------------------------------------------------------------------
# FP type → Dataset class name mapping
# ---------------------------------------------------------------------------
FP_TYPE_MAP = {
    "graph_only": "none",
    "morgan": "morgan",
    "maccs": "maccs",
    "both": "both",
}


def train_one_variant(
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    hidden: int = 64,
    layers: int = 3,
    heads: int = 4,
    lr: float = 0.001,
    batch_size: int = 128,
    epochs: int = 300,
    patience: int = 25,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """Train a single variant and return metrics."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    # Determine fp_type from model name
    fp_type_key = model_name.split("_", 1)[1] if "_" in model_name else "graph_only"
    dataset_type = FP_TYPE_MAP.get(fp_type_key, "none")

    # Load datasets
    DatasetCls = get_dataset_class(dataset_type)
    train_ds = DatasetCls(data_dir / "train")
    val_ds = DatasetCls(data_dir / "val")
    test_ds = DatasetCls(data_dir / "test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"  Dataset: {dataset_type} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Build model
    model = build_pyg_model(model_name, in_dim=32, hidden=hidden, layers=layers, heads=heads)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model_name} | Parameters: {n_params:,}")

    # Train
    t0 = time.time()
    model = train_fused_variant(
        name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        hidden=hidden,
        layers=layers,
        heads=heads,
        lr=lr,
        epochs=epochs,
        patience=patience,
        seed=seed,
        device=device,
    )
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Evaluate
    val_probs = predict_fused_variant(model, val_loader, device=device)
    val_true = [int(d.y.item()) for d in val_ds]
    val_metrics = classification_metrics(val_true, val_probs.tolist())

    test_probs = predict_fused_variant(model, test_loader, device=device)
    test_true = [int(d.y.item()) for d in test_ds]
    test_metrics = classification_metrics(test_true, test_probs.tolist())

    print(f"  Val  AUC: {val_metrics.get('auc', 0):.4f} | Acc: {val_metrics.get('accuracy', 0):.4f}")
    print(f"  Test AUC: {test_metrics.get('auc', 0):.4f} | Acc: {test_metrics.get('accuracy', 0):.4f}")

    # Save checkpoint
    ckpt_dir = output_dir / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "init_kwargs": model.init_kwargs if hasattr(model, 'init_kwargs') else {},
        "model_name": model_name,
    }, ckpt_dir / "model.pt")

    result = {
        "model": model_name,
        "dataset_type": dataset_type,
        "n_params": n_params,
        "train_time_s": round(train_time, 1),
        "val_auc": val_metrics.get("auc", 0),
        "val_acc": val_metrics.get("accuracy", 0),
        "test_auc": test_metrics.get("auc", 0),
        "test_acc": test_metrics.get("accuracy", 0),
    }

    # Save result
    with open(ckpt_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fused variant models")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Single model to train (e.g., gin_morgan). Choices: {ALL_VARIANTS}")
    parser.add_argument("--all", action="store_true",
                        help="Train all variants")
    parser.add_argument("--group", type=str, default=None,
                        help="Train a group: gin, gat, mpnn, morgan, maccs, both, graph_only")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory with pre-processed data")
    parser.add_argument("--output-dir", type=str, default="runs/variants",
                        help="Output directory for checkpoints")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to train
    models_to_train = []
    if args.all:
        models_to_train = ALL_VARIANTS
    elif args.model:
        models_to_train = [args.model]
    elif args.group:
        group = args.group.lower()
        if group in _GNN_TYPES:
            models_to_train = [f"{group}_{fp}" for fp in _FP_TYPES]
        elif group in _FP_TYPES:
            models_to_train = [f"{gnn}_{group}" for gnn in _GNN_TYPES]
        else:
            print(f"Unknown group: {group}")
            return
    else:
        # Default: train gin variants
        models_to_train = [f"gin_{fp}" for fp in _FP_TYPES]

    print(f"Models to train: {models_to_train}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")

    # Check if plain data exists
    if not (data_dir / "train" / "node_feats_plain.pt").exists():
        print("\nPlain data not found. Running data pipeline first...")
        from vegfr2.data_pipeline import VEGFR2Pipeline
        pipeline = VEGFR2Pipeline()
        # Find raw CSV
        raw_csv = data_dir / "raw" / "chembl_vegfr2.csv"
        if not raw_csv.exists():
            raw_csv = "data/raw/chembl_vegfr2.csv"
        pipeline.run_plain(raw_csv, data_dir)

    # Train all
    results = []
    for model_name in models_to_train:
        try:
            result = train_one_variant(
                model_name=model_name,
                data_dir=data_dir,
                output_dir=output_dir,
                hidden=args.hidden,
                layers=args.layers,
                heads=args.heads,
                lr=args.lr,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                seed=args.seed,
                device=args.device,
            )
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"model": model_name, "error": str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Dataset':<10} {'Val AUC':<10} {'Test AUC':<10} {'Time':<10}")
    print("-" * 65)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<25} {'ERROR':<10} {r['error']}")
        else:
            print(f"{r['model']:<25} {r['dataset_type']:<10} {r['val_auc']:<10.4f} {r['test_auc']:<10.4f} {r['train_time_s']:<10.1f}s")

    # Save summary
    with open(output_dir / "results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/results_summary.json")


if __name__ == "__main__":
    main()
