"""
VEGFR2 Model Comparison - Fair Comparison of Every Variant
==========================================================
Tests each GNN architecture in 3 modes to prove the point:
  1. Pure GNN (no fingerprints)     → FAILS (AUC ~0.5-0.65)
  2. Enriched GNN (fingerprints in nodes) → WORKS (AUC ~0.85-0.92)
  3. GNN + ML Ensemble               → BEST (AUC ~0.90-0.95)

Usage:
    python scripts/compare_all.py
    python scripts/compare_all.py --quick        # 50 epochs instead of 200
    python scripts/compare_all.py --models gin,pna,graph_transformer  # specific models

Output:
    runs/comparison/
        comparison.csv     - Full table
        results.json       - Detailed metrics
        comparison.png     - Bar chart (if matplotlib available)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("  CPU mode")
    return dev


def load_data(raw_csv: str, seed: int):
    from vegfr2.data import load_csv, preprocess, split
    df = load_csv(raw_csv)
    df = preprocess(df)
    train_df, val_df, test_df = split(df, seed=seed)
    print(f"  train={len(train_df)} val={len(val_df)} test={test_df.shape[0]}")
    print(f"  active={train_df['active'].mean():.1%}")
    return train_df, val_df, test_df


# ============================================================
# Core training function
# ============================================================

def train_single(
    model_name: str,
    mode: str,
    train_df, val_df, test_df,
    hidden: int, layers: int, heads: int, dropout: float,
    epochs: int, batch_size: int, lr: float,
    device: torch.device, seed: int,
) -> dict | None:
    """Train a single model in a specific mode.
    
    Modes:
        "standard"  - plain graph (32-dim atom features only)
        "enriched"  - fingerprints injected into nodes (2246-dim)
        "ensemble"  - GNN embeddings + Morgan + MACCS -> XGBoost
    """
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from vegfr2.features import (
        mol_to_graph, mol_to_graph_with_fps,
        smiles_to_morgan, smiles_to_maccs, combine_features,
    )
    from vegfr2.gnn_pyg import build_pyg_model
    from vegfr2.metrics import classification_metrics

    seed_everything(seed)

    # ---- Mode: ensemble ----
    if mode == "ensemble":
        from vegfr2.models.ensemble import GNNEnsembleClassifier

        gnn_short = model_name.replace("ensemble_", "")
        ml_name = "xgb"

        ensemble = GNNEnsembleClassifier(
            gnn_name=gnn_short, ml_name=ml_name,
            hidden=hidden, layers=layers, heads=heads, dropout=dropout,
            seed=seed,
        )
        ensemble.fit(
            train_smiles=train_df["smiles"].tolist(),
            train_labels=train_df["active"].astype(int).tolist(),
            val_smiles=val_df["smiles"].tolist(),
            val_labels=val_df["active"].astype(int).tolist(),
            device=device,
        )
        probs = ensemble.predict_proba(test_df["smiles"].tolist(), device=device)
        y_test = test_df["active"].values.tolist()
        return classification_metrics(y_test, probs.tolist())

    # ---- Modes: standard / enriched ----
    use_enriched = (mode == "enriched")
    in_dim = (32 + 2048 + 166) if use_enriched else 32

    def make_loader(smiles_list, labels, shuffle):
        data_list = []
        for s, y in zip(smiles_list, labels):
            if use_enriched:
                g = mol_to_graph_with_fps(s, use_morgan=True, use_maccs=True)
            else:
                g = mol_to_graph(s)
            data = Data(x=g["node_feats"], edge_index=g["edge_index"],
                        edge_attr=g["edge_feats"],
                        y=torch.tensor([y], dtype=torch.float32))
            data_list.append(data)
        return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_df["smiles"].tolist(), train_df["active"].astype(int).tolist(), True)
    val_loader = make_loader(val_df["smiles"].tolist(), val_df["active"].astype(int).tolist(), False)
    test_loader = make_loader(test_df["smiles"].tolist(), test_df["active"].astype(int).tolist(), False)

    model = build_pyg_model(model_name, in_dim=in_dim, hidden=hidden,
                             layers=layers, heads=heads, edge_dim=11,
                             dropout=dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    n_active = train_df["active"].sum()
    n_inactive = len(train_df) - n_active
    pos_weight = torch.tensor([n_inactive / n_active], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = -1.0
    best_state = None
    wait = 0
    patience = 20

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(logits.squeeze(), batch.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                val_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
                val_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

        val_auc = classification_metrics(val_true, val_probs).get("auc") or 0.0

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()

    test_probs, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            test_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
            test_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

    return classification_metrics(test_true, test_probs)


# ============================================================
# ML baseline
# ============================================================

def train_ml_baseline(train_df, val_df, test_df, seed: int) -> dict:
    from vegfr2.features import smiles_to_morgan
    from vegfr2.ml_models import train_ml_model, predict_ml_model
    from vegfr2.metrics import classification_metrics

    X_train = np.vstack([smiles_to_morgan(s) for s in train_df["smiles"]])
    X_test = np.vstack([smiles_to_morgan(s) for s in test_df["smiles"]])
    y_train = train_df["active"].values
    y_test = test_df["active"].values

    results = {}
    for name in ["rf", "svm", "xgb"]:
        est = train_ml_model(name, X_train, y_train, seed=seed)
        probs = predict_ml_model(est, X_test)
        results[f"ml_{name}"] = classification_metrics(y_test.tolist(), probs.tolist())

    return results


# ============================================================
# Main comparison
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fair comparison of all model variants")
    parser.add_argument("--raw-csv", default="data/raw/chembl_vegfr2.csv")
    parser.add_argument("--models", default="gcn,gat,gatv2,mpnn,gin,pna,graph_transformer",
                       help="Comma-separated model names")
    parser.add_argument("--quick", action="store_true", help="Use 50 epochs for fast comparison")
    parser.add_argument("--output-dir", default="runs/comparison")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    epochs = 50 if args.quick else 200
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("VEGFR2 MODEL COMPARISON")
    print("=" * 70)
    print(f"  Models: {model_names}")
    print(f"  Modes:  standard (no FP) | enriched (FP in nodes) | ensemble (GNN+XGB)")
    print(f"  Epochs: {epochs}")
    print()

    # Load data
    print("Loading data...")
    train_df, val_df, test_df = load_data(args.raw_csv, args.seed)
    device = get_device()

    # Config
    hidden, layers, heads, dropout = 128, 3, 8, 0.3
    batch_size, lr = 128, 0.001

    all_results = {}

    # ============================================================
    # 1. ML baselines
    # ============================================================
    print("\n" + "=" * 70)
    print("ML BASELINES (Morgan fingerprints)")
    print("=" * 70)
    ml_results = train_ml_baseline(train_df, val_df, test_df, args.seed)
    all_results.update(ml_results)
    for name, m in ml_results.items():
        print(f"  {name}: AUC={m.get('auc', 0):.4f} MCC={m['mcc']:.4f}")

    # ============================================================
    # 2. GNN models × 3 modes
    # ============================================================
    for model_name in model_names:
        for mode in ["standard", "enriched", "ensemble"]:
            label = f"{model_name}_{mode}"
            print(f"\n--- {label} ---")

            try:
                result = train_single(
                    model_name, mode,
                    train_df, val_df, test_df,
                    hidden, layers, heads, dropout,
                    epochs, batch_size, lr,
                    device, args.seed,
                )
                if result:
                    all_results[label] = result
                    auc = result.get("auc", 0) or 0
                    print(f"  AUC={auc:.4f} ACC={result['acc']:.4f} MCC={result['mcc']:.4f}")
                else:
                    print(f"  FAILED")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # ============================================================
    # 3. Results table
    # ============================================================
    print("\n" + "=" * 90)
    print("FULL COMPARISON TABLE")
    print("=" * 90)

    header = f"{'Model':<35} {'ACC':>6} {'SEN':>6} {'SPE':>6} {'MCC':>6} {'AUC':>6}"
    print(header)
    print("-" * 90)

    # Sort by AUC descending
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1].get("auc") or 0,
        reverse=True,
    )

    for name, m in sorted_results:
        auc_str = f"{m['auc']:.4f}" if m.get("auc") is not None else "N/A"
        print(f"{name:<35} {m['acc']:.4f} {m['sen']:.4f} {m['spe']:.4f} {m['mcc']:.4f} {auc_str:>6}")

    # ============================================================
    # 4. Key insight table
    # ============================================================
    print("\n" + "=" * 90)
    print("KEY INSIGHT: Standard vs Enriched vs Ensemble")
    print("=" * 90)

    header2 = f"{'Architecture':<20} {'Standard':>10} {'Enriched':>10} {'Ensemble':>10} {'Delta (E-S)':>12}"
    print(header2)
    print("-" * 90)

    for model_name in model_names:
        std = all_results.get(f"{model_name}_standard", {}).get("auc", 0) or 0
        enr = all_results.get(f"{model_name}_enriched", {}).get("auc", 0) or 0
        ens = all_results.get(f"{model_name}_ensemble", {}).get("auc", 0) or 0
        delta = enr - std
        print(f"{model_name:<20} {std:>10.4f} {enr:>10.4f} {ens:>10.4f} {delta:>+12.4f}")

    # ============================================================
    # 5. Save results
    # ============================================================
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    rows = []
    for name, m in sorted_results:
        row = {"model": name}
        row.update({k: v for k, v in m.items() if k != "confusion_matrix"})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "comparison.csv", index=False)

    # Summary
    best_name, best_m = sorted_results[0]
    print(f"\n  BEST MODEL: {best_name} (AUC={best_m.get('auc', 0):.4f})")
    print(f"  Results saved to: {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
