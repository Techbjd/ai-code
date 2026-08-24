"""
VEGFR2 Advanced Training Pipeline
==================================
Trains ALL models (ML + old GNN + new GNN) and compares them.

Usage:
    # Train everything on actual data
    python scripts/train_all.py

    # Train only new models (GIN/PNA/Transformer)
    python scripts/train_all.py --group advanced

    # Train only with enriched graphs
    python scripts/train_all.py --enriched

    # Train specific model
    python scripts/train_all.py --model gin

    # Train and run HPO
    python scripts/train_all.py --model gin --hpo --hpo-trials 30

    # Use pre-split data
    python scripts/train_all.py --train-csv data/train.csv --val-csv data/val.csv --test-csv data/test.csv

    # Compare all models
    python scripts/train_all.py --compare

Output:
    runs/advanced/
        results.json          - All metrics
        best/                 - Best model checkpoints
        comparison.csv        - Side-by-side comparison table
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

# ============================================================
# Configuration
# ============================================================

DEFAULT_CONFIG = {
    "seed": 42,
    "gnn": {
        "hidden": 128,
        "layers": 3,
        "heads": 8,
        "batch": 128,
        "lr": 0.001,
        "epochs": 300,
        "patience": 25,
        "dropout": 0.3,
    },
    "fingerprint": {
        "radius": 2,
        "n_bits": 2048,
    },
    "hpo": {
        "n_trials": 50,
    },
}

# Model groups
ALL_ML_MODELS = ["rf", "svm", "xgb"]
ALL_OLD_GNN = ["gcn", "gat", "mpnn"]
ALL_ADVANCED_GNN = ["gin", "pna", "graph_transformer", "gatv2"]
ALL_ENSEMBLES = ["ensemble_gin_xgb", "ensemble_pna_xgb", "ensemble_gin_rf"]
ALL_MODELS = ALL_ML_MODELS + ALL_OLD_GNN + ALL_ADVANCED_GNN + ALL_ENSEMBLES


# ============================================================
# Utility functions
# ============================================================

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
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        dev = torch.device("cpu")
        print("  GPU not available, using CPU")
    return dev


def timer(func):
    """Decorator to time function calls."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper


# ============================================================
# Data handling
# ============================================================

def load_and_preprocess(
    raw_csv: str | Path,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV, preprocess, and split into train/val/test."""
    from vegfr2.data import load_csv, preprocess, split

    df = load_csv(raw_csv)
    df = preprocess(df)
    train_df, val_df, test_df = split(df, seed=seed)
    print(f"  Data loaded: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"  Class balance: {train_df['active'].mean():.1%} active")
    return train_df, val_df, test_df


# ============================================================
# Training functions for each model type
# ============================================================

def train_ml_model(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
) -> dict:
    """Train a traditional ML model (RF, SVM, XGBoost)."""
    from vegfr2.features import smiles_to_morgan
    from vegfr2.ml_models import train_ml_model as _train_ml, predict_ml_model, save_ml_model
    from vegfr2.metrics import classification_metrics

    seed_everything(cfg["seed"])

    radius = cfg["fingerprint"]["radius"]
    n_bits = cfg["fingerprint"]["n_bits"]

    def fps(df):
        return np.vstack([smiles_to_morgan(s, radius=radius, n_bits=n_bits) for s in df["smiles"]])

    X_train, y_train = fps(train_df), train_df["active"].values
    X_test, y_test = fps(test_df), test_df["active"].values

    estimator = _train_ml(name, X_train, y_train, seed=cfg["seed"])
    probs = predict_ml_model(estimator, X_test)

    model_path = output_dir / f"{name}/model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_ml_model(estimator, model_path)

    return classification_metrics(y_test.tolist(), probs.tolist())


def train_gnn_model(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Train a GNN model with enriched graphs (Morgan + MACCS + atom features)."""
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    from vegfr2.features import mol_to_graph_with_fps
    from vegfr2.gnn_pyg import build_pyg_model, save_checkpoint
    from vegfr2.metrics import classification_metrics

    seed_everything(cfg["seed"])

    gnn_cfg = cfg["gnn"]

    # Always enriched: 32 atom + 2048 morgan + 166 maccs = 2246
    in_dim = 2246
    print(f"  Enriched graph: {in_dim}-dim input (32 atom + 2048 morgan + 166 maccs)")

    # Build dataset (always enriched)
    def make_data(smiles_list, labels):
        data_list = []
        for s, y in zip(smiles_list, labels):
            g = mol_to_graph_with_fps(s, use_morgan=True, use_maccs=True)
            data = Data(x=g["node_feats"], edge_index=g["edge_index"],
                       edge_attr=g["edge_feats"],
                       y=torch.tensor([y], dtype=torch.float32))
            data_list.append(data)
        return data_list

    train_data = make_data(train_df["smiles"].tolist(), train_df["active"].astype(int).tolist())
    val_data = make_data(val_df["smiles"].tolist(), val_df["active"].astype(int).tolist())
    test_data = make_data(test_df["smiles"].tolist(), test_df["active"].astype(int).tolist())

    train_loader = DataLoader(train_data, batch_size=gnn_cfg["batch"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=gnn_cfg["batch"], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=gnn_cfg["batch"], shuffle=False)

    # Build model
    model = build_pyg_model(
        name, in_dim=in_dim, hidden=gnn_cfg["hidden"],
        layers=gnn_cfg["layers"], heads=gnn_cfg["heads"],
        edge_dim=11, dropout=gnn_cfg["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {name} ({n_params:,} params)")

    # Training setup
    opt = torch.optim.AdamW(model.parameters(), lr=gnn_cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=gnn_cfg["epochs"], eta_min=1e-6)

    n_active = train_df["active"].sum()
    n_inactive = len(train_df) - n_active
    pos_weight = torch.tensor([n_inactive / n_active], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop
    best_auc = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, gnn_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(logits.squeeze(), batch.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.y.shape[0]
        scheduler.step()

        # Validation
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
            if wait >= gnn_cfg["patience"]:
                print(f"  Early stopping at epoch {epoch} (best AUC={best_auc:.4f})")
                break

        if epoch % 25 == 0:
            print(f"  Epoch {epoch:3d} | loss={total_loss/len(train_df):.4f} val_AUC={val_auc:.4f}")

    # Load best and evaluate on test
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

    # Save checkpoint
    ckpt_dir = output_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, ckpt_dir / "best.pt")

    return classification_metrics(test_true, test_probs)


def train_ensemble_model(
    gnn_name: str,
    ml_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Train GNN + ML ensemble."""
    from vegfr2.models.ensemble import GNNEnsembleClassifier
    from vegfr2.metrics import classification_metrics

    seed_everything(cfg["seed"])

    ensemble_name = f"ensemble_{gnn_name}_{ml_name}"
    print(f"  Ensemble: GNN={gnn_name} + ML={ml_name}")

    ensemble = GNNEnsembleClassifier(
        gnn_name=gnn_name,
        ml_name=ml_name,
        hidden=cfg["gnn"]["hidden"],
        layers=cfg["gnn"]["layers"],
        heads=cfg["gnn"]["heads"],
        dropout=cfg["gnn"]["dropout"],
        seed=cfg["seed"],
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

    # Save
    save_dir = output_dir / ensemble_name
    save_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(str(save_dir / "ensemble.pkl"))

    return classification_metrics(y_test, probs.tolist())


# ============================================================
# HPO (Hyperparameter Optimization)
# ============================================================

def run_hpo(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    n_trials: int = 30,
) -> dict:
    """Run Optuna HPO for a GNN model."""
    try:
        import optuna
    except ImportError:
        print("  optuna not installed, skipping HPO")
        return {}

    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from vegfr2.features import mol_to_graph_with_fps
    from vegfr2.gnn_pyg import build_pyg_model
    from vegfr2.metrics import classification_metrics

    fp_cfg = cfg["fingerprint"]
    in_dim = 32 + fp_cfg["n_bits"] + 166

    def make_data(smiles_list, labels):
        data_list = []
        for s, y in zip(smiles_list, labels):
            g = mol_to_graph_with_fps(s, use_morgan=True, use_maccs=True,
                                      morgan_n_bits=fp_cfg["n_bits"])
            data = Data(x=g["node_feats"], edge_index=g["edge_index"],
                       edge_attr=g["edge_feats"],
                       y=torch.tensor([y], dtype=torch.float32))
            data_list.append(data)
        return data_list

    train_data = make_data(train_df["smiles"].tolist(), train_df["active"].astype(int).tolist())
    val_data = make_data(val_df["smiles"].tolist(), val_df["active"].astype(int).tolist())
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

    def objective(trial):
        hidden = trial.suggest_categorical("hidden", [64, 128, 256])
        layers = trial.suggest_int("layers", 2, 5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        heads = trial.suggest_categorical("heads", [4, 8]) if model_name in ["gat", "gatv2", "graph_transformer"] else 8

        model = build_pyg_model(model_name, in_dim=in_dim, hidden=hidden,
                                layers=layers, heads=heads, edge_dim=11, dropout=dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(50):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(logits.squeeze(), batch.y)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                val_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
                val_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

        return classification_metrics(val_true, val_probs).get("auc") or 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.optimize(objective, n_trials=n_trials)

    print(f"  Best params: {study.best_params}")
    print(f"  Best val AUC: {study.best_value:.4f}")

    return study.best_params


# ============================================================
# Results display
# ============================================================

def print_results_table(results: dict[str, dict]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    header = f"{'Model':<25} {'ACC':>6} {'SEN':>6} {'SPE':>6} {'MCC':>6} {'AUC':>6}"
    print(header)
    print("-" * 80)

    for name, m in sorted(results.items(), key=lambda x: x[1].get("auc") or 0, reverse=True):
        auc_str = f"{m['auc']:.4f}" if m.get("auc") is not None else "N/A"
        print(f"{name:<25} {m['acc']:.4f} {m['sen']:.4f} {m['spe']:.4f} {m['mcc']:.4f} {auc_str:>6}")

    # Find best model
    best_name = max(results.keys(), key=lambda k: results[k].get("auc") or 0)
    best_auc = results[best_name].get("auc", 0)
    print("-" * 80)
    print(f"  BEST MODEL: {best_name} (AUC={best_auc:.4f})")
    print("=" * 80)


def save_results(results: dict, output_dir: Path) -> None:
    """Save results to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV
    rows = []
    for name, m in results.items():
        row = {"model": name}
        row.update(m)
        if "confusion_matrix" in row:
            cm = row.pop("confusion_matrix")
            row.update({f"cm_{k}": v for k, v in cm.items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "comparison.csv", index=False)
    print(f"\n  Results saved to {output_dir}")


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="VEGFR2 Advanced Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_all.py                        # Train everything
  python scripts/train_all.py --group ml             # Only ML models
  python scripts/train_all.py --group advanced       # Only GIN/PNA/Transformer
  python scripts/train_all.py --model gin --enriched # Single model with enriched graphs
  python scripts/train_all.py --model gin --hpo      # With hyperparameter optimization
  python scripts/train_all.py --compare              # Compare all models
        """,
    )

    # Data options
    parser.add_argument("--train-csv", help="Pre-processed train CSV")
    parser.add_argument("--val-csv", help="Pre-processed val CSV")
    parser.add_argument("--test-csv", help="Pre-processed test CSV")
    parser.add_argument("--raw-csv", default="data/raw/chembl_vegfr2.csv",
                       help="Raw CSV for preprocessing (default: data/raw/chembl_vegfr2.csv)")

    # Model selection
    parser.add_argument("--model", choices=ALL_MODELS, help="Train a specific model")
    parser.add_argument("--group", choices=["ml", "old_gnn", "advanced", "ensemble", "all"],
                       default="all", help="Train a group of models")

    # Training options
    parser.add_argument("--config", default=None, help="YAML config file (optional)")
    parser.add_argument("--hpo", action="store_true", help="Run HPO for GNN models")
    parser.add_argument("--hpo-trials", type=int, default=30, help="Number of HPO trials")
    parser.add_argument("--epochs", type=int, help="Override max epochs")
    parser.add_argument("--hidden", type=int, help="Override hidden dimension")
    parser.add_argument("--layers", type=int, help="Override number of layers")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--dropout", type=float, help="Override dropout")

    # Output
    parser.add_argument("--output-dir", default="runs/advanced", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load config
    if args.config:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text())
    else:
        cfg = DEFAULT_CONFIG.copy()

    # Override config from CLI
    cfg["seed"] = args.seed
    if args.epochs:
        cfg["gnn"]["epochs"] = args.epochs
    if args.hidden:
        cfg["gnn"]["hidden"] = args.hidden
    if args.layers:
        cfg["gnn"]["layers"] = args.layers
    if args.lr:
        cfg["gnn"]["lr"] = args.lr
    if args.batch_size:
        cfg["gnn"]["batch"] = args.batch_size
    if args.dropout:
        cfg["gnn"]["dropout"] = args.dropout

    output_dir = Path(args.output_dir)
    seed_everything(cfg["seed"])

    # ============================================================
    # Load data
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)

    if args.train_csv and args.val_csv and args.test_csv:
        train_df = pd.read_csv(args.train_csv)
        val_df = pd.read_csv(args.val_csv)
        test_df = pd.read_csv(args.test_csv)
        print(f"  Loaded pre-split data: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    else:
        raw_path = Path(args.raw_csv)
        if not raw_path.exists():
            print(f"  ERROR: Raw data not found at {raw_path}")
            print("  Run: python scripts/download_data.py first")
            return 1
        train_df, val_df, test_df = load_and_preprocess(raw_path, cfg["seed"])

    # Device
    print("\n" + "=" * 60)
    print("STEP 2: Setting up Device")
    print("=" * 60)
    device = get_device()

    # ============================================================
    # Determine models to train
    # ============================================================
    models_to_train = []

    if args.model:
        models_to_train = [args.model]
    else:
        if args.group in ["ml", "all"]:
            models_to_train.extend(ALL_ML_MODELS)
        if args.group in ["old_gnn", "all"]:
            models_to_train.extend(ALL_OLD_GNN)
        if args.group in ["advanced", "all"]:
            models_to_train.extend(ALL_ADVANCED_GNN)
        if args.group in ["ensemble", "all"]:
            models_to_train.extend(ALL_ENSEMBLES)

    print(f"\n  Models to train: {models_to_train}")
    print(f"  All models use enriched graphs (Morgan + MACCS + atom features)")
    print(f"  Output: {output_dir}")

    # ============================================================
    # Train models
    # ============================================================
    results = {}
    total_start = time.time()

    for i, model_name in enumerate(models_to_train, 1):
        print(f"\n{'=' * 60}")
        print(f"STEP 3: Training [{i}/{len(models_to_train)}] - {model_name.upper()}")
        print(f"{'=' * 60}")

        model_start = time.time()

        try:
            if model_name in ALL_ML_MODELS:
                metrics, elapsed = timer(train_ml_model)(
                    model_name, train_df, val_df, test_df, cfg, output_dir
                )
                results[model_name] = metrics

            elif model_name in ALL_OLD_GNN + ALL_ADVANCED_GNN:
                # Run HPO if requested
                if args.hpo and model_name in ALL_ADVANCED_GNN:
                    print(f"  Running HPO ({args.hpo_trials} trials)...")
                    best_params = run_hpo(
                        model_name, train_df, val_df, cfg, device, args.hpo_trials
                    )
                    # Update config with best params
                    if best_params:
                        if "hidden" in best_params:
                            cfg["gnn"]["hidden"] = best_params["hidden"]
                        if "layers" in best_params:
                            cfg["gnn"]["layers"] = best_params["layers"]
                        if "lr" in best_params:
                            cfg["gnn"]["lr"] = best_params["lr"]
                        if "dropout" in best_params:
                            cfg["gnn"]["dropout"] = best_params["dropout"]

                metrics, elapsed = timer(train_gnn_model)(
                    model_name, train_df, val_df, test_df, cfg, output_dir,
                    device
                )
                results[model_name] = metrics

            elif model_name.startswith("ensemble_"):
                parts = model_name.split("_")
                gnn_name = parts[1]
                ml_name = parts[2]
                metrics, elapsed = timer(train_ensemble_model)(
                    gnn_name, ml_name, train_df, val_df, test_df, cfg, output_dir, device
                )
                results[model_name] = metrics

            else:
                print(f"  Unknown model: {model_name}, skipping")
                continue

            auc = metrics.get("auc", 0) or 0
            print(f"  DONE in {elapsed:.1f}s | AUC={auc:.4f} ACC={metrics['acc']:.4f} MCC={metrics['mcc']:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {"error": str(e)}

    total_elapsed = time.time() - total_start

    # ============================================================
    # Results
    # ============================================================
    print(f"\n  Total training time: {total_elapsed:.1f}s")

    # Save results
    save_results(results, output_dir)

    # Print comparison
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        print_results_table(valid_results)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Models trained: {len(results)}")
    print(f"  All models use enriched graphs (Morgan + MACCS + atom features)")
    print(f"  Output directory: {output_dir}")
    print(f"  Results file: {output_dir}/results.json")
    print(f"  Comparison CSV: {output_dir}/comparison.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
