"""Train VEGFR2 activity models (ML + GNN) with GPU enforcement and optional HPO."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from vegfr2.data import load_csv, preprocess, split
from vegfr2.device import get_device
from vegfr2.features import mol_to_graph, smiles_to_morgan, collate_graphs
from vegfr2.gnn_models import build_model, save_checkpoint
from vegfr2.ml_models import train_ml_model, predict_ml_model, save_ml_model
from vegfr2.metrics import classification_metrics
from vegfr2.types import GraphBatch
from typing import cast


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, smiles: list[str], labels: list[int]):
        self.graphs = [mol_to_graph(s) for s in smiles]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return self.graphs[idx], self.labels[idx]


def collate_batch(batch: list[tuple[dict, int]]) -> GraphBatch:
    graphs, labels = zip(*batch)
    return cast(GraphBatch, collate_graphs(list(graphs), list(labels)))


class _PyGDataset(torch.utils.data.Dataset):
    def __init__(self, smiles, labels):
        from torch_geometric.data import Data
        self.data_list = []
        for s, y in zip(smiles, labels):
            g = mol_to_graph(s)
            data = Data(x=g["node_feats"], edge_index=g["edge_index"], edge_attr=g["edge_feats"], y=torch.tensor([y], dtype=torch.float32))
            self.data_list.append(data)
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]


def train_gnn(
    name: str,
    train_df,
    val_df,
    test_df,
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    do_hpo: bool = False,
    use_pyg: bool = False,
) -> dict:
    seed_everything(cfg["seed"])

    if use_pyg:
        from vegfr2.gnn_pyg import train_gnn_pyg, save_checkpoint as save_checkpoint_pyg
        print(f"  Using PyTorch Geometric implementation for {name.upper()}")
        model = train_gnn_pyg(
            name=name,
            train_smiles=train_df["smiles"].tolist(),
            train_labels=train_df["active"].astype(int).tolist(),
            val_smiles=val_df["smiles"].tolist(),
            val_labels=val_df["active"].astype(int).tolist(),
            hidden=cfg["gnn"]["hidden"],
            layers=cfg["gnn"]["layers"],
            heads=cfg["gnn"]["heads"],
            lr=cfg["gnn"]["lr"],
            batch_size=cfg["gnn"]["batch"],
            epochs=cfg["gnn"]["epochs"],
            patience=cfg["gnn"]["patience"],
            seed=cfg["seed"],
            device=device,
        )
        best_path = output_dir / f"{name}/best.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint_pyg(model, best_path)

        from torch_geometric.loader import DataLoader as PyGLoader
        test_ds_pyg = _PyGDataset(test_df["smiles"].tolist(), test_df["active"].astype(int).tolist())
        test_loader = PyGLoader(test_ds_pyg, batch_size=cfg["gnn"]["batch"], shuffle=False)
        test_probs: list[float] = []
        test_true: list[int] = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                if name == "mpnn":
                    logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    logits = model(batch.x, batch.edge_index, batch.batch)
                test_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
                test_true.extend(batch.y.squeeze().cpu().numpy().astype(int))
        return classification_metrics(test_true, test_probs)

    train_ds = GraphDataset(train_df["smiles"].tolist(), train_df["active"].astype(int).tolist())
    val_ds = GraphDataset(val_df["smiles"].tolist(), val_df["active"].astype(int).tolist())
    test_ds = GraphDataset(test_df["smiles"].tolist(), test_df["active"].astype(int).tolist())

    batch_size = cfg["gnn"]["batch"]
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    if do_hpo:
        from vegfr2.hpo import optimize_gnn

        def objective(trial):
            hidden = trial.suggest_categorical("hidden", [64, 128, 256])
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            layers = trial.suggest_int("layers", 2, 5)
            heads = trial.suggest_categorical("heads", [2, 4, 8]) if name == "gat" else 1
            dropout = trial.suggest_float("dropout", 0.1, 0.5)

            model = build_model(name, in_dim=32, hidden=hidden, layers=layers, heads=heads, edge_dim=11, dropout=dropout).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            loss_fn = nn.BCEWithLogitsLoss()

            for _ in range(30):
                model.train()
                for batch in train_loader:
                    batch_tr = cast(GraphBatch, {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
                    logits = model(batch_tr, device)
                    loss = loss_fn(logits.squeeze(), batch_tr["labels"].squeeze())
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            model.eval()
            val_probs: list[float] = []
            val_true: list[int] = []
            with torch.no_grad():
                for batch in val_loader:
                    batch_v = cast(GraphBatch, {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
                    logits = model(batch_v, device)
                    val_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
                    val_true.extend(batch_v["labels"].squeeze().cpu().numpy().astype(int))
            metrics = classification_metrics(val_true, val_probs)
            return metrics.get("auc") or 0.0

        best_params, best_val = optimize_gnn(objective, n_trials=cfg["hpo"]["n_trials"])
        print(f"[HPO] {name}: best val AUC={best_val:.4f}, params={best_params}")

    hidden = cfg["gnn"]["hidden"]
    layers = cfg["gnn"]["layers"]
    heads = cfg["gnn"]["heads"]
    lr = cfg["gnn"]["lr"]
    epochs = cfg["gnn"]["epochs"]
    patience = cfg["gnn"]["patience"]
    dropout = cfg["gnn"].get("dropout", 0.2)

    model = build_model(name, in_dim=32, hidden=hidden, layers=layers, heads=heads, edge_dim=11, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    n_active = train_df["active"].sum()
    n_inactive = len(train_df) - n_active
    pos_weight = torch.tensor([n_inactive / n_active], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = -1.0
    best_path = output_dir / f"{name}/best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch_t = cast(GraphBatch, {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
            logits = model(batch_t, device)
            loss = loss_fn(logits.squeeze(), batch_t["labels"].squeeze())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch_t["num_graphs"]

        model.eval()
        val_probs: list[float] = []
        val_true: list[int] = []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_v = cast(GraphBatch, {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
                logits = model(batch_v, device)
                loss = loss_fn(logits.squeeze(), batch_v["labels"].squeeze())
                val_loss += loss.item() * batch_v["num_graphs"]
                val_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
                val_true.extend(batch_v["labels"].squeeze().cpu().numpy().astype(int))

        val_metrics = classification_metrics(val_true, val_probs)
        val_auc = val_metrics.get("auc") or 0.0

        scheduler.step()

        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, best_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch} (best AUC={best_auc:.4f})")
                break

        print(f"  Epoch {epoch:3d} | train_loss={total_loss/len(train_df):.4f} val_loss={val_loss/len(val_df):.4f} val_AUC={val_auc:.4f}")

    # Final test evaluation with best checkpoint
    best_model = type(model)(**cast(dict, model.init_kwargs)).to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    best_model.eval()
    test_probs: list[float] = []
    test_true: list[int] = []
    with torch.no_grad():
        for batch in test_loader:
            batch_te = cast(GraphBatch, {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
            logits = best_model(batch_te, device)
            test_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
            test_true.extend(batch_te["labels"].squeeze().cpu().numpy().astype(int))

    return classification_metrics(test_true, test_probs)


def train_ml(
    name: str,
    train_df,
    val_df,
    test_df,
    cfg: dict,
    output_dir: Path,
) -> dict:
    seed_everything(cfg["seed"])

    radius = cfg["fingerprint"]["radius"]
    n_bits = cfg["fingerprint"]["n_bits"]

    def fps(df):
        return np.vstack([smiles_to_morgan(s, radius=radius, n_bits=n_bits) for s in df["smiles"]])

    X_train, y_train = fps(train_df), train_df["active"].values
    X_test, y_test = fps(test_df), test_df["active"].values

    estimator = train_ml_model(name, X_train, y_train, seed=cfg["seed"])
    probs = predict_ml_model(estimator, X_test)

    model_path = output_dir / f"{name}/model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_ml_model(estimator, model_path)

    return classification_metrics(y_test.tolist(), probs.tolist())


def main() -> int:
    parser = argparse.ArgumentParser(description="Train VEGFR2 activity models")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default="all", choices=["gcn", "gat", "mpnn", "rf", "svm", "xgb", "all"])
    parser.add_argument("--hpo", action="store_true", help="Run Optuna HPO for GNNs")
    parser.add_argument("--train-csv", help="Pre-processed train CSV (skips preprocessing)")
    parser.add_argument("--val-csv", help="Pre-processed val CSV")
    parser.add_argument("--test-csv", help="Pre-processed test CSV")
    parser.add_argument("--pyg", action="store_true", help="Use PyTorch Geometric GNN implementations")
    args = parser.parse_args()

    # GPU enforcement at entrypoint (paper requirement: GPU-only training)
    device = get_device()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed_everything(cfg["seed"])

    if args.train_csv and args.val_csv and args.test_csv:
        train_df = load_csv(args.train_csv)
        val_df = load_csv(args.val_csv)
        test_df = load_csv(args.test_csv)
        print(f"Loaded pre-processed data: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    else:
        raw_csv = Path(cfg["paths"]["raw_csv"])
        if not raw_csv.exists():
            print(f"Raw data not found: {raw_csv}. Run scripts/download_data.py first.", file=sys.stderr)
            return 1

        df = load_csv(raw_csv)
        df = preprocess(df)
        train_df, val_df, test_df = split(df, seed=cfg["seed"])

        print(f"Data: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    models_to_run = [args.model] if args.model != "all" else ["rf", "svm", "xgb", "gcn", "gat", "mpnn"]
    results = {}
    output_dir = Path(cfg["paths"]["output_dir"])

    for name in models_to_run:
        print(f"\n=== Training {name.upper()} ===")
        if name in {"rf", "svm", "xgb"}:
            results[name] = train_ml(name, train_df, val_df, test_df, cfg, output_dir)
        else:
            results[name] = train_gnn(name, train_df, val_df, test_df, cfg, device, output_dir, do_hpo=args.hpo, use_pyg=args.pyg)

    # Save results summary
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))

    # Print table
    print("\n=== RESULTS ===")
    header = f"{'Model':<8} {'ACC':>6} {'SEN':>6} {'SPE':>6} {'MCC':>6} {'AUC':>6}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        auc_str = f"{m['auc']:.4f}" if m["auc"] is not None else "N/A"
        print(f"{name:<8} {m['acc']:.4f} {m['sen']:.4f} {m['spe']:.4f} {m['mcc']:.4f} {auc_str:>6}")

    return 0


if __name__ == "__main__":
    sys.exit(main())