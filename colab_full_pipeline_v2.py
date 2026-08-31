#!/usr/bin/env python3
"""
VEGFR2 Virtual Screening - Full Pipeline V2 (Colab Ready)
==========================================================
EVERY model trained with EVERY feature combination.
- ML: MACCS only, Morgan only, Morgan+MACCS
- GNN: Graph only (32-dim), Enriched (2246-dim)
- Ensemble: GNN + XGBoost/RF

How to use:
  1. Open Google Colab (https://colab.research.google.com)
  2. Upload this file OR paste the contents
  3. Runtime -> Change runtime type -> T4 GPU
  4. Runtime -> Run all
"""

# %% [markdown]
# # VEGFR2 Virtual Screening Pipeline V2
# ## Complete ML + GNN Model Comparison (All Feature Combinations)
# Runs on GPU (T4) in ~30-40 minutes.

# %%
# @title 1. Install Dependencies (run first)

print("Installing packages...")

%pip install -q rdkit torch_geometric xgboost optuna scikit-learn pandas numpy pyyaml

print("All packages ready!")

# %%
# @title 2. Clone Repository
import os
import sys

REPO_URL = "https://github.com/Techbjd/ai-code.git"
REPO_DIR = "/content/ai-code"

if not os.path.exists(REPO_DIR):
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
    print("Repository cloned!")
else:
    os.system(f"cd {REPO_DIR} && git pull")
    print("Repository updated!")

sys.path.insert(0, os.path.join(REPO_DIR, "src"))
os.chdir(REPO_DIR)
print(f"Working directory: {os.getcwd()}")

# %%
# @title 3. Check GPU
import torch
import warnings
warnings.filterwarnings("ignore", message=".*scatter.*")
warnings.filterwarnings("ignore", message=".*index_reduce.*")

print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    DEVICE = torch.device("cuda")
else:
    print("No GPU found - using CPU (will be slower)")
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# %%
# @title 4. Load and Preprocess Data
import pandas as pd
import numpy as np
from vegfr2.data import load_csv, preprocess, split

RAW_CSV = "data/raw/chembl_vegfr2.csv"
print(f"Loading data from {RAW_CSV}...")

df = load_csv(RAW_CSV)
df = preprocess(df)
train_df, val_df, test_df = split(df, seed=42)

print(f"\nDataset Statistics:")
print(f"  Total molecules: {len(df)}")
print(f"  Train: {len(train_df)} ({train_df['active'].mean():.1%} active)")
print(f"  Val:   {len(val_df)} ({val_df['active'].mean():.1%} active)")
print(f"  Test:  {len(test_df)} ({test_df['active'].mean():.1%} active)")

# %%
# @title 5. Extract ALL Feature Types
from vegfr2.features import smiles_to_morgan, smiles_to_maccs

print("Extracting Morgan fingerprints (2048-bit)...")
X_train_morgan = np.vstack([smiles_to_morgan(s) for s in train_df["smiles"]])
X_val_morgan = np.vstack([smiles_to_morgan(s) for s in val_df["smiles"]])
X_test_morgan = np.vstack([smiles_to_morgan(s) for s in test_df["smiles"]])

print("Extracting MACCS keys (166-bit)...")
X_train_maccs = np.vstack([smiles_to_maccs(s) for s in train_df["smiles"]])
X_val_maccs = np.vstack([smiles_to_maccs(s) for s in val_df["smiles"]])
X_test_maccs = np.vstack([smiles_to_maccs(s) for s in test_df["smiles"]])

X_train_both = np.hstack([X_train_morgan, X_train_maccs])
X_val_both = np.hstack([X_val_morgan, X_val_maccs])
X_test_both = np.hstack([X_test_morgan, X_test_maccs])

y_train = train_df["active"].values.astype(int)
y_val = val_df["active"].values.astype(int)
y_test = test_df["active"].values.astype(int)

print(f"\nFeature Dimensions:")
print(f"  Morgan only:     {X_train_morgan.shape[1]}-dim")
print(f"  MACCS only:      {X_train_maccs.shape[1]}-dim")
print(f"  Morgan + MACCS:  {X_train_both.shape[1]}-dim")

# %%
# @title 6. Train ML Models - MACCS Only (166-dim)
from vegfr2.ml_models import train_ml_model, predict_ml_model
from vegfr2.metrics import classification_metrics

results = {}

print("=" * 70)
print("PART A: ML MODELS - MACCS ONLY (166-dim)")
print("=" * 70)

for name in ["rf", "svm", "xgb"]:
    print(f"\n--- Training {name.upper()} on MACCS ---")
    model = train_ml_model(name, X_train_maccs, y_train, seed=42)
    probs = predict_ml_model(model, X_test_maccs)
    metrics = classification_metrics(y_test.tolist(), probs.tolist())
    results[f"ml_{name}_maccs"] = metrics
    print(f"  AUC={metrics.get('auc', 0):.4f} ACC={metrics['acc']:.4f} MCC={metrics['mcc']:.4f}")

# %%
# @title 7. Train ML Models - Morgan Only (2048-dim)
print("\n" + "=" * 70)
print("PART B: ML MODELS - MORGAN ONLY (2048-dim)")
print("=" * 70)

for name in ["rf", "svm", "xgb"]:
    print(f"\n--- Training {name.upper()} on Morgan ---")
    model = train_ml_model(name, X_train_morgan, y_train, seed=42)
    probs = predict_ml_model(model, X_test_morgan)
    metrics = classification_metrics(y_test.tolist(), probs.tolist())
    results[f"ml_{name}_morgan"] = metrics
    print(f"  AUC={metrics.get('auc', 0):.4f} ACC={metrics['acc']:.4f} MCC={metrics['mcc']:.4f}")

# %%
# @title 8. Train ML Models - Morgan + MACCS (2214-dim)
print("\n" + "=" * 70)
print("PART C: ML MODELS - MORGAN + MACCS (2214-dim)")
print("=" * 70)

for name in ["rf", "svm", "xgb"]:
    print(f"\n--- Training {name.upper()} on Morgan+MACCS ---")
    model = train_ml_model(name, X_train_both, y_train, seed=42)
    probs = predict_ml_model(model, X_test_both)
    metrics = classification_metrics(y_test.tolist(), probs.tolist())
    results[f"ml_{name}_both"] = metrics
    print(f"  AUC={metrics.get('auc', 0):.4f} ACC={metrics['acc']:.4f} MCC={metrics['mcc']:.4f}")

# %%
# @title 9. Train GNN Models - Graph Only (32-dim, NO fingerprints)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from vegfr2.features import mol_to_graph, mol_to_graph_with_fps
from vegfr2.gnn_pyg import build_pyg_model
from vegfr2.metrics import classification_metrics


def make_graph_only_loader(smiles_list, labels, batch_size=128, shuffle=False):
    """Create DataLoader with ONLY molecular graph features (32-dim)."""
    data_list = []
    for s, y in zip(smiles_list, labels):
        try:
            g = mol_to_graph(s)  # 32-dim atom features only
            data = Data(
                x=g["node_feats"],
                edge_index=g["edge_index"],
                edge_attr=g["edge_feats"],
                y=torch.tensor([y], dtype=torch.float32),
            )
            data_list.append(data)
        except Exception:
            pass
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)


def make_enriched_loader(smiles_list, labels, batch_size=128, shuffle=False):
    """Create DataLoader with enriched graphs (32 + 2048 + 166 = 2246-dim)."""
    data_list = []
    for s, y in zip(smiles_list, labels):
        g = mol_to_graph_with_fps(s, use_morgan=True, use_maccs=True)
        data = Data(
            x=g["node_feats"],
            edge_index=g["edge_index"],
            edge_attr=g["edge_feats"],
            y=torch.tensor([y], dtype=torch.float32),
        )
        data_list.append(data)
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)


def forward_model(model, model_name, batch):
    if model_name == "mpnn":
        return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    elif model_name == "graph_transformer":
        return model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
    else:
        return model(batch.x, batch.edge_index, batch.batch)


def train_gnn(model_name, train_df, val_df, test_df, device, in_dim, feature_type, epochs=100, patience=15):
    import torch.nn as nn
    torch.manual_seed(42)

    if feature_type == "graph_only":
        train_loader = make_graph_only_loader(
            train_df["smiles"].tolist(),
            train_df["active"].astype(int).tolist(),
            shuffle=True,
        )
        val_loader = make_graph_only_loader(
            val_df["smiles"].tolist(), val_df["active"].astype(int).tolist()
        )
        test_loader = make_graph_only_loader(
            test_df["smiles"].tolist(), test_df["active"].astype(int).tolist()
        )
    else:  # enriched
        train_loader = make_enriched_loader(
            train_df["smiles"].tolist(),
            train_df["active"].astype(int).tolist(),
            shuffle=True,
        )
        val_loader = make_enriched_loader(
            val_df["smiles"].tolist(), val_df["active"].astype(int).tolist()
        )
        test_loader = make_enriched_loader(
            test_df["smiles"].tolist(), test_df["active"].astype(int).tolist()
        )

    hidden = 64 if model_name == "mpnn" else 128
    model = build_pyg_model(
        model_name, in_dim=in_dim, hidden=hidden, layers=3, heads=8, dropout=0.3
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    n_active = train_df["active"].sum()
    n_inactive = len(train_df) - n_active
    pos_weight = torch.tensor([n_inactive / n_active], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            logits = forward_model(model, model_name, batch)
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
                logits = forward_model(model, model_name, batch)
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
            logits = forward_model(model, model_name, batch)
            test_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
            test_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

    return classification_metrics(test_true, test_probs)


gnn_names = ["gcn", "gat", "gatv2", "mpnn", "gin", "pna", "graph_transformer"]

print("\n" + "=" * 70)
print("PART D: GNN MODELS - GRAPH ONLY (32-dim, NO fingerprints)")
print("=" * 70)

for name in gnn_names:
    print(f"\n--- Training {name.upper()} (graph only) ---")
    try:
        metrics = train_gnn(name, train_df, val_df, test_df, DEVICE, in_dim=32, feature_type="graph_only", epochs=100, patience=15)
        results[f"gnn_{name}_graph_only"] = metrics
        print(f"  AUC={metrics.get('auc', 0):.4f} ACC={metrics['acc']:.4f} MCC={metrics['mcc']:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")

# %%
# @title 10. Train GNN Models - Enriched Graphs (2246-dim)
print("\n" + "=" * 70)
print("PART E: GNN MODELS - ENRICHED GRAPHS (2246-dim = atom + Morgan + MACCS)")
print("=" * 70)

for name in gnn_names:
    print(f"\n--- Training {name.upper()} (enriched) ---")
    try:
        metrics = train_gnn(name, train_df, val_df, test_df, DEVICE, in_dim=2246, feature_type="enriched", epochs=100, patience=15)
        results[f"gnn_{name}_enriched"] = metrics
        print(f"  AUC={metrics.get('auc', 0):.4f} ACC={metrics['acc']:.4f} MCC={metrics['mcc']:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")

# %%
# @title 11. Train Ensemble Models (GNN + ML)
from vegfr2.models.ensemble import GNNEnsembleClassifier

ensemble_configs = [
    ("gin", "xgb"),
    ("pna", "xgb"),
    ("gin", "rf"),
]

print("\n" + "=" * 70)
print("PART F: ENSEMBLE MODELS (GNN embeddings + Morgan + MACCS + XGBoost/RF)")
print("=" * 70)

for gnn_name, ml_name in ensemble_configs:
    name = f"ensemble_{gnn_name}_{ml_name}"
    print(f"\n--- Training {name} ---")
    try:
        ensemble = GNNEnsembleClassifier(
            gnn_name=gnn_name,
            ml_name=ml_name,
            hidden=128,
            layers=3,
            heads=8,
            dropout=0.3,
            seed=42,
        )
        ensemble.fit(
            train_smiles=train_df["smiles"].tolist(),
            train_labels=train_df["active"].astype(int).tolist(),
            val_smiles=val_df["smiles"].tolist(),
            val_labels=val_df["active"].astype(int).tolist(),
            device=DEVICE,
            gnn_epochs=50,
        )
        probs = ensemble.predict_proba(test_df["smiles"].tolist(), device=DEVICE)
        y_test_ens = test_df["active"].values.tolist()
        metrics = classification_metrics(y_test_ens, probs.tolist())
        results[name] = metrics
        print(f"  AUC={metrics.get('auc', 0):.4f} ACC={metrics['acc']:.4f} MCC={metrics['mcc']:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")

# %%
# @title 12. FINAL COMPARISON TABLE
print("\n" + "=" * 100)
print("VEGFR2 V2 - COMPLETE MODEL COMPARISON (ALL FEATURES)")
print("=" * 100)

# Group results by category
ml_maccs = {k: v for k, v in results.items() if k.startswith("ml_") and "maccs" in k}
ml_morgan = {k: v for k, v in results.items() if k.startswith("ml_") and "morgan" in k and "both" not in k}
ml_both = {k: v for k, v in results.items() if k.startswith("ml_") and "both" in k}
gnn_graph = {k: v for k, v in results.items() if k.startswith("gnn_") and "graph_only" in k}
gnn_enriched = {k: v for k, v in results.items() if k.startswith("gnn_") and "enriched" in k}
ensemble = {k: v for k, v in results.items() if k.startswith("ensemble_")}

# Print ML MACCS
print("\n--- ML MODELS - MACCS ONLY (166-dim) ---")
print(f"{'Model':<15} {'AUC':>6} {'ACC':>6} {'SEN':>6} {'SPE':>6} {'MCC':>6}")
print("-" * 60)
for name, m in sorted(ml_maccs.items(), key=lambda x: x[1].get("auc") or 0, reverse=True):
    auc_str = f"{m['auc']:.4f}" if m.get("auc") is not None else "N/A"
    print(f"{name.upper():<15} {auc_str:>6} {m['acc']:.4f} {m['sen']:.4f} {m['spe']:.4f} {m['mcc']:.4f}")

# Print ML Morgan
print("\n--- ML MODELS - MORGAN ONLY (2048-dim) ---")
print(f"{'Model':<15} {'AUC':>6} {'ACC':>6} {'SEN':>6} {'SPE':>6} {'MCC':>6}")
print("-" * 60)
for name, m in sorted(ml_morgan.items(), key=lambda x: x[1].get("auc") or 0, reverse=True):
    auc_str = f"{m['auc']:.4f}" if m.get("auc") is not None else "N/A"
    print(f"{name.upper():<15} {auc_str:>6} {m['acc']:.4f} {m['sen']:.4f} {m['spe']:.4f} {m['mcc']:.4f}")

# Print ML Both
print("\n--- ML MODELS - MORGAN + MACCS (2214-dim) ---")
print(f"{'Model':<15} {'AUC':>6} {'ACC':>6} {'SEN':>6} {'SPE':>6} {'MCC':>6}")
print("-" * 60)
for name, m in sorted(ml_both.items(), key=lambda x: x[1].get("auc") or 0, reverse=True):
    auc_str = f"{m['auc']:.4f}" if m.get("auc") is not None else "N/A"
    print(f"{name.upper():<15} {auc_str:>6} {m['acc']:.4f} {m['sen']:.4f} {m['spe']:.4f} {m['mcc']:.4f}")

# Print GNN Graph Only
print("\n--- GNN MODELS - GRAPH ONLY (32-dim, NO fingerprints) ---")
print(f"{'Model':<18} {'AUC':>6} {'ACC':>6} {'SEN':>6} {'SPE':>6} {'MCC':>6}")
print("-" * 60)
for name, m in sorted(gnn_graph.items(), key=lambda x: x[1].get("auc") or 0, reverse=True):
    parts = name.split("_")
    model_type = parts[1].upper()
    auc_str = f"{m['auc']:.4f}" if m.get("auc") is not None else "N/A"
    print(f"{model_type:<18} {auc_str:>6} {m['acc']:.4f} {m['sen']:.4f} {m['spe']:.4f} {m['mcc']:.4f}")

# Print GNN Enriched
print("\n--- GNN MODELS - ENRICHED GRAPHS (2246-dim = atom + Morgan + MACCS) ---")
print(f"{'Model':<18} {'AUC':>6} {'ACC':>6} {'SEN':>6} {'SPE':>6} {'MCC':>6}")
print("-" * 60)
for name, m in sorted(gnn_enriched.items(), key=lambda x: x[1].get("auc") or 0, reverse=True):
    parts = name.split("_")
    model_type = parts[1].upper()
    auc_str = f"{m['auc']:.4f}" if m.get("auc") is not None else "N/A"
    print(f"{model_type:<18} {auc_str:>6} {m['acc']:.4f} {m['sen']:.4f} {m['spe']:.4f} {m['mcc']:.4f}")

# Print Ensemble
print("\n--- ENSEMBLE MODELS (GNN + ML) ---")
print(f"{'Model':<25} {'AUC':>6} {'ACC':>6} {'SEN':>6} {'SPE':>6} {'MCC':>6}")
print("-" * 60)
for name, m in sorted(ensemble.items(), key=lambda x: x[1].get("auc") or 0, reverse=True):
    parts = name.split("_")
    method = f"{parts[1]}+{parts[2].upper()}"
    auc_str = f"{m['auc']:.4f}" if m.get("auc") is not None else "N/A"
    print(f"{method:<25} {auc_str:>6} {m['acc']:.4f} {m['sen']:.4f} {m['spe']:.4f} {m['mcc']:.4f}")

# Top 5 Overall
all_sorted = sorted(results.items(), key=lambda x: x[1].get("auc") or 0, reverse=True)
print("\n" + "=" * 100)
print("TOP 5 MODELS OVERALL:")
print("-" * 100)
for i, (name, m) in enumerate(all_sorted[:5], 1):
    auc_str = f"{m['auc']:.4f}" if m.get("auc") is not None else "N/A"
    print(f"  {i}. {name:<40} AUC={auc_str} ACC={m['acc']:.4f} MCC={m['mcc']:.4f}")

print("\n" + "=" * 100)
print("KEY FINDINGS:")
print("-" * 100)

# Calculate averages
maccs_aucs = [v.get("auc", 0) for v in ml_maccs.values() if v.get("auc")]
morgan_aucs = [v.get("auc", 0) for v in ml_morgan.values() if v.get("auc")]
both_aucs = [v.get("auc", 0) for v in ml_both.values() if v.get("auc")]
graph_aucs = [v.get("auc", 0) for v in gnn_graph.values() if v.get("auc")]
enriched_aucs = [v.get("auc", 0) for v in gnn_enriched.values() if v.get("auc")]

if maccs_aucs:
    print(f"  ML (MACCS only) avg AUC:      {np.mean(maccs_aucs):.4f}")
if morgan_aucs:
    print(f"  ML (Morgan only) avg AUC:     {np.mean(morgan_aucs):.4f}")
if both_aucs:
    print(f"  ML (Morgan+MACCS) avg AUC:    {np.mean(both_aucs):.4f}")
if graph_aucs:
    print(f"  GNN (graph only) avg AUC:     {np.mean(graph_aucs):.4f}")
if enriched_aucs:
    print(f"  GNN (enriched) avg AUC:       {np.mean(enriched_aucs):.4f}")

if graph_aucs and enriched_aucs:
    print(f"\n  WHY enriched works: +{np.mean(enriched_aucs) - np.mean(graph_aucs):.4f} AUC")
    print(f"  Fingerprints inject 20 years of cheminformatics knowledge into GNN")

print("=" * 100)

# %%
# @title 13. VISUALIZATION
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: ML Feature Comparison
    ax1 = axes[0]
    ml_models = ["RF", "SVM", "XGBoost"]
    x = np.arange(len(ml_models))
    width = 0.25

    maccs_vals = [ml_maccs.get(f"ml_{m.lower()}_maccs", {}).get("auc", 0) for m in ml_models]
    morgan_vals = [ml_morgan.get(f"ml_{m.lower()}_morgan", {}).get("auc", 0) for m in ml_models]
    both_vals = [ml_both.get(f"ml_{m.lower()}_both", {}).get("auc", 0) for m in ml_models]

    bars1 = ax1.bar(x - width, maccs_vals, width, label="MACCS (166)", color="#2196F3")
    bars2 = ax1.bar(x, morgan_vals, width, label="Morgan (2048)", color="#4CAF50")
    bars3 = ax1.bar(x + width, both_vals, width, label="Morgan+MACCS (2214)", color="#FF9800")

    ax1.set_xlabel("ML Model")
    ax1.set_ylabel("AUC")
    ax1.set_title("ML Models: Feature Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ml_models)
    ax1.legend()
    ax1.set_ylim(0.5, 1.0)

    # Plot 2: GNN Feature Comparison
    ax2 = axes[1]
    gnn_models = ["GCN", "GAT", "GATv2", "MPNN", "GIN", "PNA", "Transformer"]
    x2 = np.arange(len(gnn_models))
    width2 = 0.35

    graph_vals = [gnn_graph.get(f"gnn_{m.lower()}_graph_only", {}).get("auc", 0) for m in gnn_models]
    enriched_vals = [gnn_enriched.get(f"gnn_{m.lower()}_enriched", {}).get("auc", 0) for m in gnn_models]

    bars4 = ax2.bar(x2 - width2/2, graph_vals, width2, label="Graph only (32)", color="#F44336")
    bars5 = ax2.bar(x2 + width2/2, enriched_vals, width2, label="Enriched (2246)", color="#9C27B0")

    ax2.set_xlabel("GNN Model")
    ax2.set_ylabel("AUC")
    ax2.set_title("GNN Models: Enriched vs Graph-only")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(gnn_models)
    ax2.legend()
    ax2.set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig("feature_comparison_v2.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved to feature_comparison_v2.png")

except ImportError:
    print("matplotlib not available - skipping visualization")

# %%
# @title 14. Unit Tests
import pytest

print("\nRunning test suite...")
exit_code = pytest.main(["tests/", "-v", "--tb=short", "-q"])
print(f"\nTest suite: {'ALL PASSED' if exit_code == 0 else 'SOME FAILED'}")

# %%
# @title 15. Save Results to JSON
import json

output = {
    "device": str(DEVICE),
    "dataset": {
        "total": len(df),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    },
    "results": {},
    "summary": {
        "ml_maccs_avg_auc": np.mean(maccs_aucs) if maccs_aucs else None,
        "ml_morgan_avg_auc": np.mean(morgan_aucs) if morgan_aucs else None,
        "ml_both_avg_auc": np.mean(both_aucs) if both_aucs else None,
        "gnn_graph_avg_auc": np.mean(graph_aucs) if graph_aucs else None,
        "gnn_enriched_avg_auc": np.mean(enriched_aucs) if enriched_aucs else None,
    },
}

for name, m in results.items():
    output["results"][name] = {
        k: float(v) if isinstance(v, (np.floating, float)) else v
        for k, v in m.items()
        if k != "confusion_matrix"
    }

with open("colab_results_v2.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to colab_results_v2.json")
print("Done! All models trained and evaluated.")
