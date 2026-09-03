#!/usr/bin/env python3
"""
VEGFR2 - Model Variants Comparison (Colab Ready)
=================================================
Train and compare all model variants:
- Graph-Only (32-dim atom features, no fingerprints)
- Graph + Morgan (32-dim + 2048-dim Morgan FP branch)
- Graph + MACCS (32-dim + 166-dim MACCS FP branch)
- Graph + Both (32-dim + 2214-dim combined FP branch)

Fingerprints are computed ONCE and cached, not recalculated per atom.

How to use:
  1. Open Google Colab (https://colab.research.google.com)
  2. Upload this file OR paste the contents
  3. Runtime -> Change runtime type -> T4 GPU (or CPU works too)
  4. Runtime -> Run all
"""

# %% [markdown]
# # VEGFR2 Model Variants Comparison
# ## Graph-Only vs Graph+Morgan vs Graph+MACCS vs Graph+Both
# Fingerprints computed once and cached. Clean ablation study.

# %%
# @title 1. Install Dependencies
print("Installing packages...")
%pip install -q rdkit torch_geometric xgboost scikit-learn pandas numpy
print("Done!")

# %%
# @title 2. Clone Repository
import os
import sys

REPO_URL = "https://github.com/Techbjd/ai-code.git"
REPO_DIR = "/content/ai-code"

if not os.path.exists(REPO_URL):
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
    print("Cloned!")
else:
    os.system(f"cd {REPO_URL} && git pull")

sys.path.insert(0, REPO_URL)
os.chdir(REPO_URL)
print(f"Working directory: {os.getcwd()}")

# %%
# @title 3. Import Libraries
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from vegfr2.data import load_csv, preprocess, split
from vegfr2.data_pipeline import VEGFR2Pipeline
from vegfr2.datasets import (
    GraphOnlyDataset, MorganFPDataset, MACCSFPDataset, BothFPDataset,
    get_dataset_class,
)
from vegfr2.gnn_pyg import build_pyg_model, train_fused_variant, predict_fused_variant
from vegfr2.metrics import classification_metrics
from vegfr2.features import smiles_to_morgan, smiles_to_maccs, clear_fp_cache

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# %%
# @title 4. Load and Preprocess Data
raw_csv = "data/raw/chembl_vegfr2.csv"
df = load_csv(raw_csv)
df = preprocess(df)
train_df, val_df, test_df = split(df, seed=42)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# %%
# @title 5. Run Plain Data Pipeline (Caches Fingerprints)
pipeline = VEGFR2Pipeline()
pipeline.run_plain(raw_csv, "data/processed_variants")
print("Plain data pipeline complete!")
print("Files saved:")
import os
for f in sorted(os.listdir("data/processed_variants/train")):
    size = os.path.getsize(f"data/processed_variants/train/{f}") / 1024
    print(f"  {f}: {size:.1f} KB")

# %%
# @title 6. Load Datasets (from Cached Data)
graph_only_train = GraphOnlyDataset("data/processed_variants/train")
graph_only_val = GraphOnlyDataset("data/processed_variants/test")
graph_only_test = GraphOnlyDataset("data/processed_variants/test")

morgan_train = MorganFPDataset("data/processed_variants/train")
morgan_val = MorganFPDataset("data/processed_variants/val")
morgan_test = MorganFPDataset("data/processed_variants/test")

maccs_train = MACCSFPDataset("data/processed_variants/train")
maccs_val = MACCSFPDataset("data/processed_variants/val")
maccs_test = MACCSFPDataset("data/processed_variants/test")

both_train = BothFPDataset("data/processed_variants/train")
both_val = BothFPDataset("data/processed_variants/val")
both_test = BothFPDataset("data/processed_variants/test")

print(f"Graph-Only: {len(graph_only_train)} train samples")
print(f"Morgan:     {len(morgan_train)} train samples")
print(f"MACCS:      {len(maccs_train)} train samples")
print(f"Both:       {len(both_train)} train samples")

# %%
# @title 7. Train Graph-Only GIN
print("=" * 60)
print("Training: gin_graph_only")
print("=" * 60)

BATCH_SIZE = 128
EPOCHS = 300
PATIENCE = 25
HIDDEN = 64
LAYERS = 3

train_loader = DataLoader(graph_only_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(graph_only_val, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(graph_only_test, batch_size=BATCH_SIZE, shuffle=False)

gin_graph_only = build_pyg_model("gin_graph_only", in_dim=32, hidden=HIDDEN, layers=LAYERS)
print(f"Parameters: {sum(p.numel() for p in gin_graph_only.parameters()):,}")

t0 = time.time()
gin_graph_only = train_fused_variant(
    "gin_graph_only", train_loader, val_loader,
    hidden=HIDDEN, layers=LAYERS, epochs=EPOCHS, patience=PATIENCE, device=device,
)
t_graph_only = time.time() - t0

test_probs = predict_fused_variant(gin_graph_only, test_loader, device=device)
test_true = [int(d.y.item()) for d in graph_only_test]
test_metrics = classification_metrics(test_true, test_probs.tolist())
print(f"Test AUC: {test_metrics['auc']:.4f} | Time: {t_graph_only:.1f}s")

# %%
# @title 8. Train GIN + Morgan
print("=" * 60)
print("Training: gin_morgan")
print("=" * 60)

train_loader = DataLoader(morgan_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(morgan_val, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(morgan_test, batch_size=BATCH_SIZE, shuffle=False)

gin_morgan = build_pyg_model("gin_morgan", in_dim=32, hidden=HIDDEN, layers=LAYERS)
print(f"Parameters: {sum(p.numel() for p in gin_morgan.parameters()):,}")

t0 = time.time()
gin_morgan = train_fused_variant(
    "gin_morgan", train_loader, val_loader,
    hidden=HIDDEN, layers=LAYERS, epochs=EPOCHS, patience=PATIENCE, device=device,
)
t_morgan = time.time() - t0

test_probs = predict_fused_variant(gin_morgan, test_loader, device=device)
test_true = [int(d.y.item()) for d in morgan_test]
morgan_metrics = classification_metrics(test_true, test_probs.tolist())
print(f"Test AUC: {morgan_metrics['auc']:.4f} | Time: {t_morgan:.1f}s")

# %%
# @title 9. Train GIN + MACCS
print("=" * 60)
print("Training: gin_maccs")
print("=" * 60)

train_loader = DataLoader(maccs_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(maccs_val, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(maccs_test, batch_size=BATCH_SIZE, shuffle=False)

gin_maccs = build_pyg_model("gin_maccs", in_dim=32, hidden=HIDDEN, layers=LAYERS)
print(f"Parameters: {sum(p.numel() for p in gin_maccs.parameters()):,}")

t0 = time.time()
gin_maccs = train_fused_variant(
    "gin_maccs", train_loader, val_loader,
    hidden=HIDDEN, layers=LAYERS, epochs=EPOCHS, patience=PATIENCE, device=device,
)
t_maccs = time.time() - t0

test_probs = predict_fused_variant(gin_maccs, test_loader, device=device)
test_true = [int(d.y.item()) for d in maccs_test]
maccs_metrics = classification_metrics(test_true, test_probs.tolist())
print(f"Test AUC: {maccs_metrics['auc']:.4f} | Time: {t_maccs:.1f}s")

# %%
# @title 10. Train GIN + Both (Morgan + MACCS)
print("=" * 60)
print("Training: gin_both")
print("=" * 60)

train_loader = DataLoader(both_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(both_val, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(both_test, batch_size=BATCH_SIZE, shuffle=False)

gin_both = build_pyg_model("gin_both", in_dim=32, hidden=HIDDEN, layers=LAYERS)
print(f"Parameters: {sum(p.numel() for p in gin_both.parameters()):,}")

t0 = time.time()
gin_both = train_fused_variant(
    "gin_both", train_loader, val_loader,
    hidden=HIDDEN, layers=LAYERS, epochs=EPOCHS, patience=PATIENCE, device=device,
)
t_both = time.time() - t0

test_probs = predict_fused_variant(gin_both, test_loader, device=device)
test_true = [int(d.y.item()) for d in both_test]
both_metrics = classification_metrics(test_true, test_probs.tolist())
print(f"Test AUC: {both_metrics['auc']:.4f} | Time: {t_both:.1f}s")

# %%
# @title 11. Results Comparison
results = {
    "gin_graph_only": {"auc": test_metrics["auc"], "acc": test_metrics.get("accuracy", 0), "time": t_graph_only},
    "gin_morgan":     {"auc": morgan_metrics["auc"], "acc": morgan_metrics.get("accuracy", 0), "time": t_morgan},
    "gin_maccs":      {"auc": maccs_metrics["auc"], "acc": maccs_metrics.get("accuracy", 0), "time": t_maccs},
    "gin_both":       {"auc": both_metrics["auc"], "acc": both_metrics.get("accuracy", 0), "time": t_both},
}

print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)
print(f"{'Model':<20} {'Test AUC':<12} {'Test Acc':<12} {'Time':<10}")
print("-" * 54)
for name, r in results.items():
    print(f"{name:<20} {r['auc']:<12.4f} {r['acc']:<12.4f} {r['time']:<10.1f}s")

# Find best
best_name = max(results, key=lambda k: results[k]["auc"])
print(f"\nBest: {best_name} (AUC = {results[best_name]['auc']:.4f})")

# %%
# @title 12. Visualization
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# AUC comparison
names = list(results.keys())
aucs = [results[n]["auc"] for n in names]
colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

axes[0].barh(names, aucs, color=colors)
axes[0].set_xlabel("Test AUC")
axes[0].set_title("Model Variants - AUC Comparison")
axes[0].set_xlim(0.5, 1.0)
for i, v in enumerate(aucs):
    axes[0].text(v + 0.005, i, f"{v:.4f}", va="center")

# Time comparison
times = [results[n]["time"] for n in names]
axes[1].barh(names, times, color=colors)
axes[1].set_xlabel("Training Time (s)")
axes[1].set_title("Model Variants - Training Time")

plt.tight_layout()
plt.savefig("variant_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: variant_comparison.png")

# %%
# @title 13. Save Results
with open("variant_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved: variant_results.json")

# Save model checkpoints
for name, model in [("gin_graph_only", gin_graph_only), ("gin_morgan", gin_morgan),
                     ("gin_maccs", gin_maccs), ("gin_both", gin_both)]:
    torch.save(model.state_dict(), f"{name}.pt")
    print(f"Saved: {name}.pt")

clear_fp_cache()
print("\nDone! All variants trained and compared.")
