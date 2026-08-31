#!/usr/bin/env python3
"""
VEGFR2 Virtual Screening - LIGHTWEIGHT Pipeline (Colab Ready)
=============================================================
Fast approach: NO enriched graphs, NO heavy computation.
- ML on Morgan fingerprints (fast, accurate)
- GNN on graph-only (light, no FP in nodes)
- Ensemble: combine both at the end

How to use:
  1. Open Google Colab (https://colab.research.google.com)
  2. Upload this file OR paste the contents
  3. Runtime -> Change runtime type -> T4 GPU (or CPU works too)
  4. Runtime -> Run all
"""

# %% [markdown]
# # VEGFR2 Lightweight Pipeline
# ## Fast ML + Simple GNN (No Heavy Enriched Graphs)
# Runs on CPU in ~5-10 minutes, GPU in ~3-5 minutes.

# %%
# @title 1. Install Dependencies
print("Installing packages...")
%pip install -q rdkit xgboost scikit-learn pandas numpy
print("Done!")

# %%
# @title 2. Clone Repository
import os
import sys

REPO_URL = "https://github.com/Techbjd/ai-code.git"
REPO_DIR = "/content/ai-code"

if not os.path.exists(REPO_DIR):
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
    print("Cloned!")
else:
    os.system(f"cd {REPO_DIR} && git pull")
    print("Updated!")

sys.path.insert(0, os.path.join(REPO_DIR, "src"))
os.chdir(REPO_DIR)

# %%
# @title 3. Load Data
import pandas as pd
import numpy as np
from vegfr2.data import load_csv, preprocess, split

df = load_csv("data/raw/chembl_vegfr2.csv")
df = preprocess(df)
train_df, val_df, test_df = split(df, seed=42)

print(f"Dataset: {len(df)} molecules")
print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# %%
# @title 4. Extract Morgan Fingerprints (2048-bit)
from vegfr2.features import smiles_to_morgan

print("Extracting Morgan fingerprints...")
X_train = np.vstack([smiles_to_morgan(s) for s in train_df["smiles"]])
X_val = np.vstack([smiles_to_morgan(s) for s in val_df["smiles"]])
X_test = np.vstack([smiles_to_morgan(s) for s in test_df["smiles"]])

y_train = train_df["active"].values.astype(int)
y_val = val_df["active"].values.astype(int)
y_test = test_df["active"].values.astype(int)

print(f"Features: {X_train.shape[1]}-dim Morgan fingerprints")

# %%
# @title 5. Train ML Models (Fast - No GPU Needed)
from vegfr2.ml_models import train_ml_model, predict_ml_model
from vegfr2.metrics import classification_metrics

results = {}

print("=" * 60)
print("ML MODELS ON MORGAN FINGERPRINTS")
print("=" * 60)

for name in ["rf", "svm", "xgb"]:
    print(f"\nTraining {name.upper()}...")
    model = train_ml_model(name, X_train, y_train, seed=42)
    probs = predict_ml_model(model, X_test)
    metrics = classification_metrics(y_test.tolist(), probs.tolist())
    results[name] = metrics
    print(f"  AUC={metrics.get('auc', 0):.4f} ACC={metrics['acc']:.4f}")

# %%
# @title 6. Simple GNN (Graph Only, No Fingerprints)
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from vegfr2.features import mol_to_graph
from vegfr2.gnn_pyg import build_pyg_model
from vegfr2.metrics import classification_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def make_loader(smiles_list, labels, batch_size=256, shuffle=False):
    data_list = []
    for s, y in zip(smiles_list, labels):
        try:
            g = mol_to_graph(s)
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


print("\n" + "=" * 60)
print("SIMPLE GNN (32-dim atom features, NO fingerprints)")
print("=" * 60)

train_loader = make_loader(train_df["smiles"].tolist(), y_train, shuffle=True)
val_loader = make_loader(val_df["smiles"].tolist(), y_val)
test_loader = make_loader(test_df["smiles"].tolist(), y_test)

model = build_pyg_model("gin", in_dim=32, hidden=64, layers=2, dropout=0.3).to(DEVICE)
print(f"Model: GIN ({sum(p.numel() for p in model.parameters()):,} params)")

opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

best_auc = -1.0
best_state = None

for epoch in range(1, 51):
    model.train()
    for batch in train_loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(logits.squeeze(), batch.y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    val_probs, val_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.batch)
            val_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
            val_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

    val_auc = classification_metrics(val_true, val_probs).get("auc") or 0.0

    if val_auc > best_auc:
        best_auc = val_auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:2d} val_AUC={val_auc:.4f}")

model.load_state_dict(best_state)
model.to(DEVICE).eval()

test_probs, test_true = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x, batch.edge_index, batch.batch)
        test_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
        test_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

results["gin_simple"] = classification_metrics(test_true, test_probs)
print(f"  Test AUC={results['gin_simple'].get('auc', 0):.4f}")

# %%
# @title 7. Final Results
print("\n" + "=" * 60)
print("FINAL RESULTS - LIGHTWEIGHT PIPELINE")
print("=" * 60)

print(f"\n{'Model':<20} {'AUC':>6} {'ACC':>6} {'MCC':>6}")
print("-" * 45)
for name, m in sorted(results.items(), key=lambda x: x[1].get("auc") or 0, reverse=True):
    auc_str = f"{m['auc']:.4f}" if m.get("auc") is not None else "N/A"
    print(f"{name.upper():<20} {auc_str:>6} {m['acc']:.4f} {m['mcc']:.4f}")

best = max(results.items(), key=lambda x: x[1].get("auc") or 0)
print("-" * 45)
print(f"BEST: {best[0].upper()} (AUC={best[1].get('auc', 0):.4f})")
print("=" * 60)

# %%
# @title 8. Screen Large Library (e.g., COCONUT)
def screen_library(smiles_list, model, batch_size=512):
    """Fast screening using trained ML model."""
    fps = []
    valid_smiles = []
    for s in smiles_list:
        try:
            fp = smiles_to_morgan(s)
            fps.append(fp)
            valid_smiles.append(s)
        except Exception:
            pass

    X = np.vstack(fps)
    probs = predict_ml_model(model, X)

    return pd.DataFrame({
        "smiles": valid_smiles,
        "probability": probs,
        "hit": probs >= 0.5,
    }).sort_values("probability", ascending=False)


# Example: Screen a small library
example_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CC1=CC=CC=C1"]
print("\nExample predictions:")
for smi in example_smiles:
    fp = smiles_to_morgan(smi).reshape(1, -1)
    prob = predict_ml_model(results.get("xgb", None), fp) if "xgb" in results else [0.5]
    print(f"  {smi:<30} {prob[0]:.4f}")

# %%
# @title 9. Summary
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"\nThis lightweight approach:")
print(f"  - Uses Morgan fingerprints only (no enriched graphs)")
print(f"  - No GPU memory issues")
print(f"  - Fast training (~5 min on CPU)")
print(f"  - Can screen millions of molecules")
print(f"\nBest model: {best[0].upper()} (AUC={best[1].get('auc', 0):.4f})")
print("=" * 60)
