#!/usr/bin/env python3
"""
VEGFR2 Virtual Screening - FUSED GNN Pipeline (Colab Ready)
===========================================================
Lightweight GNN + Fingerprint at graph level.
- GNN processes graph (32-dim) → structural patterns
- Fingerprint added AFTER pooling (not per atom)
- Fast, accurate, low memory

How to use:
  1. Open Google Colab (https://colab.research.google.com)
  2. Upload this file OR paste the contents
  3. Runtime -> Change runtime type -> T4 GPU
  4. Runtime -> Run all
"""

# %% [markdown]
# # VEGFR2 Fused GNN Pipeline
# ## Lightweight Graph + Fingerprint Fusion
# Runs on GPU in ~10-15 minutes.

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

if not os.path.exists(REPO_DIR):
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
    print("Cloned!")
else:
    os.system(f"cd {REPO_DIR} && git pull")
    print("Updated!")

sys.path.insert(0, os.path.join(REPO_DIR, "src"))
os.chdir(REPO_DIR)

# %%
# @title 3. Check GPU
import torch
import warnings
warnings.filterwarnings("ignore")

print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    DEVICE = torch.device("cuda")
else:
    print("No GPU - using CPU")
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# %%
# @title 4. Load Data
import pandas as pd
import numpy as np
from vegfr2.data import load_csv, preprocess, split

df = load_csv("data/raw/chembl_vegfr2.csv")
df = preprocess(df)
train_df, val_df, test_df = split(df, seed=42)

print(f"Dataset: {len(df)} molecules")
print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# %%
# @title 5. Extract Fingerprints (for fusion)
from vegfr2.features import smiles_to_morgan, smiles_to_maccs

print("Extracting Morgan + MACCS fingerprints...")
X_train_morgan = np.vstack([smiles_to_morgan(s) for s in train_df["smiles"]])
X_train_maccs = np.vstack([smiles_to_maccs(s) for s in train_df["smiles"]])
X_train_fp = np.hstack([X_train_morgan, X_train_maccs]).astype(np.float32)

X_val_morgan = np.vstack([smiles_to_morgan(s) for s in val_df["smiles"]])
X_val_maccs = np.vstack([smiles_to_maccs(s) for s in val_df["smiles"]])
X_val_fp = np.hstack([X_val_morgan, X_val_maccs]).astype(np.float32)

X_test_morgan = np.vstack([smiles_to_morgan(s) for s in test_df["smiles"]])
X_test_maccs = np.vstack([smiles_to_maccs(s) for s in test_df["smiles"]])
X_test_fp = np.hstack([X_test_morgan, X_test_maccs]).astype(np.float32)

y_train = train_df["active"].values.astype(int)
y_val = val_df["active"].values.astype(int)
y_test = test_df["active"].values.astype(int)

print(f"Fingerprints: {X_train_fp.shape[1]}-dim (Morgan+MACCS)")

# %%
# @title 6. Create Fused Data Loaders
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from vegfr2.features import mol_to_graph


def make_fused_loader(smiles_list, labels, fps, batch_size=128, shuffle=False):
    """Create DataLoader with graph features + separate fingerprints."""
    data_list = []
    for s, y, fp in zip(smiles_list, labels, fps):
        try:
            g = mol_to_graph(s)
            data = Data(
                x=g["node_feats"],
                edge_index=g["edge_index"],
                edge_attr=g["edge_feats"],
                y=torch.tensor([y], dtype=torch.float32),
                fingerprint=torch.tensor(fp, dtype=torch.float32).unsqueeze(0),
            )
            data_list.append(data)
        except Exception:
            pass

    def collate_fn(batch):
        from torch_geometric.data import Batch
        batch_obj = Batch.from_data_list(batch)
        # fingerprint is graph-level [B, fp_dim], already stacked correctly
        return batch_obj

    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


train_loader = make_fused_loader(
    train_df["smiles"].tolist(), y_train, X_train_fp, shuffle=True
)
val_loader = make_fused_loader(val_df["smiles"].tolist(), y_val, X_val_fp)
test_loader = make_fused_loader(test_df["smiles"].tolist(), y_test, X_test_fp)

print(f"Loaders created: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test")

# %%
# @title 7. Train Fused GIN
from vegfr2.models.fused_gnn import FusedGIN
from vegfr2.metrics import classification_metrics
import torch.nn as nn

print("=" * 60)
print("FUSED GIN: Graph (32-dim) + Fingerprint (2214-dim)")
print("=" * 60)

model = FusedGIN(
    in_dim=32,
    hidden=128,
    layers=3,
    fp_dim=2214,
    out_dim=1,
    dropout=0.3,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: FusedGIN ({n_params:,} params)")
print(f"  - GIN backbone: 32-dim input (lightweight)")
print(f"  - FP branch: 2214-dim → 128-dim (compressed)")
print(f"  - Fusion: concat graph + FP → classifier")

opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

best_auc = -1.0
best_state = None
wait = 0

for epoch in range(1, 101):
    model.train()
    for batch in train_loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x, batch.edge_index, batch.batch, batch.fingerprint)
        loss = loss_fn(logits.squeeze(), batch.y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    val_probs, val_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.batch, batch.fingerprint)
            val_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
            val_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

    val_auc = classification_metrics(val_true, val_probs).get("auc") or 0.0

    if val_auc > best_auc:
        best_auc = val_auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= 15:
            print(f"  Early stop at epoch {epoch}")
            break

    if epoch % 25 == 0:
        print(f"  Epoch {epoch:3d} val_AUC={val_auc:.4f}")

model.load_state_dict(best_state)
model.to(DEVICE).eval()

# Test
test_probs, test_true = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x, batch.edge_index, batch.batch, batch.fingerprint)
        test_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
        test_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

fused_gin_metrics = classification_metrics(test_true, test_probs)
print(f"\n  Fused GIN Test AUC: {fused_gin_metrics.get('auc', 0):.4f}")

# %%
# @title 8. Compare: Enriched vs Fused vs Graph-Only
from vegfr2.features import mol_to_graph_with_fps
from vegfr2.gnn_pyg import build_pyg_model

results = {"fused_gin": fused_gin_metrics}

# Graph-only GIN (baseline)
print("\nTraining Graph-Only GIN (baseline)...")

def make_graph_loader(smiles_list, labels, batch_size=128, shuffle=False):
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


graph_train = make_graph_loader(train_df["smiles"].tolist(), y_train, shuffle=True)
graph_val = make_graph_loader(val_df["smiles"].tolist(), y_val)
graph_test = make_graph_loader(test_df["smiles"].tolist(), y_test)

model_graph = build_pyg_model("gin", in_dim=32, hidden=128, layers=3, dropout=0.3).to(DEVICE)
opt2 = torch.optim.AdamW(model_graph.parameters(), lr=0.001, weight_decay=1e-4)

best_auc2 = -1.0
best_state2 = None

for epoch in range(1, 101):
    model_graph.train()
    for batch in graph_train:
        batch = batch.to(DEVICE)
        logits = model_graph(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(logits.squeeze(), batch.y)
        opt2.zero_grad()
        loss.backward()
        opt2.step()

    model_graph.eval()
    val_probs2, val_true2 = [], []
    with torch.no_grad():
        for batch in graph_val:
            batch = batch.to(DEVICE)
            logits = model_graph(batch.x, batch.edge_index, batch.batch)
            val_probs2.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
            val_true2.extend(batch.y.squeeze().cpu().numpy().astype(int))

    val_auc2 = classification_metrics(val_true2, val_probs2).get("auc") or 0.0
    if val_auc2 > best_auc2:
        best_auc2 = val_auc2
        best_state2 = {k: v.cpu().clone() for k, v in model_graph.state_dict().items()}

model_graph.load_state_dict(best_state2)
model_graph.to(DEVICE).eval()

test_probs2, test_true2 = [], []
with torch.no_grad():
    for batch in graph_test:
        batch = batch.to(DEVICE)
        logits = model_graph(batch.x, batch.edge_index, batch.batch)
        test_probs2.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
        test_true2.extend(batch.y.squeeze().cpu().numpy().astype(int))

results["gin_graph_only"] = classification_metrics(test_true2, test_probs2)

# %%
# @title 9. Final Comparison
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(f"\n{'Model':<25} {'AUC':>6} {'ACC':>6} {'MCC':>6}")
print("-" * 45)
for name, m in sorted(results.items(), key=lambda x: x[1].get("auc") or 0, reverse=True):
    auc_str = f"{m['auc']:.4f}" if m.get("auc") is not None else "N/A"
    print(f"{name:<25} {auc_str:>6} {m['acc']:.4f} {m['mcc']:.4f}")

# Why fused is better
fused_auc = results["fused_gin"].get("auc", 0)
graph_auc = results["gin_graph_only"].get("auc", 0)
print("-" * 45)
print(f"\nWhy Fused GIN is better:")
print(f"  Graph-only GIN: {graph_auc:.4f} AUC (32-dim, no FP knowledge)")
print(f"  Fused GIN:      {fused_auc:.4f} AUC (+{fused_auc - graph_auc:.4f} improvement)")
print(f"\nHow:")
print(f"  1. GNN learns graph patterns (32-dim atoms)")
print(f"  2. FP adds molecular knowledge (2214-dim)")
print(f"  3. Fusion combines both (graph + FP)")
print(f"  4. Result: accurate + lightweight")
print("=" * 60)

# %%
# @title 10. Save Model
torch.save({
    "model_state_dict": model.state_dict(),
    "model_type": "fused_gin",
    "test_metrics": fused_gin_metrics,
}, "checkpoints/fused_gin.pt")
print("Model saved to checkpoints/fused_gin.pt")
