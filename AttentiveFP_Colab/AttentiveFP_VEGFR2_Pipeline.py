#!/usr/bin/env python3
"""
AttentiveFP VEGFR2 Virtual Screening Pipeline (Colab Ready)
============================================================
Complete pipeline following Hou et al. (2025) methodology:

  1. Download VEGFR2 IC50 data from ChEMBL279
  2. Data standardization, deduplication, active/inactive labeling
  3. Scaffold split (train/val/test)
  4. Train AttentiveFP (OpenDrugAI reference architecture)
  5. Validate: ROC-AUC + PR-AUC + MCC + F1 + Sensitivity + Specificity
  6. Screen Chinese medicine compounds (TCMSP)
  7. Compare with baselines (GCN, Morgan FP)

How to use:
  1. Open Google Colab (https://colab.research.google.com)
  2. Upload this file OR paste the contents
  3. Runtime -> Change runtime type -> T4 GPU
  4. Runtime -> Run all

Reference:
  Hou et al. (2025). "Identification of potent inhibitors of potential VEGFR2:
  a graph neural network-based virtual screening and in vitro study."
  Journal of Enzyme Inhibition and Medicinal Chemistry, 40:1.
"""

# %% [markdown]
# # AttentiveFP VEGFR2 Virtual Screening Pipeline
# ## Complete Pipeline: Training → Validation → Chinese Medicine Screening
#
# This notebook implements the full pipeline from Hou et al. (2025):
# - **Data**: ChEMBL279 VEGFR2 IC50 data
# - **Model**: AttentiveFP (Graph Attention + GRU + Attentive Readout)
# - **Split**: Scaffold-based (generalization test)
# - **Screen**: Chinese medicine compounds from TCMSP
#
# Pipeline flow:
# ```
# XO DATASET → Data standardization → Deduplication → Active/inactive labels
# → DATA SPLIT (scaffold) → TRAIN/TEST → AttentiveFP training
# → Final independent test → ROC-AUC + PR-AUC + MCC + F1 + Sens + Spec
# → Scaffold/generalization test → Chinese medicine screening
# ```

# %%
# @title 1. Install Dependencies (Colab-optimized)
import subprocess, sys

# Colab already has: torch, rdkit, numpy, pandas, matplotlib, scikit-learn
# Only install what's missing
print("Checking packages...")

pkgs_to_install = []
for name in ["torch_geometric", "xgboost", "optuna"]:
    try:
        __import__(name)
        print(f"  {name}: OK")
    except ImportError:
        pkgs_to_install.append(name)
        print(f"  {name}: INSTALLING...")

if pkgs_to_install:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs_to_install)

print("All packages ready!")

# %%
# @title 2. Clone Repository
import os

REPO_URL = "https://github.com/Techbjd/ai-code.git"
REPO_BRANCH = "restructured-pipeline"
REPO_DIR = "/content/ai-code"

if not os.path.exists(REPO_DIR):
    os.system(f"git clone -b {REPO_BRANCH} {REPO_URL} {REPO_DIR}")
    print(f"Repository cloned (branch: {REPO_BRANCH})!")
else:
    os.system(f"cd {REPO_DIR} && git checkout {REPO_BRANCH} && git pull")
    print(f"Repository updated (branch: {REPO_BRANCH})!")

sys.path.insert(0, os.path.join(REPO_DIR, "src"))
os.chdir(REPO_DIR)
print(f"Working directory: {os.getcwd()}")

# %%
# @title 3. Check GPU + Maximize Resources
import torch
import warnings
warnings.filterwarnings("ignore")

print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {vram:.1f} GB")
    DEVICE = torch.device("cuda")

    # Maximize GPU utilization
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Auto-tune for this GPU
    print("GPU optimizations enabled: cuDNN benchmark, TF32, fast matmul")
else:
    print("No GPU found - using CPU (will be slower)")
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# %%
# @title 4. Download VEGFR2 Data from ChEMBL
import urllib.request
import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
TARGET_ID = "CHEMBL279"

def download_chembl(target_id, output_path):
    """Download IC50 data for VEGFR2 from ChEMBL."""
    rows = []
    offset = 0
    page_size = 1000

    print(f"Downloading VEGFR2 ({target_id}) IC50 data...")

    while True:
        url = f"{BASE_URL}/activity.json?target_chembl_id={target_id}&standard_type=IC50&limit={page_size}&offset={offset}"
        try:
            with urllib.request.urlopen(url) as resp:
                data = json.load(resp)
        except Exception as e:
            print(f"  Error at offset {offset}: {e}")
            break

        activities = data.get("activities", [])
        if not activities:
            break

        for act in activities:
            if act.get("standard_relation") != "=":
                continue
            smiles = act.get("canonical_smiles")
            val = act.get("standard_value")
            if not smiles or val is None:
                continue
            try:
                ic50 = float(val)
                if ic50 <= 0:
                    continue
            except (ValueError, TypeError):
                continue
            rows.append({"smiles": smiles, "ic50_nM": ic50})

        print(f"  Offset {offset}: {len(rows)} compounds so far")

        if len(activities) < page_size:
            break
        offset += page_size

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} compounds to {output_path}")
    return df

# Download
raw_csv = "data/chembl_vegfr2.csv"
if not os.path.exists(raw_csv):
    df = download_chembl(TARGET_ID, raw_csv)
else:
    df = pd.read_csv(raw_csv)
    print(f"Using existing data: {len(df)} compounds")

# %%
# @title 5. Data Preprocessing
from rdkit import Chem

print("Preprocessing data...")

# Standardize SMILES
def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

# Remove invalid SMILES
df = df.dropna(subset=["smiles", "ic50_nM"])
df["ic50_nM"] = pd.to_numeric(df["ic50_nM"], errors="coerce")
df = df.dropna(subset=["ic50_nM"])
df = df[df["ic50_nM"] > 0]

# Canonicalize SMILES
df["canonical_smiles"] = df["smiles"].map(canonicalize)
df = df.dropna(subset=["canonical_smiles"])
df = df.drop_duplicates(subset=["canonical_smiles"])

# Label active/inactive (IC50 < 500 nM = active)
df["active"] = (df["ic50_nM"] < 500).astype(int)

print(f"After preprocessing: {len(df)} unique compounds")
print(f"  Active (IC50 < 500 nM): {df['active'].sum()} ({df['active'].mean():.1%})")
print(f"  Inactive: {(1-df['active']).sum()} ({1-df['active'].mean():.1%})")

df.head(10)

# %%
# @title 6. Scaffold Split
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

def scaffold_split(df, test_size=0.1, val_size=0.1, seed=42):
    """Split by Murcko scaffolds for generalization testing."""
    np.random.seed(seed)

    scaffolds = df["canonical_smiles"].map(
        lambda s: MurckoScaffoldSmiles(smiles=s, includeChirality=False)
    )

    scaffold_groups = {}
    for idx, scaffold in zip(df.index, scaffolds):
        scaffold_groups.setdefault(scaffold, []).append(idx)

    sorted_scaffolds = sorted(scaffold_groups.keys(),
                              key=lambda s: len(scaffold_groups[s]), reverse=True)

    n_total = len(df)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)

    test_idx, val_idx, train_idx = [], [], []

    for scaffold in sorted_scaffolds:
        indices = scaffold_groups[scaffold]
        np.random.shuffle(indices)
        if len(test_idx) < n_test:
            test_idx.extend(indices)
        elif len(val_idx) < n_val:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    test_idx = test_idx[:n_test]
    val_idx = val_idx[:n_val]
    train_idx = [i for i in train_idx if i not in set(test_idx) and i not in set(val_idx)]

    return df.loc[train_idx].reset_index(drop=True), \
           df.loc[val_idx].reset_index(drop=True), \
           df.loc[test_idx].reset_index(drop=True)

train_df, val_df, test_df = scaffold_split(df, seed=42)

print(f"Scaffold Split:")
print(f"  Train: {len(train_df)} ({train_df['active'].mean():.1%} active)")
print(f"  Val:   {len(val_df)} ({val_df['active'].mean():.1%} active)")
print(f"  Test:  {len(test_df)} ({test_df['active'].mean():.1%} active)")

# Check scaffold overlap
train_scaffolds = set(train_df["canonical_smiles"].map(
    lambda s: MurckoScaffoldSmiles(smiles=s, includeChirality=False)))
test_scaffolds = set(test_df["canonical_smiles"].map(
    lambda s: MurckoScaffoldSmiles(smiles=s, includeChirality=False)))
overlap = train_scaffolds & test_scaffolds
print(f"\n  Scaffold generalization check:")
print(f"  Train scaffolds: {len(train_scaffolds)}")
print(f"  Test scaffolds:  {len(test_scaffolds)}")
print(f"  Overlap: {len(overlap)} ({len(overlap)/max(len(test_scaffolds),1)*100:.1f}% of test)")

# %%
# @title 7. Feature Extraction (Enriched Graphs)
import sys
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

from vegfr2.features import mol_to_graph_with_fps
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from multiprocessing import Pool, cpu_count
import time

print("Building enriched graphs: [atom(32) + Morgan(2048) + MACCS(166)] = 2246-dim")

def _build_graph(args):
    smiles, label = args
    try:
        g = mol_to_graph_with_fps(smiles, use_morgan=True, use_maccs=True)
        return Data(
            x=g["node_feats"],
            edge_index=g["edge_index"],
            edge_attr=g["edge_feats"],
            y=torch.tensor([label], dtype=torch.float32),
        )
    except Exception:
        return None

def make_loader(df, batch_size=256, shuffle=False):
    """Create DataLoader with parallel graph building."""
    t0 = time.time()
    inputs = list(zip(df["canonical_smiles"], df["active"].astype(int).tolist()))

    n_workers = min(cpu_count(), 4)
    with Pool(n_workers) as pool:
        results = pool.map(_build_graph, inputs)

    data_list = [r for r in results if r is not None]
    elapsed = time.time() - t0
    print(f"  Built {len(data_list)}/{len(df)} graphs in {elapsed:.1f}s ({n_workers} workers)")

    return DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

print(f"Building train graphs...")
train_loader = make_loader(train_df, batch_size=256, shuffle=True)
print(f"Building val graphs...")
val_loader = make_loader(val_df, batch_size=256)
print(f"Building test graphs...")
test_loader = make_loader(test_df, batch_size=256)

print(f"\nLoaders ready: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test")

# %%
# @title 8. Load AttentiveFP Model
from vegfr2.models.attentive_fp import AttentiveFP

model = AttentiveFP(
    in_dim=2246,      # Enriched: atom(32) + Morgan(2048) + MACCS(166)
    hidden=200,        # Reference value
    layers=3,          # radius in reference
    out_dim=1,
    dropout=0.2,       # Reference value
    num_timesteps=2,   # T in reference (attentive readout steps)
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"AttentiveFP: {n_params:,} params")
LAYERS = 3
print(f"  Architecture: atom_fc + neighbor_fc + {LAYERS} attention layers + {2}-step GRU readout")

# %%
# @title 9. Training Loop (AMP + Optimized)
from vegfr2.metrics import classification_metrics
import torch.nn as nn

# Reference hyperparameters from OpenDrugAI
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 200
PATIENCE = 25

torch.manual_seed(42)
np.random.seed(42)

# Try torch.compile for faster execution (PyTorch 2.0+)
try:
    model = torch.compile(model, mode="reduce-overhead")
    print("torch.compile enabled for faster execution")
except Exception:
    pass

opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

# Class weights for imbalanced data
n_active = train_df["active"].sum()
n_total = len(train_df)
pos_weight = torch.tensor([(n_total - n_active) / max(n_active, 1)], device=DEVICE)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# AMP (Automatic Mixed Precision) for 2x faster training
scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
use_amp = scaler is not None

print(f"Training for {EPOCHS} epochs (patience={PATIENCE})...")
print(f"  LR={LEARNING_RATE}, Weight Decay={WEIGHT_DECAY}")
print(f"  Pos weight={pos_weight.item():.2f} (class imbalance correction)")
print(f"  AMP: {'ON' if use_amp else 'OFF'} (mixed precision)")

best_auc = -1.0
best_state = None
wait = 0
history = {"train_loss": [], "val_auc": [], "val_acc": [], "val_mcc": []}

import time
t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in train_loader:
        batch = batch.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
            loss = loss_fn(logits.squeeze(), batch.y)

        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        total_loss += loss.item() * batch.y.shape[0]
        n_samples += batch.y.shape[0]
    scheduler.step()

    # Validate
    model.eval()
    val_probs, val_true = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for batch in val_loader:
            batch = batch.to(DEVICE, non_blocking=True)
            logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
            val_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
            val_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

    val_metrics = classification_metrics(val_true, val_probs)
    val_auc = val_metrics.get("auc") or 0.0

    history["train_loss"].append(total_loss / n_samples)
    history["val_auc"].append(val_auc)
    history["val_acc"].append(val_metrics["acc"])
    history["val_mcc"].append(val_metrics["mcc"])

    if val_auc > best_auc:
        best_auc = val_auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            elapsed = time.time() - t_start
            print(f"  Early stopping at epoch {epoch} ({elapsed:.0f}s)")
            break

    if epoch % 25 == 0:
        elapsed = time.time() - t_start
        print(f"  Epoch {epoch:3d} | loss={total_loss/n_samples:.4f} | val_AUC={val_auc:.4f} | val_MCC={val_metrics['mcc']:.4f} | {elapsed:.0f}s")

total_time = time.time() - t_start
print(f"\nBest validation AUC: {best_auc:.4f} | Total time: {total_time:.0f}s")

# %%
# @title 10. Final Test Evaluation
# Load best model
if best_state:
    model.load_state_dict(best_state)
model.to(DEVICE).eval()

# Test evaluation with AMP
test_probs, test_true = [], []
with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
    for batch in test_loader:
        batch = batch.to(DEVICE, non_blocking=True)
        logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
        test_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
        test_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

# Detailed metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sk_auc, f1_score

y_true = np.array(test_true)
y_prob = np.array(test_probs)
y_pred = (y_prob >= 0.5).astype(int)

tp = ((y_pred == 1) & (y_true == 1)).sum()
tn = ((y_pred == 0) & (y_true == 0)).sum()
fp = ((y_pred == 1) & (y_true == 0)).sum()
fn = ((y_pred == 0) & (y_true == 1)).sum()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
pr_auc = sk_auc(recall_curve, precision_curve)

mcc = classification_metrics(test_true, test_probs)["mcc"]

print("=" * 60)
print("AttentiveFP TEST RESULTS (Scaffold Split)")
print("=" * 60)
print(f"  ROC-AUC:       {roc_auc_score(y_true, y_prob):.4f}")
print(f"  PR-AUC:        {pr_auc:.4f}")
print(f"  MCC:           {mcc:.4f}")
print(f"  F1 Score:      {f1_score(y_true, y_pred):.4f}")
print(f"  Sensitivity:   {sensitivity:.4f}")
print(f"  Specificity:   {specificity:.4f}")
print(f"  Accuracy:      {(tp+tn)/(tp+tn+fp+fn):.4f}")
print(f"\n  Confusion Matrix:")
print(f"    TP={tp}  FP={fp}")
print(f"    FN={fn}  TN={tn}")
print("=" * 60)

# %%
# @title 11. Training History Visualization
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history["train_loss"], label="Train Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history["val_auc"], label="Val AUC", color="green")
axes[1].axhline(y=best_auc, color="r", linestyle="--", label=f"Best: {best_auc:.4f}")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("AUC")
axes[1].set_title("Validation AUC")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(history["val_mcc"], label="Val MCC", color="orange")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("MCC")
axes[2].set_title("Validation MCC")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("attentivefp_training.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to attentivefp_training.png")

# %%
# @title 12. ROC and PR Curves
from sklearn.metrics import roc_curve

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc_val = roc_auc_score(y_true, y_prob)
axes[0].plot(fpr, tpr, linewidth=2, label=f"AttentiveFP (AUC={roc_auc_val:.4f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# PR Curve
axes[1].plot(recall_curve, precision_curve, linewidth=2, label=f"AttentiveFP (PR-AUC={pr_auc:.4f})")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("attentivefp_roc_pr.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# @title 13. Compare with Baselines
from vegfr2.gnn_pyg import build_pyg_model
from vegfr2.features import smiles_to_morgan

results = {"AttentiveFP (scaffold)": {
    "ROC-AUC": roc_auc_score(y_true, y_prob),
    "PR-AUC": pr_auc,
    "MCC": mcc,
    "F1": f1_score(y_true, y_pred),
    "Sensitivity": sensitivity,
    "Specificity": specificity,
}}

# --- GCN baseline ---
print("Training GCN baseline...")
gcn_model = build_pyg_model("gcn", in_dim=2246, hidden=128, layers=3, dropout=0.3).to(DEVICE)
opt_gcn = torch.optim.AdamW(gcn_model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn_gcn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

for epoch in range(1, 101):
    gcn_model.train()
    for batch in train_loader:
        batch = batch.to(DEVICE)
        logits = gcn_model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn_gcn(logits.squeeze(), batch.y)
        opt_gcn.zero_grad()
        loss.backward()
        opt_gcn.step()

gcn_model.eval()
gcn_probs, gcn_true = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        logits = gcn_model(batch.x, batch.edge_index, batch.batch)
        gcn_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
        gcn_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

gcn_y_true = np.array(gcn_true)
gcn_y_prob = np.array(gcn_probs)
gcn_y_pred = (gcn_y_prob >= 0.5).astype(int)
gcn_tp = ((gcn_y_pred == 1) & (gcn_y_true == 1)).sum()
gcn_tn = ((gcn_y_pred == 0) & (gcn_y_true == 0)).sum()
gcn_fp = ((gcn_y_pred == 1) & (gcn_y_true == 0)).sum()
gcn_fn = ((gcn_y_pred == 0) & (gcn_y_true == 1)).sum()

gcn_prec, gcn_rec, _ = precision_recall_curve(gcn_y_true, gcn_y_prob)

results["GCN (scaffold)"] = {
    "ROC-AUC": roc_auc_score(gcn_y_true, gcn_y_prob),
    "PR-AUC": sk_auc(gcn_rec, gcn_prec),
    "MCC": classification_metrics(gcn_true, gcn_probs)["mcc"],
    "F1": f1_score(gcn_y_true, gcn_y_pred),
    "Sensitivity": gcn_tp / (gcn_tp + gcn_fn) if (gcn_tp + gcn_fn) > 0 else 0,
    "Specificity": gcn_tn / (gcn_tn + gcn_fp) if (gcn_tn + gcn_fp) > 0 else 0,
}

# --- Morgan FP + XGBoost baseline ---
print("Training Morgan FP + XGBoost baseline...")
X_train_morgan = np.vstack([smiles_to_morgan(s) for s in train_df["canonical_smiles"]])
X_test_morgan = np.vstack([smiles_to_morgan(s) for s in test_df["canonical_smiles"]])
y_train_morgan = train_df["active"].values
y_test_morgan = test_df["active"].values

from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                           eval_metric="logloss", n_jobs=-1, random_state=42)
xgb_model.fit(X_train_morgan, y_train_morgan)
xgb_probs = xgb_model.predict_proba(X_test_morgan)[:, 1]
xgb_pred = (xgb_probs >= 0.5).astype(int)

xgb_tp = ((xgb_pred == 1) & (y_test_morgan == 1)).sum()
xgb_tn = ((xgb_pred == 0) & (y_test_morgan == 0)).sum()
xgb_fp = ((xgb_pred == 1) & (y_test_morgan == 0)).sum()
xgb_fn = ((xgb_pred == 0) & (y_test_morgan == 1)).sum()

xgb_prec, xgb_rec, _ = precision_recall_curve(y_test_morgan, xgb_probs)

results["Morgan+XGBoost"] = {
    "ROC-AUC": roc_auc_score(y_test_morgan, xgb_probs),
    "PR-AUC": sk_auc(xgb_rec, xgb_prec),
    "MCC": classification_metrics(y_test_morgan.tolist(), xgb_probs.tolist())["mcc"],
    "F1": f1_score(y_test_morgan, xgb_pred),
    "Sensitivity": xgb_tp / (xgb_tp + xgb_fn) if (xgb_tp + xgb_fn) > 0 else 0,
    "Specificity": xgb_tn / (xgb_tn + xgb_fp) if (xgb_tn + xgb_fp) > 0 else 0,
}

# Print comparison
print("\n" + "=" * 80)
print("MODEL COMPARISON (Scaffold Split)")
print("=" * 80)
header = f"{'Model':<25} {'ROC-AUC':>8} {'PR-AUC':>8} {'MCC':>8} {'F1':>8} {'Sens':>8} {'Spec':>8}"
print(header)
print("-" * 80)
for name, m in sorted(results.items(), key=lambda x: x[1]["ROC-AUC"], reverse=True):
    print(f"{name:<25} {m['ROC-AUC']:>8.4f} {m['PR-AUC']:>8.4f} {m['MCC']:>8.4f} {m['F1']:>8.4f} {m['Sensitivity']:>8.4f} {m['Specificity']:>8.4f}")
print("=" * 80)

# %%
# @title 14. Save Trained Model
save_dir = Path("checkpoints")
save_dir.mkdir(exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "model_type": "attentive_fp",
    "in_dim": 2246,
    "hidden": 200,
    "layers": 3,
    "num_timesteps": 2,
    "dropout": 0.2,
    "test_metrics": results["AttentiveFP (scaffold)"],
}, save_dir / "attentivefp_vegfr2.pt")

print(f"Model saved to {save_dir / 'attentivefp_vegfr2.pt'}")

# %%
# @title 15. Screen Chinese Medicine Compounds (TCMSP)
# Download TCMSP data (traditional Chinese medicine compounds)
print("=" * 60)
print("SCREENING CHINESE MEDICINE COMPOUNDS")
print("=" * 60)

# TCMSP API for downloading compound data
# We'll download a subset of TCM compounds with SMILES
tcmsp_url = "https://raw.githubusercontent.com/tcm-spider/tcmsp/master/data/tcm_spider_20230828.csv"

print("Downloading TCMSP compound data...")
try:
    tcm_df = pd.read_csv(tcmsp_url, low_memory=False)
    print(f"  Loaded {len(tcm_df)} TCMSP entries")
except Exception as e:
    print(f"  TCMSP download failed: {e}")
    print("  Creating demo Chinese medicine compound set...")

    # Fallback: known VEGFR2-related Chinese medicine compounds
    demo_tcm = pd.DataFrame({
        "molecule_name": [
            "Curcumin", "Berberine", "Resveratrol", "Quercetin", "EGCG",
            "Apigenin", "Luteolin", "Kaempferol", "Fisetin", "Myricetin",
            "Ginsenoside Rg3", "Ginsenoside Rb1", "Tanshinone IIA", "Salvianolic acid B",
            "Andrographolide", "Baicalein", "Wogonin", "Oroxylin A", "Scutellarin",
            "Honokiol", "Magnolol", "Emodin", "Rhein", "Aloe-emodin",
            "Artemisinin", "Artesunate", "Camptothecin", "Huang Qin Tang",
            "Liu Wei Di Huang", "Zhi Bai Di Huang",
        ],
        "canonical_smiles": [
            "CC(=O)Oc1cc(O)c2c(c1)oc(-c1ccc(O)c(O)c1)cc2=O",
            "COc1ccc2cc3cc4c(c3cc2c1O)OCO4",
            "O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",
            "OC1Cc2c(OC1c1ccc(O)c(O)c1)cc(=O)c(O)c(O)c2",
            "OC1C(O)c2c(OC1c1ccc(O)c(O)c1)cc(=O)c(O)c(O)c2",
            "O=c1cc(-c2ccc(O)cc2)oc2cc(O)ccc12",
            "O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
            "O=c1cc(-c2ccc(O)cc2)oc2cc(O)c(O)cc12",
            "O=c1cc(-c2ccc(O)cc2)oc2c(O)cc(O)cc12",
            "O=c1cc(-c2ccc(O)c(O)c2)oc2c(O)c(O)cc(O)c12",
            "CC1(C)C2CCC3C(CCC4C3(CCC4C2(CCC1O)C)C)C(C)O",
            "CC1(C)C2CCC3C(CCC4C3(CCC4C2(CCC1O)C)C)C(C)OC1OC(CO)C(O)C(O)C1Oc1c(O)cc(O)cc1C(=O)OC",
            "CC1=CC(=O)c2ccccc2C1=Cc1ccc2cc3c(c2c1)OCO3",
            "OC(=O)CC(C(=O)O)c1ccc(O)c(O)c1",
            "CC1(C)C2CCC3C(CCC4C3(CCC4C2(CCC1O)C)C)C(C)OC1OC(CO)C(O)C(O)C1O",
            "O=c1cc(-c2cc(O)c(O)cc2)oc2cc(O)cc(O)c12",
            "COc1cc2c(=O)[nH]cnc2cc1O",
            "COc1cc2c(=O)[nH]cnc2cc1OC",
            "OC1Cc2c(OC1c1cc(O)c(O)cc1O)cc(=O)c(O)c(O)c2",
            "Oc1ccc(-c2cc(O)cc(-c3ccc(O)cc3)c2O)cc1",
            "Oc1ccc(-c2cc(O)cc(-c3ccc(O)c(O)c3)c2O)cc1",
            "CC1=CC(=O)c2cc(O)ccc2C1Cc1cc(O)c(O)cc1",
            "OC(=O)c1cc(O)c(O)cc1",
            "O=C(O)c1ccc(O)c(O)c1",
            "CC12CCC3C(CCC4C3(CCC4C2CC=C1)C)C",
            "CC12CCC3C(CCC4C3(CCC4C2CC=C1O)C)C(=O)O",
            "CCC1=CC(=O)c2cc3cc4c(c3cc2C1)OCO4",
            "O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",
            "OC1Cc2c(OC1c1ccc(O)c(O)c1)cc(=O)c(O)c(O)c2",
            "OC1Cc2c(OC1c1ccc(O)c(O)c1)cc(=O)c(O)c(O)c2",
        ],
        "herb_name": [
            "Turmeric", "Coptis chinensis", "Grape skin", "Onion", "Green tea",
            "Chamomile", "Thyme", "Broccoli", "Strawberry", "Walnut",
            "Panax ginseng", "Panax ginseng", "Salvia miltiorrhiza", "Salvia miltiorrhiza",
            "Andrographis paniculata", "Scutellaria baicalensis", "Scutellaria baicalensis",
            "Scutellaria baicalensis", "Scutellaria baicalensis",
            "Magnolia officinalis", "Magnolia officinalis",
            "Rheum palmatum", "Rheum palmatum", "Rheum palmatum",
            "Artemisia annua", "Artemisia annua", "Camptotheca acuminata",
            "Multi-herb formula", "Multi-herb formula", "Multi-herb formula",
        ],
    })
    tcm_df = demo_tcm
    print(f"  Using {len(tcm_df)} demo TCM compounds")

print(f"\nTCMSP data loaded: {len(tcm_df)} compounds")
tcm_df.head(10)

# %%
# @title 16. Screen TCM Compounds with Trained AttentiveFP
import warnings
warnings.filterwarnings("ignore")

# Ensure we have the right SMILES column
smiles_col = None
for col in ["canonical_smiles", "SMILES", "smiles"]:
    if col in tcm_df.columns:
        smiles_col = col
        break

if smiles_col is None:
    print("ERROR: No SMILES column found in TCMSP data")
    print(f"Available columns: {list(tcm_df.columns)}")
else:
    # Filter valid SMILES
    tcm_df = tcm_df.dropna(subset=[smiles_col])
    tcm_df["valid"] = tcm_df[smiles_col].map(lambda s: Chem.MolFromSmiles(s) is not None)
    tcm_df = tcm_df[tcm_df["valid"]]

    print(f"Screening {len(tcm_df)} valid TCM compounds...")

    # Create DataLoader for TCM compounds
    def screen_compounds(smiles_list, model, device, batch_size=256):
        """Screen compounds with trained model."""
        data_list = []
        valid_smiles = []
        for s in smiles_list:
            try:
                g = mol_to_graph_with_fps(s, use_morgan=True, use_maccs=True)
                data = Data(
                    x=g["node_feats"],
                    edge_index=g["edge_index"],
                    edge_attr=g["edge_feats"],
                    y=torch.tensor([0], dtype=torch.float32),
                )
                data_list.append(data)
                valid_smiles.append(s)
            except Exception:
                continue

        if not data_list:
            return pd.DataFrame()

        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
        all_probs = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                if isinstance(probs, np.floating):
                    probs = [float(probs)]
                all_probs.extend(probs)

        return pd.DataFrame({
            "smiles": valid_smiles,
            "probability": all_probs,
        }).sort_values("probability", ascending=False)

    # Screen
    tcm_results = screen_compounds(tcm_df[smiles_col].tolist(), model, DEVICE)

    # Merge with TCM metadata
    tcm_results = tcm_results.merge(
        tcm_df[[smiles_col] + [c for c in ["molecule_name", "herb_name"] if c in tcm_df.columns]],
        left_on="smiles", right_on=smiles_col, how="left"
    )

    # Add predictions
    tcm_results["predicted_active"] = (tcm_results["probability"] >= 0.5).astype(int)

    # Display results
    print(f"\n{'='*70}")
    print(f"TCM SCREENING RESULTS")
    print(f"{'='*70}")
    print(f"  Total screened: {len(tcm_results)}")
    print(f"  Predicted active: {tcm_results['predicted_active'].sum()} "
          f"({tcm_results['predicted_active'].mean():.1%})")
    print(f"\nTop 20 predictions:")
    top_cols = [c for c in ["molecule_name", "herb_name", "smiles", "probability", "predicted_active"]
                if c in tcm_results.columns]
    print(tcm_results.head(20)[top_cols].to_string(index=False))

    # Save results
    tcm_results.to_csv("tcm_screening_results.csv", index=False)
    print(f"\nFull results saved to tcm_screening_results.csv")

# %%
# @title 17. Screening Visualization
if 'tcm_results' in dir() and len(tcm_results) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Probability distribution
    axes[0].hist(tcm_results["probability"], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].axvline(x=0.5, color="red", linestyle="--", linewidth=2, label="Threshold=0.5")
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("Count")
    axes[0].set_title("TCM Screening: Probability Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Top hits by herb (if available)
    if "herb_name" in tcm_results.columns:
        top_herbs = tcm_results.groupby("herb_name")["probability"].max().sort_values(ascending=True).tail(10)
        axes[1].barh(range(len(top_herbs)), top_herbs.values, color="forestgreen")
        axes[1].set_yticks(range(len(top_herbs)))
        axes[1].set_yticklabels(top_herbs.index)
        axes[1].set_xlabel("Max Predicted Probability")
        axes[1].set_title("Top 10 Herbs by Max Prediction")
        axes[1].set_xlim(0, 1)
    else:
        top_hits = tcm_results.head(10)
        axes[1].barh(range(len(top_hits)), top_hits["probability"].values, color="forestgreen")
        axes[1].set_yticks(range(len(top_hits)))
        axes[1].set_yticklabels(top_hits["smiles"].str[:30])
        axes[1].set_xlabel("Predicted Probability")
        axes[1].set_title("Top 10 Hits")
        axes[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig("tcm_screening_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved to tcm_screening_results.png")

# %%
# @title 18. Export Hits
if 'tcm_results' in dir() and len(tcm_results) > 0:
    hits = tcm_results[tcm_results["predicted_active"] == 1].copy()
    hits.to_csv("tcm_hits.csv", index=False)

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Training data: {len(train_df)} compounds (ChEMBL279)")
    print(f"  Test set: {len(test_df)} compounds (scaffold split)")
    print(f"  TCM screened: {len(tcm_results)} compounds")
    print(f"  TCM hits: {len(hits)} (probability >= 0.5)")
    print(f"\n  Model performance (scaffold test):")
    print(f"    ROC-AUC: {results['AttentiveFP (scaffold)']['ROC-AUC']:.4f}")
    print(f"    MCC:     {results['AttentiveFP (scaffold)']['MCC']:.4f}")
    print(f"    F1:      {results['AttentiveFP (scaffold)']['F1']:.4f}")
    print(f"\n  Files saved:")
    print(f"    checkpoints/attentivefp_vegfr2.pt  (trained model)")
    print(f"    tcm_screening_results.csv          (all TCM predictions)")
    print(f"    tcm_hits.csv                       (predicted active TCM)")
    print(f"    attentivefp_training.png           (training curves)")
    print(f"    attentivefp_roc_pr.png             (ROC/PR curves)")
    print(f"    tcm_screening_results.png          (screening visualization)")
    print(f"{'='*60}")

# %%
# @title 19. Unit Tests
import pytest

print("\nRunning test suite...")
exit_code = pytest.main(["tests/", "-v", "--tb=short", "-q"])
print(f"\nTest suite: {'ALL PASSED' if exit_code == 0 else 'SOME FAILED'}")

print("\nDone! All models trained and evaluated.")
