#!/usr/bin/env python3
"""
VEGFR2 Virtual Screening - Full Pipeline (Colab Ready)
======================================================
Single file: install deps, clone repo, run ALL models, show results.
Works on CPU or GPU.

How to use:
  1. Open Google Colab (https://colab.research.google.com)
  2. Upload this file OR paste the contents
  3. Runtime -> Change runtime type -> T4 GPU
  4. Runtime -> Run all
"""

# %% [markdown]
# # VEGFR2 Virtual Screening Pipeline
# ## Full ML + GNN Model Comparison
# Runs on GPU (T4) in ~15-20 minutes.

# %%
# @title 1. Install Dependencies (run first)
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("Installing packages...")
install("rdkit-pypi")
install("torch_geometric")
install("xgboost")
install("optuna")
install("scikit-learn")
install("pandas")
install("numpy")
install("pyyaml")
print("All packages installed!")

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
# @title 5. Extract Fingerprints (Morgan + MACCS)
from vegfr2.features import smiles_to_morgan, smiles_to_maccs

print("Extracting Morgan fingerprints (2048-bit)...")
X_train_morgan = np.vstack([smiles_to_morgan(s) for s in train_df["smiles"]])
X_test_morgan = np.vstack([smiles_to_morgan(s) for s in test_df["smiles"]])

print("Extracting MACCS keys (166-bit)...")
X_train_maccs = np.vstack([smiles_to_maccs(s) for s in train_df["smiles"]])
X_test_maccs = np.vstack([smiles_to_maccs(s) for s in test_df["smiles"]])

X_train_both = np.hstack([X_train_morgan, X_train_maccs])
X_test_both = np.hstack([X_test_morgan, X_test_maccs])

y_train = train_df["active"].values.astype(int)
y_test = test_df["active"].values.astype(int)

print(f"  Morgan: {X_train_morgan.shape[1]}-dim")
print(f"  MACCS:  {X_train_maccs.shape[1]}-dim")
print(f"  Combined: {X_train_both.shape[1]}-dim")

# %%
# @title 6. Train Classical ML Models
from vegfr2.ml_models import train_ml_model, predict_ml_model
from vegfr2.metrics import classification_metrics

results = {}

for name in ["rf", "svm", "xgb"]:
    print(f"\n--- Training {name.upper()} ---")
    model = train_ml_model(name, X_train_both, y_train, seed=42)
    probs = predict_ml_model(model, X_test_both)
    metrics = classification_metrics(y_test.tolist(), probs.tolist())
    results[name] = metrics
    print(f"  AUC={metrics.get('auc', 0):.4f} ACC={metrics['acc']:.4f} MCC={metrics['mcc']:.4f}")

# %%
# @title 7. Train GNN Models on GPU (Enriched Graphs)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from vegfr2.features import mol_to_graph_with_fps
from vegfr2.gnn_pyg import build_pyg_model
from vegfr2.metrics import classification_metrics


def make_enriched_loader(smiles_list, labels, batch_size=128, shuffle=False):
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


def train_gnn(model_name, train_df, val_df, test_df, device, epochs=100, patience=15):
    torch.manual_seed(42)

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

    model = build_pyg_model(
        model_name, in_dim=2246, hidden=128, layers=3, heads=8, dropout=0.3
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model_name} ({n_params:,} params)")

    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-6
    )

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
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

        if epoch % 25 == 0:
            print(f"  Epoch {epoch:3d} val_AUC={val_auc:.4f}")

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
for name in gnn_names:
    print(f"\n--- Training {name.upper()} ---")
    try:
        metrics = train_gnn(name, train_df, val_df, test_df, DEVICE, epochs=100, patience=15)
        results[f"gnn_{name}"] = metrics
        print(
            f"  AUC={metrics.get('auc', 0):.4f} ACC={metrics['acc']:.4f} MCC={metrics['mcc']:.4f}"
        )
    except Exception as e:
        print(f"  ERROR: {e}")

# %%
# @title 8. Train Ensemble Models (GNN + XGBoost)
from vegfr2.models.ensemble import GNNEnsembleClassifier

ensemble_configs = [
    ("gin", "xgb"),
    ("pna", "xgb"),
    ("gin", "rf"),
]

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
        print(
            f"  AUC={metrics.get('auc', 0):.4f} ACC={metrics['acc']:.4f} MCC={metrics['mcc']:.4f}"
        )
    except Exception as e:
        print(f"  ERROR: {e}")

# %%
# @title 9. Final Comparison Table
print("=" * 80)
print("VEGFR2 MODEL COMPARISON - ALL RESULTS")
print("=" * 80)

header = f"{'Model':<30} {'ACC':>6} {'SEN':>6} {'SPE':>6} {'MCC':>6} {'AUC':>6}"
print(header)
print("-" * 80)

sorted_results = sorted(
    results.items(), key=lambda x: x[1].get("auc") or 0, reverse=True
)
for name, m in sorted_results:
    auc_str = f"{m['auc']:.4f}" if m.get("auc") is not None else "N/A"
    print(
        f"{name:<30} {m['acc']:.4f} {m['sen']:.4f} {m['spe']:.4f} {m['mcc']:.4f} {auc_str:>6}"
    )

best_name = sorted_results[0][0]
best_auc = sorted_results[0][1].get("auc", 0)
print("-" * 80)
print(f"  BEST MODEL: {best_name} (AUC={best_auc:.4f})")
print("=" * 80)

# %%
# @title 10. Unit Tests (verify all models work)
import pytest

print("\nRunning test suite...")
exit_code = pytest.main(["tests/", "-v", "--tb=short", "-q"])
print(f"\nTest suite: {'ALL PASSED' if exit_code == 0 else 'SOME FAILED'}")

# %%
# @title 11. Save Results to JSON
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
}

for name, m in results.items():
    output["results"][name] = {
        k: float(v) if isinstance(v, (np.floating, float)) else v
        for k, v in m.items()
        if k != "confusion_matrix"
    }

with open("colab_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to colab_results.json")
print("Done! All models trained and evaluated.")
