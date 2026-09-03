#!/usr/bin/env python3
"""
VEGFR2 Virtual Screening - Pre-Training Pipeline (Colab Ready)
=============================================================
Adds self-supervised pre-training to the existing pipeline.
Pre-train on unlabeled molecules, then fine-tune on VEGFR2.

How to use:
  1. Open Google Colab (https://colab.research.google.com)
  2. Upload this file OR paste the contents
  3. Runtime -> Change runtime type -> T4 GPU
  4. Runtime -> Run all
"""

# %% [markdown]
# # VEGFR2 Pre-Training Pipeline
# ## Self-Supervised Pre-Training + Fine-Tuning + Prediction
# Runs on GPU (T4) in ~20-30 minutes.

# %%
# @title 1. Install Dependencies (run first)

print("Installing packages...")

%pip install -q rdkit torch_geometric xgboost optuna scikit-learn pandas numpy pyyaml

print("All packages ready!")

# %%
# @title 2. Clone Repository (latest version with pre-training)
import os
import sys

REPO_URL = "https://github.com/Techbjd/ai-code.git"
REPO_DIR = "/content/ai-code"

if not os.path.exists(REPO_DIR):
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
    print("Repository cloned!")
else:
    # Pull latest changes to get pre-training code
    os.system(f"cd {REPO_DIR} && git pull")
    print("Repository updated to latest version!")

sys.path.insert(0, os.path.join(REPO_DIR, "src"))
os.chdir(REPO_DIR)
print(f"Working directory: {os.getcwd()}")

# %%
# @title 3. Check GPU
import torch
import warnings
warnings.filterwarnings("ignore", message=".*scatter.*")

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
# @title 4. Load Data
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
# @title 5. Self-Supervised Pre-Training (Contrastive Learning)
from vegfr2.pretrain import SelfSupervisedPretrainer

# Use ALL molecules for pre-training (no labels needed)
all_smiles = df["smiles"].tolist()
print(f"Pre-training on {len(all_smiles)} unlabeled molecules...")

pretrainer = SelfSupervisedPretrainer(
    model_name="gin",           # gcn, gat, gatv2, mpnn, gin, pna, graph_transformer
    method="contrastive",       # contrastive or masked
    hidden=128,
    layers=3,
    heads=8,
    dropout=0.3,
    lr=0.001,
    batch_size=128,
    seed=42,
)

history = pretrainer.pretrain(
    smiles_list=all_smiles,
    epochs=100,                 # More epochs = better representations
    patience=20,
    device=DEVICE,
    verbose=True,
)

print(f"\nPre-training complete!")
print(f"  Final loss: {history['train_loss'][-1]:.4f}")

# %%
# @title 6. Save Pre-Trained Model
import json
from pathlib import Path

output_dir = Path("checkpoints")
output_dir.mkdir(exist_ok=True)

pretrained_path = output_dir / "pretrained_gin_contrastive.pt"
pretrainer.save_pretrained(pretrained_path)
print(f"Pre-trained model saved to: {pretrained_path}")

# %%
# @title 7. Fine-Tune on VEGFR2 Activity Prediction
import torch.nn as nn
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


# Load pre-trained weights into fresh GNN
model_name = "gin"
model = build_pyg_model(model_name, in_dim=2246, hidden=128, layers=3, heads=8, dropout=0.3)

# Load pre-trained GNN weights (ignore pre-training head)
pretrained_state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
if "gnn_state_dict" in pretrained_state:
    model.load_state_dict(pretrained_state["gnn_state_dict"], strict=False)
    print("Loaded pre-trained GNN weights!")
else:
    print("Warning: No GNN weights found in checkpoint")

model = model.to(DEVICE)

# Create data loaders
train_loader = make_enriched_loader(
    train_df["smiles"].tolist(),
    train_df["active"].astype(int).tolist(),
    shuffle=True,
)
val_loader = make_enriched_loader(
    val_df["smiles"].tolist(),
    val_df["active"].astype(int).tolist(),
)
test_loader = make_enriched_loader(
    test_df["smiles"].tolist(),
    test_df["active"].astype(int).tolist(),
)

# Fine-tune with lower learning rate
EPOCHS = 100
PATIENCE = 15
LR = 0.0001  # 10x lower than pre-training

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

n_active = train_df["active"].sum()
n_inactive = len(train_df) - n_active
pos_weight = torch.tensor([n_inactive / n_active], device=DEVICE)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_auc = -1.0
best_state = None
wait = 0

print(f"\nFine-tuning on VEGFR2 activity ({EPOCHS} epochs, LR={LR})...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    for batch in train_loader:
        batch = batch.to(DEVICE)
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
            batch = batch.to(DEVICE)
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
        if wait >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    if epoch % 25 == 0:
        print(f"  Epoch {epoch:3d} val_AUC={val_auc:.4f}")

if best_state is not None:
    model.load_state_dict(best_state)
model.to(DEVICE).eval()

print(f"\nFine-tuning complete! Best val AUC: {best_auc:.4f}")

# %%
# @title 8. Evaluate on Test Set
test_probs, test_true = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        logits = forward_model(model, model_name, batch)
        test_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
        test_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

test_metrics = classification_metrics(test_true, test_probs)
print("\n" + "=" * 60)
print("TEST SET RESULTS (Pre-trained + Fine-tuned)")
print("=" * 60)
print(f"  AUC:  {test_metrics.get('auc', 0):.4f}")
print(f"  ACC:  {test_metrics['acc']:.4f}")
print(f"  SEN:  {test_metrics['sen']:.4f}")
print(f"  SPE:  {test_metrics['spe']:.4f}")
print(f"  MCC:  {test_metrics['mcc']:.4f}")
print("=" * 60)

# %%
# @title 9. Compare: Pre-trained vs No Pre-training
print("\nTraining GIN WITHOUT pre-training for comparison...")

model_no_pretrain = build_pyg_model(model_name, in_dim=2246, hidden=128, layers=3, heads=8, dropout=0.3)
model_no_pretrain = model_no_pretrain.to(DEVICE)

opt2 = torch.optim.AdamW(model_no_pretrain.parameters(), lr=0.001, weight_decay=1e-4)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=EPOCHS, eta_min=1e-6)

best_auc2 = -1.0
best_state2 = None
wait2 = 0

for epoch in range(1, EPOCHS + 1):
    model_no_pretrain.train()
    for batch in train_loader:
        batch = batch.to(DEVICE)
        logits = forward_model(model_no_pretrain, model_name, batch)
        loss = loss_fn(logits.squeeze(), batch.y)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    scheduler2.step()

    model_no_pretrain.eval()
    val_probs2, val_true2 = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            logits = forward_model(model_no_pretrain, model_name, batch)
            val_probs2.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
            val_true2.extend(batch.y.squeeze().cpu().numpy().astype(int))

    val_auc2 = classification_metrics(val_true2, val_probs2).get("auc") or 0.0

    if val_auc2 > best_auc2:
        best_auc2 = val_auc2
        best_state2 = {k: v.cpu().clone() for k, v in model_no_pretrain.state_dict().items()}
        wait2 = 0
    else:
        wait2 += 1
        if wait2 >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

if best_state2 is not None:
    model_no_pretrain.load_state_dict(best_state2)
model_no_pretrain.to(DEVICE).eval()

test_probs2, test_true2 = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        logits = forward_model(model_no_pretrain, model_name, batch)
        test_probs2.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
        test_true2.extend(batch.y.squeeze().cpu().numpy().astype(int))

test_metrics2 = classification_metrics(test_true2, test_probs2)

print("\n" + "=" * 60)
print("COMPARISON: Pre-trained vs No Pre-training")
print("=" * 60)
print(f"  {'Model':<30} {'AUC':>6} {'ACC':>6} {'MCC':>6}")
print(f"  {'-'*50}")
print(f"  {'GIN + Pre-training':<30} {test_metrics.get('auc', 0):.4f} {test_metrics['acc']:.4f} {test_metrics['mcc']:.4f}")
print(f"  {'GIN (no pre-training)':<30} {test_metrics2.get('auc', 0):.4f} {test_metrics2['acc']:.4f} {test_metrics2['mcc']:.4f}")
print(f"  {'-'*50}")
improvement = test_metrics.get('auc', 0) - test_metrics2.get('auc', 0)
print(f"  Pre-training improvement: {improvement:+.4f} AUC")
print("=" * 60)

# %%
# @title 10. Predict on New Molecules
def predict_new_molecules(smiles_list, model, device):
    """Predict activity for new SMILES strings."""
    model.eval()

    # Convert to enriched graphs
    data_list = []
    for s in smiles_list:
        g = mol_to_graph_with_fps(s, use_morgan=True, use_maccs=True)
        data = Data(
            x=g["node_feats"],
            edge_index=g["edge_index"],
            edge_attr=g["edge_feats"],
            y=torch.tensor([0], dtype=torch.float32),  # dummy label
        )
        data_list.append(data)

    loader = DataLoader(data_list, batch_size=64, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = forward_model(model, model_name, batch)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            all_probs.extend(probs)

    return np.array(all_probs)


# Example: predict on some molecules
example_smiles = [
    "CCO",                    # Ethanol
    "c1ccccc1",               # Benzene
    "CC(=O)O",                # Acetic acid
    "CC1=CC=CC=C1",           # Toluene
    "c1ccc2c(c1)ccc1ccccc12", # Phenanthrene
]

print("\nExample predictions:")
print("-" * 50)
probs = predict_new_molecules(example_smiles, model, DEVICE)
for smi, prob in zip(example_smiles, probs):
    label = "ACTIVE" if prob > 0.5 else "INACTIVE"
    print(f"  {smi:<30} {prob:.4f} ({label})")

# %%
# @title 11. Save Fine-Tuned Model
finetuned_path = output_dir / "finetuned_gin_pretrained.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "model_name": model_name,
    "pretrained_path": str(pretrained_path),
    "test_metrics": test_metrics,
    "finetune_config": {
        "lr": LR,
        "epochs": EPOCHS,
        "patience": PATIENCE,
    },
}, finetuned_path)
print(f"Fine-tuned model saved to: {finetuned_path}")

# %%
# @title 12. Load and Predict (Reusable Code)
# Use this cell to load a saved model and predict on new data

def load_and_predict(smiles_list, model_path, device):
    """Load saved model and predict on new SMILES."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = build_pyg_model(
        checkpoint["model_name"],
        in_dim=2246,
        hidden=128,
        layers=3,
        heads=8,
        dropout=0.3,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    return predict_new_molecules(smiles_list, model, device)


# Example usage:
# new_smiles = ["CCCO", "c1ccc(O)cc1"]
# probs = load_and_predict(new_smiles, str(finetuned_path), DEVICE)

# %%
# @title 13. Unit Tests
import pytest

print("\nRunning test suite...")
exit_code = pytest.main(["tests/test_pretrain.py", "-v", "--tb=short", "-q"])
print(f"\nTest suite: {'ALL PASSED' if exit_code == 0 else 'SOME FAILED'}")

# %%
# @title 14. Summary
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"  Pre-trained model: {pretrained_path}")
print(f"  Fine-tuned model:  {finetuned_path}")
print(f"  Test AUC:          {test_metrics.get('auc', 0):.4f}")
print(f"  Improvement:       {improvement:+.4f} AUC")
print("=" * 60)
print("\nTo use later:")
print(f"  1. Load: torch.load('{finetuned_path}')")
print(f"  2. Predict: load_and_predict(['CCO'], model_path, device)")
