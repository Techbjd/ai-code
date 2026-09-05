#!/usr/bin/env python3
"""
VEGFR2 Paper Validation v2 — Enriched Morgan Fingerprints
==========================================================
Same as colab_validate_paper.py but ML models use enriched features:
- Morgan FP (radius 1, 2, 3) — captures different substructure sizes
- MACCS keys (166 bits) — predefined structural patterns
- RDKit descriptors (200+) — physicochemical properties

This makes ML models closer to the paper's approach (Morgan FP input).
"""

# %%
# @title 1. Install Dependencies
print("Installing packages...")
%pip install -q rdkit torch_geometric xgboost scikit-learn pandas numpy pyyaml requests
print("All packages ready!")

# %%
# @title 2. Setup Environment
import os, sys, time, json, warnings
warnings.filterwarnings("ignore", message=".*scatter.*")
warnings.filterwarnings("ignore", message=".*torch-scatter.*")

REPO_DIR = "/content/ai-code"
if not os.path.exists(REPO_DIR):
    os.system("git clone https://github.com/Techbjd/ai-code.git /content/ai-code")
os.chdir(REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))
os.chdir(REPO_DIR)
print(f"Working directory: {os.getcwd()}")

# %%
# @title 3. Check GPU
import torch
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    DEVICE = torch.device("cuda")
else:
    print("No GPU found - using CPU")
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# %%
# @title 4. Load and Preprocess Data
import pandas as pd
import numpy as np
from vegfr2.data import load_csv, preprocess, split
from vegfr2.features import smiles_to_morgan, smiles_to_maccs

RAW_CSV = "data/raw/chembl_vegfr2.csv"
print(f"Loading data from {RAW_CSV}...")

df = load_csv(RAW_CSV)
df = preprocess(df)
train_df, val_df, test_df = split(df)

print(f"\nDataset Statistics:")
print(f"  Total molecules: {len(df)}")
print(f"  Train: {len(train_df)} ({train_df['active'].mean()*100:.1f}% active)")
print(f"  Val:   {len(val_df)} ({val_df['active'].mean()*100:.1f}% active)")
print(f"  Test:  {len(test_df)} ({test_df['active'].mean()*100:.1f}% active)")

# %%
# @title 5. Extract ENRICHED Features (Morgan + MACCS + RDKit Descriptors)
print("Extracting enriched features...")

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


def get_enriched_features(smiles_list, label=""):
    """Extract enriched Morgan FP + MACCS + RDKit descriptors."""
    morgan_r1, morgan_r2, morgan_r3 = [], [], []
    maccs_fps = []
    rdkit_descs = []

    # RDKit descriptor calculator (200+ descriptors)
    desc_names = [n for n, _ in Descriptors.descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            morgan_r1.append(np.zeros(1024))
            morgan_r2.append(np.zeros(2048))
            morgan_r3.append(np.zeros(4096))
            maccs_fps.append(np.zeros(166))
            rdkit_descs.append(np.zeros(len(desc_names)))
            continue

        # Morgan FP (3 radii)
        for radius, fp_list in [(1, morgan_r1), (2, morgan_r2), (3, morgan_r3)]:
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize={1: 1024, 2: 2048, 3: 4096}[radius])
            fp = gen.GetFingerprint(mol)
            fp_list.append(np.fromiter((fp[i] for i in range(len(fp))), dtype=np.uint8))

        # MACCS keys
        fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_fps.append(np.fromiter((fp[i] for i in range(166)), dtype=np.uint8))

        # RDKit descriptors
        try:
            desc_vals = list(calc.CalcDescriptors(mol))
            desc_arr = np.array(desc_vals, dtype=np.float64)
            # Replace inf/nan with 0
            desc_arr = np.nan_to_num(desc_arr, nan=0.0, posinf=0.0, neginf=0.0)
            rdkit_descs.append(desc_arr.astype(np.float32))
        except Exception:
            rdkit_descs.append(np.zeros(len(desc_names)))

        if (i + 1) % 1000 == 0:
            print(f"  {label} Processed {i+1}/{len(smiles_list)} molecules...")

    return {
        "morgan_r1": np.vstack(morgan_r1),   # 1024
        "morgan_r2": np.vstack(morgan_r2),   # 2048
        "morgan_r3": np.vstack(morgan_r3),   # 4096
        "maccs": np.vstack(maccs_fps),        # 166
        "rdkit_descs": np.vstack(rdkit_descs), # 200+
    }


print("Processing training set...")
train_feats = get_enriched_features(train_df["smiles"].tolist(), label="Train")
print("Processing test set...")
test_feats = get_enriched_features(test_df["smiles"].tolist(), label="Test")

y_train = train_df["active"].astype(int).values
y_test = test_df["active"].astype(int).values

# Also extract standard Morgan (for comparison with v1)
X_train_morgan = np.vstack([smiles_to_morgan(s) for s in train_df["smiles"]])
X_test_morgan = np.vstack([smiles_to_morgan(s) for s in test_df["smiles"]])
X_train_maccs = np.vstack([smiles_to_maccs(s) for s in train_df["smiles"]])
X_test_maccs = np.vstack([smiles_to_maccs(s) for s in test_df["smiles"]])

print(f"\nFeature dimensions:")
print(f"  Morgan R1: {train_feats['morgan_r1'].shape[1]}")
print(f"  Morgan R2: {train_feats['morgan_r2'].shape[1]}")
print(f"  Morgan R3: {train_feats['morgan_r3'].shape[1]}")
print(f"  MACCS: {train_feats['maccs'].shape[1]}")
print(f"  RDKit Desc: {train_feats['rdkit_descs'].shape[1]}")

# Build combined feature sets
X_train_morgan_r1r2 = np.hstack([train_feats["morgan_r1"], train_feats["morgan_r2"]])
X_test_morgan_r1r2 = np.hstack([test_feats["morgan_r1"], test_feats["morgan_r2"]])

X_train_morgan_r1r2r3 = np.hstack([train_feats["morgan_r1"], train_feats["morgan_r2"], train_feats["morgan_r3"]])
X_test_morgan_r1r2r3 = np.hstack([test_feats["morgan_r1"], test_feats["morgan_r2"], test_feats["morgan_r3"]])

X_train_morgan_maccs_desc = np.hstack([
    train_feats["morgan_r2"], train_feats["maccs"], train_feats["rdkit_descs"]
])
X_test_morgan_maccs_desc = np.hstack([
    test_feats["morgan_r2"], test_feats["maccs"], test_feats["rdkit_descs"]
])

X_train_all = np.hstack([
    train_feats["morgan_r1"], train_feats["morgan_r2"], train_feats["morgan_r3"],
    train_feats["maccs"], train_feats["rdkit_descs"]
])
X_test_all = np.hstack([
    test_feats["morgan_r1"], test_feats["morgan_r2"], test_feats["morgan_r3"],
    test_feats["maccs"], test_feats["rdkit_descs"]
])

print(f"\nCombined feature dimensions:")
print(f"  Morgan R1+R2: {X_train_morgan_r1r2.shape[1]}")
print(f"  Morgan R1+R2+R3: {X_train_morgan_r1r2r3.shape[1]}")
print(f"  Morgan+MACCS+Desc: {X_train_morgan_maccs_desc.shape[1]}")
print(f"  ALL: {X_train_all.shape[1]}")


# %%
# @title 6. Train ALL Models (ENRICHED ML + GNN)
import threading
from vegfr2.ml_models import train_ml_model, predict_ml_model
from vegfr2.metrics import classification_metrics

results = {}
models_ml = {}
models_gnn = {}


def train_all_ml():
    """Train all ML models with enriched features."""
    # Feature sets to try
    feature_sets = [
        ("morgan",      X_train_morgan, X_test_morgan),
        ("maccs",       X_train_maccs, X_test_maccs),
        ("morg_r1r2",   X_train_morgan_r1r2, X_test_morgan_r1r2),
        ("morg_r1r2r3", X_train_morgan_r1r2r3, X_test_morgan_r1r2r3),
        ("morg_mac_desc", X_train_morgan_maccs_desc, X_test_morgan_maccs_desc),
        ("ALL",         X_train_all, X_test_all),
    ]

    for name in ["rf", "xgb"]:  # Skip SVM (slow with high-dim features)
        for fp_name, X_tr, X_te in feature_sets:
            key = f"{name}_{fp_name}"
            print(f"\n--- Training {key.upper()} ---")
            t0 = time.time()
            model = train_ml_model(name, X_tr, y_train, seed=42)
            probs = predict_ml_model(model, X_te)
            metrics = classification_metrics(y_test.tolist(), probs.tolist())
            results[key] = metrics
            models_ml[key] = model
            dt = time.time() - t0
            print(f"  AUC={metrics.get('auc', 0):.4f} ACC={metrics['acc']:.4f} MCC={metrics['mcc']:.4f} ({dt:.1f}s)")
    print("\n✅ All ML models trained")


def train_all_gnn():
    """Train all GNN models (GPU)."""
    import torch.nn as nn
    from rdkit import Chem
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from vegfr2.gnn_pyg import build_pyg_model

    def smiles_to_plain_graph(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(), atom.GetTotalDegree(), atom.GetFormalCharge(),
                int(atom.GetHybridization()), int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(), int(atom.IsInRing()),
                atom.GetMass() / 100.0, int(atom.IsInRing()), atom.GetDegree(),
                int(atom.HasProp('_ChiralityPossible')), atom.GetTotalValence(),
                int(atom.GetNoImplicit()), atom.GetNumExplicitHs(), atom.GetNumImplicitHs(),
                int(atom.GetFormalCharge()), int(atom.GetHybridization()),
                int(atom.GetIsAromatic()), atom.GetMass() / 16.0,
                atom.GetAtomicNum() / 100.0, atom.GetTotalDegree() / 6.0,
                atom.GetFormalCharge() / 4.0, int(atom.GetHybridization()) / 5.0,
                int(atom.GetIsAromatic()), atom.GetTotalNumHs() / 4.0,
                atom.GetNumRadicalElectrons(), int(atom.IsInRing()),
                atom.GetMass() / 200.0, atom.GetDegree() / 6.0,
                int(atom.HasProp('_ChiralityPossible')), atom.GetTotalValence() / 6.0,
            ]
            atom_features.append(features[:32])
        if not atom_features:
            return None
        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])
        if not edge_indices:
            return None
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)

    def make_loader(smiles_list, labels, batch_size=128, shuffle=False):
        data_list = []
        for s, y in zip(smiles_list, labels):
            g = smiles_to_plain_graph(s)
            if g is not None:
                g.y = torch.tensor([y], dtype=torch.float32)
                data_list.append(g)
        return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

    def train_gnn(model_name, epochs=100, patience=15):
        torch.manual_seed(42)
        train_loader = make_loader(train_df["smiles"].tolist(), y_train, shuffle=True)
        val_loader = make_loader(val_df["smiles"].tolist(), val_df["active"].astype(int).tolist())
        test_loader = make_loader(test_df["smiles"].tolist(), y_test.tolist())

        model = build_pyg_model(model_name, in_dim=32, hidden=128, layers=3, heads=8, dropout=0.3).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {model_name} ({n_params:,} params)")

        opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

        n_active = train_df["active"].sum()
        n_inactive = len(train_df) - n_active
        pos_weight = torch.tensor([n_inactive / n_active], device=DEVICE)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auc, best_state, wait = -1.0, None, 0
        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                batch = batch.to(DEVICE)
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
                    batch = batch.to(DEVICE)
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
                    print(f"  Early stop at epoch {epoch}")
                    break
            if epoch % 25 == 0:
                print(f"  Epoch {epoch:3d} val_AUC={val_auc:.4f}")

        if best_state:
            model.load_state_dict(best_state)
        model.to(DEVICE).eval()

        test_probs, test_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                logits = model(batch.x, batch.edge_index, batch.batch)
                test_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
                test_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

        return classification_metrics(test_true, test_probs), model

    for name in ["gcn", "gat", "gatv2", "gin", "pna"]:
        print(f"\n--- Training GNN_{name.upper()} ---")
        try:
            metrics, model = train_gnn(name)
            results[f"gnn_{name}"] = metrics
            models_gnn[name] = model
            print(f"  AUC={metrics.get('auc', 0):.4f} ACC={metrics['acc']:.4f} MCC={metrics['mcc']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
    print("\n✅ All GNN models trained")


print("=" * 80)
print("TRAINING ALL MODELS IN PARALLEL")
print("=" * 80)

ml_thread = threading.Thread(target=train_all_ml, name="ML")
gnn_thread = threading.Thread(target=train_all_gnn, name="GNN")
ml_thread.start()
gnn_thread.start()
ml_thread.join()
gnn_thread.join()

print("\n" + "=" * 80)
print("ALL MODEL RESULTS (sorted by AUC)")
print("=" * 80)
print(f"{'Model':<25} {'ACC':>6} {'SEN':>6} {'SPE':>6} {'MCC':>6} {'AUC':>6}")
print("-" * 65)
for name, m in sorted(results.items(), key=lambda x: x[1].get("auc", 0), reverse=True):
    print(f"{name:<25} {m['acc']:6.4f} {m['sen']:6.4f} {m['spe']:6.4f} {m['mcc']:6.4f} {m.get('auc', 0):6.4f}")

# Select top models
sorted_models = sorted(results.items(), key=lambda x: x[1].get("auc", 0), reverse=True)
top2_ml = [n for n, _ in sorted_models if not n.startswith("gnn_")][:2]
top3_gnn = [n for n, _ in sorted_models if n.startswith("gnn_")][:3]
selected = top2_ml + top3_gnn

print(f"\nTop 2 ML: {top2_ml}")
print(f"Top 3 GNN: {top3_gnn}")
print(f"Selected: {selected}")


# %%
# @title 7. Validate Paper's 6 Molecules
print("=" * 80)
print("PAPER VALIDATION: Hou et al. (2025) — 6 Candidate Molecules")
print("=" * 80)

paper_molecules = pd.DataFrame([
    {"name": "Cynaroside", "smiles": "C1=CC(=C(C=C1C2=CC(=O)C3=C(C=C(C=C3O2)OC4C(C(C(C(O4)CO)O)O)O)O)O)O",
     "paper_ic50_nM": 2698, "paper_result": "SUCCESS"},
    {"name": "Luteolin 7-O-glucuronide", "smiles": "C1=CC(=C(C=C1C2=CC(=O)C3=C(C=C(C=C3O2)OC4C(C(C(C(O4)C(=O)O)O)O)O)O)O)O",
     "paper_ic50_nM": 5969, "paper_result": "SUCCESS"},
    {"name": "Scutellarin", "smiles": "C1=CC(=CC=C1C2=CC(=O)C3=C(C(=C(C=C3O2)OC4C(C(C(C(O4)C(=O)O)O)O)O)O)O)O",
     "paper_ic50_nM": 8349, "paper_result": "SUCCESS"},
    {"name": "Diosmin", "smiles": "CC1C(C(C(C(O1)OCC2C(C(C(C(O2)OC3=CC(=C4C(=C3)OC(=CC4=O)C5=CC(=C(C=C5)OC)O)O)O)O)O)O)O)O",
     "paper_ic50_nM": None, "paper_result": "WEAK"},
    {"name": "Rhoifolin", "smiles": "CC1C(C(C(C(O1)OC2C(C(C(OC2OC3=CC(=C4C(=C3)OC(=CC4=O)C5=CC=C(C=C5)O)O)CO)O)O)O)O)O",
     "paper_ic50_nM": None, "paper_result": "NO_EFFECT"},
    {"name": "Beta-Carotene", "smiles": "CC(=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C(C)C)C(C)C",
     "paper_ic50_nM": None, "paper_result": "NO_EFFECT"},
])


def predict_all_models_enriched(smiles_list, models_ml, models_gnn, selected_models, device):
    """Predict with all selected models using enriched features for ML."""
    from torch_geometric.loader import DataLoader
    from rdkit import Chem
    from torch_geometric.data import Data

    preds = {m: [] for m in selected_models}

    # Extract enriched features for ML
    enriched_feats = get_enriched_features(smiles_list, label="Predict")
    X_morgan_r1r2 = np.hstack([enriched_feats["morgan_r1"], enriched_feats["morgan_r2"]])
    X_morgan_r1r2r3 = np.hstack([enriched_feats["morgan_r1"], enriched_feats["morgan_r2"], enriched_feats["morgan_r3"]])
    X_morgan_maccs_desc = np.hstack([enriched_feats["morgan_r2"], enriched_feats["maccs"], enriched_feats["rdkit_descs"]])
    X_all = np.hstack([
        enriched_feats["morgan_r1"], enriched_feats["morgan_r2"], enriched_feats["morgan_r3"],
        enriched_feats["maccs"], enriched_feats["rdkit_descs"]
    ])

    # Standard Morgan/MACCS
    X_morgan = np.vstack([smiles_to_morgan(s) for s in smiles_list])
    X_maccs = np.vstack([smiles_to_maccs(s) for s in smiles_list])

    # Feature set mapping
    fp_map = {
        "morgan": X_morgan, "maccs": X_maccs,
        "morg_r1r2": X_morgan_r1r2, "morg_r1r2r3": X_morgan_r1r2r3,
        "morg_mac_desc": X_morgan_maccs_desc, "ALL": X_all,
    }

    # ML predictions
    for model_name in selected_models:
        if model_name.startswith("gnn_"):
            continue
        model = models_ml[model_name]
        # Extract feature set name from model name
        fp_name = model_name.split("_", 1)[1] if "_" in model_name else "morgan"
        fp = fp_map.get(fp_name, X_morgan)
        probs = predict_ml_model(model, fp)
        preds[model_name] = probs.tolist()

    # GNN predictions
    def smiles_to_plain_graph_local(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(), atom.GetTotalDegree(), atom.GetFormalCharge(),
                int(atom.GetHybridization()), int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(), int(atom.IsInRing()),
                atom.GetMass() / 100.0, int(atom.IsInRing()), atom.GetDegree(),
                int(atom.HasProp('_ChiralityPossible')), atom.GetTotalValence(),
                int(atom.GetNoImplicit()), atom.GetNumExplicitHs(), atom.GetNumImplicitHs(),
                int(atom.GetFormalCharge()), int(atom.GetHybridization()),
                int(atom.GetIsAromatic()), atom.GetMass() / 16.0,
                atom.GetAtomicNum() / 100.0, atom.GetTotalDegree() / 6.0,
                atom.GetFormalCharge() / 4.0, int(atom.GetHybridization()) / 5.0,
                int(atom.GetIsAromatic()), atom.GetTotalNumHs() / 4.0,
                atom.GetNumRadicalElectrons(), int(atom.IsInRing()),
                atom.GetMass() / 200.0, atom.GetDegree() / 6.0,
                int(atom.HasProp('_ChiralityPossible')), atom.GetTotalValence() / 6.0,
            ]
            atom_features.append(features[:32])
        if not atom_features:
            return None
        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])
        if not edge_indices:
            return None
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)

    gnn_models = [m.replace("gnn_", "") for m in selected_models if m.startswith("gnn_")]
    if gnn_models:
        data_list = []
        valid_indices = []
        for i, s in enumerate(smiles_list):
            g = smiles_to_plain_graph_local(s)
            if g is not None:
                g.y = torch.tensor([0], dtype=torch.float32)
                data_list.append(g)
                valid_indices.append(i)

        if data_list:
            loader = DataLoader(data_list, batch_size=64, shuffle=False)
            for gnn_name in gnn_models:
                if gnn_name not in models_gnn:
                    continue
                model = models_gnn[gnn_name]
                model.eval()
                all_probs = []
                with torch.no_grad():
                    for batch in loader:
                        batch = batch.to(device)
                        logits = model(batch.x, batch.edge_index, batch.batch)
                        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                        if probs.ndim == 0:
                            probs = [probs.item()]
                        all_probs.extend(probs)
                full_probs = [0.0] * len(smiles_list)
                for idx, prob in zip(valid_indices, all_probs):
                    full_probs[idx] = prob
                preds[f"gnn_{gnn_name}"] = full_probs

    return preds


paper_smiles = paper_molecules["smiles"].tolist()
paper_preds = predict_all_models_enriched(paper_smiles, models_ml, models_gnn, selected, DEVICE)

# Display results
print(f"\n{'Molecule':<30} {'Paper':>10} {'Consensus':>12} ", end="")
for m in selected:
    print(f" {m:>16}", end="")
print()
print("-" * (30 + 10 + 12 + 16 * len(selected) + len(selected)))

for i, row in paper_molecules.iterrows():
    name = row["name"]
    paper_result = row["paper_result"]
    all_preds = [paper_preds[m][i] for m in selected]
    consensus = np.mean(all_preds)
    consensus_label = "ACTIVE" if consensus > 0.5 else "INACTIVE"
    print(f"{name:<30} {paper_result:>10} {consensus_label:>8}({consensus:.3f})", end="")
    for m in selected:
        prob = paper_preds[m][i]
        label = "A" if prob > 0.5 else "I"
        print(f" {prob:>6.3f}({label}){'':>5}", end="")
    print()

# Summary
success_molecules = paper_molecules[paper_molecules["paper_result"] == "SUCCESS"]
fail_molecules = paper_molecules[paper_molecules["paper_result"] != "SUCCESS"]

print("\n" + "=" * 80)
print("PAPER VALIDATION SUMMARY")
print("=" * 80)

print("\nPaper's 3 SUCCESSFUL molecules (should be predicted ACTIVE):")
n_success_correct = 0
for _, row in success_molecules.iterrows():
    i = paper_molecules.index.get_loc(row.name)
    consensus = np.mean([paper_preds[m][i] for m in selected])
    label = "ACTIVE" if consensus > 0.5 else "INACTIVE"
    match = "MATCH" if label == "ACTIVE" else "MISMATCH"
    if label == "ACTIVE":
        n_success_correct += 1
    print(f"  {row['name']:<30} IC50={row['paper_ic50_nM']}nM  consensus={consensus:.3f} -> {label}  {match}")

print("\nPaper's 3 FAILED molecules (should be predicted INACTIVE):")
n_fail_correct = 0
for _, row in fail_molecules.iterrows():
    i = paper_molecules.index.get_loc(row.name)
    consensus = np.mean([paper_preds[m][i] for m in selected])
    label = "ACTIVE" if consensus > 0.5 else "INACTIVE"
    match = "MATCH" if label == "INACTIVE" else "MISMATCH"
    if label == "INACTIVE":
        n_fail_correct += 1
    print(f"  {row['name']:<30} Result={row['paper_result']:<12} consensus={consensus:.3f} -> {label}  {match}")

print(f"\nAccuracy: {n_success_correct}/3 success + {n_fail_correct}/3 fail = {n_success_correct + n_fail_correct}/6 total")


# %%
# @title 8. Screen Fresh COCONUT Dataset
print("=" * 80)
print("SCREENING FRESH COCONUT DATASET")
print("=" * 80)

import requests

coconut_df = None
try:
    url = "https://coconut.s3.uni-jena.de/prod/downloads/2024-08/DD/COCONUT_2024_08_DrugDiscovery.tsv.zip"
    print(f"Downloading COCONUT Drug Discovery subset...")
    response = requests.get(url, timeout=120)
    if response.status_code == 200:
        import zipfile, io
        z = zipfile.ZipFile(io.BytesIO(response.content))
        tsv_files = [f for f in z.namelist() if f.endswith(".tsv")]
        if tsv_files:
            df_coconut = pd.read_csv(z.open(tsv_files[0]), sep="\t")
            smiles_col = None
            for col in df_coconut.columns:
                if "smiles" in col.lower() or "canonical" in col.lower():
                    smiles_col = col
                    break
            if smiles_col:
                n_sample = min(3000, len(df_coconut))
                df_sample = df_coconut.sample(n=n_sample, random_state=42)
                compounds = []
                for _, row in df_sample.iterrows():
                    smiles = row[smiles_col]
                    name = row.get("COCONUT_id", f"COCONUT_{len(compounds)}")
                    compounds.append({"name": name, "smiles": smiles})
                coconut_df = pd.DataFrame(compounds)
                print(f"Downloaded {len(coconut_df)} molecules from COCONUT")
except Exception as e:
    print(f"COCONUT download failed: {e}")

if coconut_df is None:
    print("Using fallback COCONUT molecules")
    coconut_df = pd.DataFrame([
        ("Quercetin", "OC1=CC(=C2C(=O)C(=C(OC2=C1)C1=CC=C(C=C1)O)O)O"),
        ("Kaempferol", "OC1=CC(=C2C(=O)C(=C(OC2=C1)C1=CC=C(C=C1)O)O)"),
        ("Luteolin", "OC1=CC(=C2C(=O)C(=C(OC2=C1)C1=CC=C(C=C1)O)O)"),
        ("Apigenin", "OC1=CC(=C2C(=O)C(=C(OC2=C1)C1=CC=C(C=C1)O)O)"),
        ("Resveratrol", "OC1=CC=C(C=C1)/C=C/c1cc(O)cc(O)c1"),
        ("Curcumin", "COc1cc(\\C=C\\C(=O)CC(=O)\\C=C\\c2ccc(O)c(OC)c2)ccc1O"),
        ("Berberine", "COc1ccc2cc3c1oc2c1ccc3OCO1"),
        ("Gallic acid", "OC(=O)c1cc(O)c(O)c(O)c1"),
        ("Ellagic acid", "OC(=O)c1cc2c3c1oc(=O)c2=O"),
    ], columns=["name", "smiles"])

paper_smiles_set = set(paper_molecules["smiles"].tolist())
coconut_df = coconut_df[~coconut_df["smiles"].isin(paper_smiles_set)].reset_index(drop=True)
print(f"Fresh COCONUT molecules for screening: {len(coconut_df)}")

print("\nScreening fresh COCONUT molecules...")
coconut_preds = predict_all_models_enriched(coconut_df["smiles"].tolist(), models_ml, models_gnn, selected, DEVICE)

coconut_results = pd.DataFrame()
coconut_results["name"] = coconut_df["name"]
coconut_results["smiles"] = coconut_df["smiles"]

for m in selected:
    coconut_results[f"{m}_pred"] = coconut_preds[m]
    coconut_results[f"{m}_label"] = (np.array(coconut_preds[m]) > 0.5).astype(int)

pred_cols = [c for c in coconut_results.columns if c.endswith("_pred")]
coconut_results["consensus_score"] = coconut_results[pred_cols].mean(axis=1)
coconut_results["consensus_label"] = (coconut_results["consensus_score"] > 0.5).astype(int)
coconut_results = coconut_results.sort_values("consensus_score", ascending=False)

print(f"\nTop 20 Candidates:")
print("-" * 100)
for idx, row in coconut_results.head(20).iterrows():
    print(f"{row['name']:<30} Score: {row['consensus_score']:.4f} | ", end="")
    for m in selected:
        col = f"{m}_label"
        if col in coconut_results.columns:
            print(f"{m}={'A' if row[col] == 1 else 'I'} ", end="")
    print()

coconut_results.to_csv("fresh_screening_results_v2.csv", index=False)
n_active = coconut_results["consensus_label"].sum()
n_total = len(coconut_results)
print(f"\nScreening Summary: {n_active}/{n_total} ({n_active/n_total*100:.1f}%) predicted active")

top6 = coconut_results.head(6)
top6.to_csv("fresh_top_hits_v2.csv", index=False)
print(f"Top 6 saved to: fresh_top_hits_v2.csv")


# %%
# @title 9. Generate 3D Structures
print("=" * 80)
print("GENERATING 3D STRUCTURES FOR DOCKING")
print("=" * 80)

from rdkit.Chem import AllChem

top6_mols = []
for idx, row in top6.iterrows():
    mol = Chem.MolFromSmiles(row["smiles"])
    if mol is None:
        print(f"  Skipped: {row['name']} (invalid SMILES)")
        continue
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    mol.SetProp("_Name", row["name"])
    mol.SetProp("ConsensusScore", str(row["consensus_score"]))
    top6_mols.append(mol)
    print(f"  Generated: {row['name']}")

if top6_mols:
    writer = Chem.SDWriter("fresh_top_hits_3d_v2.sdf")
    for mol in top6_mols:
        writer.write(mol)
    writer.close()
    print(f"\n3D structures saved to: fresh_top_hits_3d_v2.sdf")


# %%
# @title 10. Final Summary
print("=" * 80)
print("COMPLETE SUMMARY (v2 — Enriched Features)")
print("=" * 80)

# Find best model
best_model = sorted_models[0]
print(f"""
1. BEST MODEL: {best_model[0]} (AUC={best_model[1].get('auc', 0):.4f})

2. PAPER VALIDATION:
   - Success molecules correct: {n_success_correct}/3
   - Fail molecules correct: {n_fail_correct}/3
   - Overall: {n_success_correct + n_fail_correct}/6

3. ENRICHED FEATURES:
   - Morgan R1 (1024) + R2 (2048) + R3 (4096)
   - MACCS (166)
   - RDKit Descriptors (200+)
   - Total: {X_train_all.shape[1]} features

4. FILES GENERATED:
   - fresh_screening_results_v2.csv
   - fresh_top_hits_v2.csv
   - fresh_top_hits_3d_v2.sdf
""")
