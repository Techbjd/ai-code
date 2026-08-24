# VEGFR2 Virtual Screening Pipeline (Hou et al. reproduction)

This repository reproduces the machine learning and graph neural network (GNN) virtual screening pipeline for **VEGFR2 inhibitors** described in:
> Shengzhen Hou et al. (2025). *"Identification of potent inhibitors of potential VEGFR2: a graph neural network-based virtual screening and in vitro study."* **Journal of Enzyme Inhibition and Medicinal Chemistry**, 40:1. DOI: 10.1080/14756366.2025.2518192

## Features Implemented
- **Pure PyTorch GNNs:** From-scratch, DGL-free implementations of **GCN**, **GAT**, and **MPNN** matching the paper's equations (1)-(4).
- **Classical ML Models:** **RandomForest**, **SVM**, and **XGBoost** trained on fingerprints.
- **Multiple Fingerprint Types:**
  - **Morgan Fingerprints** (2048-bit circular fingerprints)
  - **MACCS Structural Keys** (166-bit predefined structural patterns)
- **Feature Combinations for Enhanced Prediction:**
  - `morgan_only` — Morgan fingerprints alone (2048-dim)
  - `maccs_only` — MACCS keys alone (166-dim)
  - `gnn_only` — GNN embeddings alone (64-dim)
  - `morgan_maccs` — Morgan + MACCS concatenated (2214-dim)
  - `gnn_morgan` — GNN embeddings + Morgan fingerprints (2112-dim)
  - `gnn_morgan_maccs` — GNN + Morgan + MACCS (2278-dim)
- **GNN Embedding Extraction:** Extract fixed-size molecular embeddings from trained GNN models for combination with traditional fingerprints.
- **Strict Data Pipeline:** Automated download from ChEMBL279, deduplication rules (conflicting entries dropped entirely, identical ones kept once), and stratified `8:1:1` train/val/test splits.
- **GPU-Only Guards:** Hard CUDA requirements enforced at all training and screening entrypoints.
- **Virtual Screening CLI:** Re-loads any trained checkpoint and screens external SMILES libraries (outputs probabilities + binary hits).

---

## 1. Project Structure
```
README.md
requirements.txt
configs/config.yaml
src/vegfr2/
  ├── __init__.py
  ├── device.py        # Strict GPU guard
  ├── data.py          # Preprocessing & deduplication
  ├── features.py      # Morgan, MACCS, GNN embeddings, feature combinations
  ├── metrics.py       # ACC, SEN, SPE, MCC, AUC
  ├── ml_models.py     # RF, SVM, XGBoost
  ├── gnn_models.py    # GCN, GAT, MPNN nn.Modules
  ├── types.py         # Type definitions for graph batches
  └── hpo.py           # Optuna HPO interface (lazy imported)
scripts/
  ├── download_data.py # Fetch ChEMBL target CSV
  ├── train.py         # Main train & eval script
  └── screen.py        # Virtual screening script
tests/                 # 49 Unit tests (runnable on CPU)
```

---

## 2. Installation
Ensure you have an environment with RDKit and PyTorch installed.
```bash
pip install -r requirements.txt
```

---

## 3. Usage

### A. Download & Preprocess ChEMBL Data
Downloads active/inactive VEGFR2 compounds from ChEMBL target ID `CHEMBL279`:
```bash
python scripts/download_data.py --out data/raw/chembl_vegfr2.csv
```
*Note:* If network connectivity is unavailable, use `--fallback` with a local CSV.

### B. Train Models (GPU Required)
Trains the specified model or all models (RF, SVM, XGB, GCN, GAT, MPNN) on CUDA:
```bash
python scripts/train.py --config configs/config.yaml --model all
```

#### Training Combined Feature Models
Train ML models with enhanced feature combinations:
```bash
# Morgan + MACCS fingerprints with Random Forest
python scripts/train.py --model morgan_maccs --ml-model rf

# GNN embeddings + Morgan fingerprints with XGBoost
python scripts/train.py --model gnn_morgan --ml-model xgb --gnn-model gcn

# GNN + Morgan + MACCS with SVM
python scripts/train.py --model gnn_morgan_maccs --ml-model svm --gnn-model mpnn
```

To run Optuna-based Hyperparameter Optimization for GNNs, append `--hpo`.

### C. Screen a Library (GPU Required)
Rank a library of SMILES (e.g. TCM database TargetMol) to identify potential hits:
```bash
python scripts/screen.py --model runs/gcn/best.pt --input library.csv --output hits.csv --threshold 0.9
```

---

## 4. Feature Combination Methods

The pipeline supports six feature representation methods:

| Method | Description | Dimension |
|--------|-------------|-----------|
| `morgan_only` | Standard Morgan circular fingerprints | 2048 |
| `maccs_only` | MACCS structural keys | 166 |
| `gnn_only` | GNN hidden layer embeddings | 64 |
| `morgan_maccs` | Morgan + MACCS concatenated | 2214 |
| `gnn_morgan` | GNN embeddings + Morgan | 2112 |
| `gnn_morgan_maccs` | GNN + Morgan + MACCS | 2278 |

### Why Combine Features?
Deep learning models (GNNs) alone may not always outperform traditional fingerprint-based methods. By combining:
- **GNN embeddings** capture learned molecular representations
- **Morgan fingerprints** encode circular substructure patterns
- **MACCS keys** provide predefined structural pattern matching

This hybrid approach leverages the strengths of each representation for improved classification performance.

---

## 5. Test Suite (CPU-Runnable)
Unit tests run on CPU with synthetic structures to verify math correctness and preprocessing integrity without requiring a GPU or network access:
```bash
pytest -v
```

### Test Input/Output Results

| Test Name | Input | Expected Output | Status |
|-----------|-------|-----------------|--------|
| `test_morgan_shape_dtype` | SMILES `"CCO"` | Shape (2048,), uint8, sum > 0 | PASSED |
| `test_morgan_deterministic` | Same SMILES twice | Identical arrays | PASSED |
| `test_morgan_invalid_raises` | Invalid SMILES | ValueError | PASSED |
| `test_maccs_shape_dtype` | SMILES `"CCO"` | Shape (166,), uint8, sum > 0 | PASSED |
| `test_maccs_deterministic` | Same SMILES twice | Identical arrays | PASSED |
| `test_maccs_invalid_raises` | Invalid SMILES | ValueError | PASSED |
| `test_maccs_different_molecules` | Different SMILES | Different arrays | PASSED |
| `test_mol_to_graph_dimensions` | `"CCO"` (3 atoms) | nodes=3, node_feats=(3,32), edges=(2,4) | PASSED |
| `test_collate_graphs_batching` | 2 graphs | Proper offsets, node_batch correct | PASSED |
| `test_combine_features_basic` | 2 arrays of shape (2,2) & (2,3) | (2,5) combined | PASSED |
| `test_get_feature_dim` | All 6 methods | Correct dimensions | PASSED |
| `test_extract_gnn_embedding_shape` | GCN + `"CCO"` | Shape (64,) float32 | PASSED |
| `test_extract_gnn_embeddings_batch_shape` | 3 molecules | Shape (3, 64) | PASSED |
| `test_combined_features_morgan_maccs` | Morgan + MACCS | Shape (1, 2214) | PASSED |
| `test_combined_features_gnn_morgan` | GNN + Morgan | Shape (1, 2112) | PASSED |
| `test_combined_features_gnn_morgan_maccs` | GNN + Morgan + MACCS | Shape (1, 2278) | PASSED |

**Total: 49 tests passed, 0 failed**

---

## 6. How It's Achieved

### MACCS Fingerprints (`features.py:50-67`)
MACCS keys are 166 predefined structural patterns implemented via RDKit's `MACCSkeys.GenMACCSKeys()`. Each bit indicates presence/absence of a specific molecular feature (e.g., ring systems, functional groups).

### GNN Embedding Extraction (`features.py:211-307`)
The `extract_gnn_embedding()` function:
1. Converts SMILES to molecular graph via `mol_to_graph()`
2. Passes through the GNN's message-passing layers
3. Applies mean pooling over all node representations
4. Returns fixed-size embedding (e.g., 64-dim for hidden=64)

### Feature Combination (`features.py:309-349`)
The `combine_features()` function concatenates feature arrays along the feature dimension:
```python
combined = combine_features(gnn_emb, morgan_fp, maccs_fp)
# Shape: (n_samples, 64 + 2048 + 166) = (n_samples, 2278)
```

### Training Pipeline (`train.py:291-411`)
The `train_ml_combined()` function:
1. Extracts GNN embeddings if method includes GNN
2. Computes Morgan/MACCS fingerprints as needed
3. Concatenates all feature types
4. Trains classical ML model (RF/SVM/XGBoost) on combined features
