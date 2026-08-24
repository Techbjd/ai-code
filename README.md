# VEGFR2 Virtual Screening Pipeline

Reproduction of the machine learning and GNN virtual screening pipeline for **VEGFR2 inhibitors** described in:
> Shengzhen Hou et al. (2025). *"Identification of potent inhibitors of potential VEGFR2: a graph neural network-based virtual screening and in vitro study."* **Journal of Enzyme Inhibition and Medicinal Chemistry**, 40:1. DOI: 10.1080/14756366.2025.2518192

## Key Innovation: Enriched Graphs

Every GNN model receives **enriched node features** — fingerprints injected into every atom node:
```
[atom_features(32) + Morgan(2048) + MACCS(166)] = 2246-dim per node
```
This gives the GNN access to fingerprint knowledge during message passing, combining the strengths of both representations.

---

## Project Structure

```
src/vegfr2/
├── __init__.py              # Package exports
├── types.py                 # GraphBatch, GraphSample TypedDicts
├── device.py                # Strict GPU enforcement
├── data.py                  # CSV loading, labeling, deduplication, splitting
├── data_pipeline.py         # SMILES → enriched graphs → PyTorch tensors
├── features.py              # Morgan, MACCS, graph construction, enriched graphs
├── metrics.py               # ACC, SEN, SPE, MCC, AUC
├── ml_models.py             # RF, SVM, XGBoost
├── gnn_models.py            # Pure PyTorch GCN, GAT, MPNN
├── gnn_pyg.py               # PyG factory, datasets, train/predict API
├── hpo.py                   # Optuna HPO (lazy import)
├── sklearn_api.py           # GNNClassifier, GNNRegressor, EnsembleClassifier
└── models/
    ├── __init__.py           # Model registry
    ├── _base.py              # Shared checkpoint utilities
    ├── gcn.py                # GCN_PyG — Graph Convolutional Network
    ├── gat.py                # GAT_PyG — Graph Attention Network
    ├── gatv2.py              # GATv2_PyG — Dynamic Attention (strictly > GAT)
    ├── mpnn.py               # MPNN_PyG — Message Passing Neural Network
    ├── gin.py                # GIN_PyG — Graph Isomorphism Network
    ├── pna.py                # PNA_PyG — Principal Neighbourhood Aggregation
    ├── graph_transformer.py  # GraphTransformer_PyG — Global self-attention
    └── ensemble.py           # GNNEnsembleClassifier — GNN + ML ensemble
```

---

## Models

### GNN Architectures (7 models)

| Model | File | Description |
|-------|------|-------------|
| **GCN** | `models/gcn.py` | Graph Convolutional Network with LayerNorm |
| **GAT** | `models/gat.py` | Multi-head additive attention |
| **GATv2** | `models/gatv2.py` | Dynamic attention (strictly more expressive than GAT) |
| **MPNN** | `models/mpnn.py` | Edge-MLP + GRU update |
| **GIN** | `models/gin.py` | Most expressive MPNN, MLP aggregation, JK connections |
| **PNA** | `models/pna.py` | 4 aggregators + 3 scalers + residual connections |
| **Graph Transformer** | `models/graph_transformer.py` | Global self-attention + edge bias + FFN |

### Classical ML Models (3 models)

| Model | Description |
|-------|-------------|
| **Random Forest** | 300 trees, Morgan fingerprints |
| **SVM** | RBF kernel, C=10, Morgan fingerprints |
| **XGBoost** | 400 estimators, depth=6, Morgan fingerprints |

### Ensemble

| Model | Description |
|-------|-------------|
| **GNNEnsembleClassifier** | GNN embeddings + Morgan + MACCS → XGBoost/RF |

---

## Installation

```bash
# Core (ML models)
pip install -e .

# With GNN support
pip install -e ".[gnn]"

# With all features (GNN + HPO)
pip install -e ".[all]"

# Development
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+, CUDA GPU, RDKit, PyTorch, PyTorch Geometric

---

## Usage

### 1. Download & Preprocess Data
```bash
python scripts/download_data.py --out data/raw/chembl_vegfr2.csv
```

### 2. Train a Single Model
```bash
python scripts/train.py --model gin --epochs 300
```

### 3. Train All Models (Advanced Pipeline)
```bash
# Train everything
python scripts/train_all.py

# Train only advanced GNNs (GIN, PNA, GraphTransformer, GATv2)
python scripts/train_all.py --group advanced

# With hyperparameter optimization
python scripts/train_all.py --model gin --hpo --hpo-trials 50
```

### 4. Compare All Models (Standard vs Enriched vs Ensemble)
```bash
python scripts/compare_all.py
python scripts/compare_all.py --quick   # 50 epochs for fast comparison
```

### 5. Sklearn-Compatible API
```python
from vegfr2 import GNNClassifier, GNNRegressor, EnsembleClassifier

# Train a GNN classifier (always uses enriched graphs)
clf = GNNClassifier(name="gin", hidden=128, layers=3, epochs=200)
clf.fit(train_smiles, train_labels, val_smiles, val_labels)
preds = clf.predict(test_smiles)
clf.save("model.pt")

# Train ensemble (GNN + XGBoost)
ens = EnsembleClassifier(gnn_name="gin", ml_name="xgb")
ens.fit(train_smiles, train_labels, val_smiles, val_labels)
preds = ens.predict(test_smiles)
```

### 6. Preprocess Data Pipeline
```bash
python scripts/preprocess_data.py --input data/raw/chembl_vegfr2.csv
```

### 7. Virtual Screening
```bash
python scripts/screen.py --model runs/gin/best.pt --input library.csv --output hits.csv
```

---

## Test Suite (CPU-Runnable)

```bash
pytest -v
```

**100 tests** covering all modules:

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_data.py` | 7 | CSV loading, preprocessing, deduplication, splitting |
| `test_device.py` | 1 | CUDA enforcement |
| `test_features.py` | 33 | Morgan, MACCS, graphs, enriched graphs, combinations |
| `test_gnn_models.py` | 7 | Pure PyTorch GCN, GAT, MPNN forward/backward/checkpoint |
| `test_gnn_pyg.py` | 15 | PyG models forward/backward, enriched, GATv2 vs GAT |
| `test_metrics.py` | 3 | ACC, SEN, SPE, MCC, AUC |
| `test_ml_models.py` | 8 | RF, SVM, XGBoost train/predict/save/load |
| `test_new_models.py` | 27 | GIN, PNA, GraphTransformer, sklearn API |

---

## Why Enriched Graphs Work

GNN alone with small data (~10K molecules) fails (AUC ~0.5-0.65). Enriched graphs inject domain knowledge (fingerprints) into the graph structure, giving GNNs access to established chemical features during message passing. This consistently achieves AUC ~0.85-0.92.

| Mode | Typical AUC | Why |
|------|------------|-----|
| Pure GNN (32-dim) | 0.50-0.65 | Too little data to learn useful representations |
| Enriched GNN (2246-dim) | 0.85-0.92 | Fingerprint knowledge guides message passing |
| GNN + ML Ensemble | 0.90-0.95 | Best of both worlds: GNN embeddings + ML on fingerprints |
