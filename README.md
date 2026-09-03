# VEGFR2 Virtual Screening Pipeline — ML + GNN

A compact, **reproducible** notebook workflow to build, validate, and deploy ML/GNN models that prioritize candidate **VEGFR2 inhibitors**.

> Reproduction of: Shengzhen Hou et al. (2025). *"Identification of potent inhibitors of potential VEGFR2: a graph neural network-based virtual screening and in vitro study."* **Journal of Enzyme Inhibition and Medicinal Chemistry**, 40:1. DOI: 10.1080/14756366.2025.2518192

---

## What's Inside

Each part includes: a Jupyter notebook, **exactly pinned** `requirements.txt`, and the needed `data/` and `images/`.

```
Part_1/   Data Acquisition              — Download VEGFR2 IC50 from ChEMBL
Part_2/   Preprocessing & Features       — SMILES cleanup, fingerprints, enriched graphs
Part_3/   Classical ML Baselines         — RF, SVM, XGBoost on Morgan/MACCS
Part_4/   GNN Models                     — GCN, GAT, GATv2, MPNN, GIN with enriched graphs
Part_5/   Advanced GNN                   — PNA, GraphTransformer, AttentiveFP, Fused variants
Part_6/   Ensemble & Comparison          — GNN + ML ensemble, full model ranking
Part_7/   Hyperparameter Optimization    — Optuna-based HPO for GNNs
Part_8/   Virtual Screening              — Screen external compound libraries
```

---

## Key Innovation: Enriched Graphs

Every GNN model receives **enriched node features** — fingerprints injected into every atom node:
```
[atom_features(32) + Morgan(2048) + MACCS(166)] = 2246-dim per node
```
This gives the GNN access to fingerprint knowledge during message passing, combining the strengths of both representations.

| Mode | Typical AUC | Why |
|------|------------|-----|
| Pure GNN (32-dim) | 0.50–0.65 | Too little data to learn useful representations |
| Enriched GNN (2246-dim) | 0.85–0.92 | Fingerprint knowledge guides message passing |
| GNN + ML Ensemble | 0.90–0.95 | Best of both worlds: GNN embeddings + ML on fingerprints |

---

## Models

### GNN Architectures (7 models)

| Model | Architecture |
|-------|-------------|
| **GCN** | Graph Convolutional Network with LayerNorm |
| **GAT** | Multi-head additive attention |
| **GATv2** | Dynamic attention (strictly more expressive than GAT) |
| **MPNN** | Edge-MLP + GRU update |
| **GIN** | Most expressive MPNN, MLP aggregation, JK connections |
| **PNA** | 4 aggregators + 3 scalers + residual connections |
| **Graph Transformer** | Global self-attention + edge bias + FFN |

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

## Quick Start

### Option A: Run Notebooks (Recommended)
```bash
pip install -r Part_1/requirements.txt
jupyter notebook Part_1/Data_Acquisition.ipynb
```

### Option B: Install as Package
```bash
pip install -e ".[all]"
```

### Option C: Run Scripts
```bash
python scripts/download_data.py
python scripts/train_all.py
```

---

## Reproducibility

- **Exact pins:** Each part has its own `requirements.txt` with `==` versions.
- **Determinism:** Fixed seeds (`seed=42`) for splits/estimators where supported.

---

## Project Structure

```
├── Part_1/                    # Data Acquisition
│   ├── Data_Acquisition.ipynb
│   ├── requirements.txt
│   └── README.md
├── Part_2/                    # Preprocessing & Feature Engineering
│   ├── Preprocessing_Feature_Engineering.ipynb
│   ├── requirements.txt
│   └── README.md
├── Part_3/                    # Classical ML Baselines
│   ├── Classical_ML_Baselines.ipynb
│   ├── requirements.txt
│   └── README.md
├── Part_4/                    # GNN Models
│   ├── GNN_Models.ipynb
│   ├── requirements.txt
│   └── README.md
├── Part_5/                    # Advanced GNN
│   ├── Advanced_GNN_Models.ipynb
│   ├── requirements.txt
│   └── README.md
├── Part_6/                    # Ensemble & Comparison
│   ├── Ensemble_and_Comparison.ipynb
│   ├── requirements.txt
│   └── README.md
├── Part_7/                    # Hyperparameter Optimization
│   ├── Hyperparameter_Optimization.ipynb
│   ├── requirements.txt
│   └── README.md
├── Part_8/                    # Virtual Screening
│   ├── Virtual_Screening.ipynb
│   ├── requirements.txt
│   └── README.md
├── src/vegfr2/                # Core library
├── scripts/                   # CLI training scripts
├── tests/                     # Test suite (100+ tests)
├── configs/                   # YAML configuration
└── data/                      # Processed data splits
```

---

## Test Suite

```bash
pytest -v
```

100+ tests covering all modules: data loading, features, ML models, GNN models, metrics.

---

## About

This repository contains datasets, data collection methods, preprocessing scripts, and machine learning models for discovering potential VEGFR2 inhibitors. It includes analysis pipelines and computational approaches to identify bioactive compounds that target VEGFR2, a key regulator in angiogenesis and cancer pathways.
