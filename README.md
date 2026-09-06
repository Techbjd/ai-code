# VEGFR2 Virtual Screening Pipeline — ML + GNN

A compact, **reproducible** notebook workflow to build, validate, and deploy ML/GNN models that prioritize candidate **VEGFR2 inhibitors**.

> Reproduction of: Shengzhen Hou et al. (2025). *"Identification of potent inhibitors of potential VEGFR2: a graph neural network-based virtual screening and in vitro study."* **Journal of Enzyme Inhibition and Medicinal Chemistry**, 40:1. DOI: 10.1080/14756366.2025.2518192

---

## Paper's Method (Hou et al. 2025)

1. **Data**: ChEMBL279 VEGFR2, 5564 compounds (IC50 < 500 nM = active, 2841 active / 2723 inactive)
2. **Features**: Morgan fingerprints (ML) + molecular graphs via DGL (GNN)
3. **Models**: 6 models — RF, SVM, XGBoost, GCN, GAT, MPNN
4. **Best model**: GCN (AUC=0.8937, MCC=0.6554)
5. **Screening**: GCN → TargetMol (2910 monomers) → pre-score > 0.9 → 151 compounds
6. **Docking**: AutoDock Vina, 40x40x40 A grid, center (-22.465, 0.422, -11.481)
7. **MD**: 100 ns, AMBER, MM-PBSA
8. **Hits**: Cynaroside (IC50=2698 nM), Luteolin 7-O-glucuronide (IC50=5969 nM), Scutellarin (IC50=8349 nM)

### Paper's Model Comparison (Test Set)

| Model | ACC | SEN | SPE | MCC | AUC |
|-------|-----|-----|-----|-----|-----|
| **GCN** | 0.8276 | 0.8015 | 0.8526 | **0.6554** | **0.8937** |
| RF | 0.8115 | 0.8088 | 0.8140 | 0.6228 | 0.8720 |
| XGBoost | 0.8043 | 0.8051 | 0.8035 | 0.6085 | 0.8801 |
| GAT | 0.7899 | 0.7610 | 0.8175 | 0.5798 | 0.8662 |
| SVM | 0.7702 | 0.7647 | 0.7754 | 0.5401 | 0.8432 |
| MPNN | 0.7433 | 0.6912 | 0.7930 | 0.4872 | 0.8216 |

---

## What's Inside

```
Part_1/   Data Acquisition              — Download VEGFR2 IC50 from ChEMBL
Part_2/   Preprocessing & Features       — SMILES cleanup, Morgan fingerprints, plain graphs
Part_3/   Classical ML Baselines         — RF, SVM, XGBoost on Morgan fingerprints
Part_4/   GNN Models                     — GCN, GAT, GATv2, MPNN, GIN (32-dim plain graphs)
Part_5/   Advanced GNN                   — PNA, GraphTransformer, AttentiveFP
Part_6/   Ensemble & Comparison          — GNN + ML ensemble, full model ranking
Part_7/   Hyperparameter Optimization    — Optuna-based HPO for GNNs
Part_8/   Virtual Screening              — Screen external compound libraries
colab_screening.py                       — Paper-exact reproduction (Colab ready)
```

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
| **Random Forest** | 300 trees, Morgan fingerprints (2048-bit) |
| **SVM** | RBF kernel, C=10, Morgan fingerprints |
| **XGBoost** | 400 estimators, depth=6, Morgan fingerprints |

---

## Quick Start

### Option A: Run Notebooks
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

### Option D: Colab (Paper Exact)
```bash
# Upload colab_screening.py to Google Colab and run
# Trains 5 models (RF, SVM, XGB, GCN, GAT) on plain graphs
# Screens paper's 6 molecules for validation
```

---

## Reproducibility

- **Exact pins:** Each part has its own `requirements.txt` with `==` versions.
- **Determinism:** Fixed seeds (`seed=42`) for splits/estimators where supported.

---

## Project Structure

```
├── Part_1/ - Part_8/         # Jupyter notebooks
├── colab_screening.py        # Paper-exact reproduction (Colab)
├── src/vegfr2/               # Core library
├── scripts/                  # CLI training scripts
├── tests/                    # Test suite (100+ tests)
├── configs/                  # YAML configuration
└── data/                     # Processed data splits
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
