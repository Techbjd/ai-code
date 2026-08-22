# VEGFR2 Virtual Screening Pipeline (Hou et al. reproduction)

This repository reproduces the machine learning and graph neural network (GNN) virtual screening pipeline for **VEGFR2 inhibitors** described in:
> Shengzhen Hou et al. (2025). *"Identification of potent inhibitors of potential VEGFR2: a graph neural network-based virtual screening and in vitro study."* **Journal of Enzyme Inhibition and Medicinal Chemistry**, 40:1. DOI: 10.1080/14756366.2025.2518192

## Features Implemented
- **Pure PyTorch GNNs:** From-scratch, DGL-free implementations of **GCN**, **GAT**, and **MPNN** matching the paper's equations (1)-(4).
- **Classical ML Models:** **RandomForest**, **SVM**, and **XGBoost** trained on 2048-bit Morgan Fingerprints.
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
  ├── features.py      # Morgan fp & Pure-PyTorch Graph builder
  ├── metrics.py       # ACC, SEN, SPE, MCC, AUC
  ├── ml_models.py     # RF, SVM, XGBoost
  ├── gnn_models.py    # GCN, GAT, MPNN nn.Modules
  └── hpo.py           # Optuna HPO interface (lazy imported)
scripts/
  ├── download_data.py # Fetch ChEMBL target CSV
  ├── train.py         # Main train & eval script
  └── screen.py        # Virtual screening script
tests/                 # 31 Unit tests (runnable on CPU)
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
To run Optuna-based Hyperparameter Optimization for GNNs, append `--hpo`.

### C. Screen a Library (GPU Required)
Rank a library of SMILES (e.g. TCM database TargetMol) to identify potential hits:
```bash
python scripts/screen.py --model runs/gcn/best.pt --input library.csv --output hits.csv --threshold 0.9
```

---

## 4. Test Suite (CPU-Runnable)
Unit tests run on CPU with synthetic structures to verify math correctness and preprocessing integrity without requiring a GPU or network access:
```bash
pytest -v
```
All 31 unit tests pass successfully.
