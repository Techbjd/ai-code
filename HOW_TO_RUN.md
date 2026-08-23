# How to Run the VEGFR2 Virtual Screening Pipeline

Follow these step-by-step instructions to set up, download data, train, and run virtual screening.

---

## Step 1: Install Dependencies
Create a clean virtual environment and install the required libraries:
```bash
# 1. Create and activate environment (example using Conda)
conda create -n vegfr2 python=3.10 -y
conda activate vegfr2

# 2. Install dependencies (RDKit, scikit-learn, PyTorch with CUDA)
# Ensure your PyTorch installation supports CUDA (GPU)
pip install -r requirements.txt
```

---

## Step 2: Download the Data
The dataset needs to be downloaded from the ChEMBL database (target ID `CHEMBL279` for VEGFR2):
```bash
# Downloads raw data and saves it under data/raw/
python scripts/download_data.py --out data/raw/chembl_vegfr2.csv
```

### Fallback (Offline Mode)
If you have no internet access or the ChEMBL API is down, you can pass a local fallback CSV (which must have columns `smiles` and `ic50_nM`):
```bash
python scripts/download_data.py --out data/raw/chembl_vegfr2.csv --fallback /path/to/your/fallback.csv
```

---

## Step 3: Train the Models (GPU Required)
Since the paper requires strict GPU-only training, running training on a CPU-only machine will raise a clear `RuntimeError`. Move to a GPU-enabled machine and run:

```bash
# Train ALL 6 models (RF, SVM, XGBoost, GCN, GAT, MPNN)
python scripts/train.py --config configs/config.yaml --model all

# Train ONLY a specific GNN model (e.g. Graph Convolutional Network)
python scripts/train.py --config configs/config.yaml --model gcn

# Train ONLY a specific classical ML model (e.g. XGBoost)
python scripts/train.py --config configs/config.yaml --model xgb
```

### Hyperparameter Optimization (Optuna)
To run Optuna-based Hyperparameter Optimization for GNNs before standard training, append the `--hpo` flag:
```bash
python scripts/train.py --config configs/config.yaml --model gcn --hpo
```
*Note: This requires `optuna` to be installed (`pip install optuna`).*

### Outputs
Once training finishes:
- Trained model checkpoints will be saved as `runs/<model_name>/best.pt` (GNNs) or `runs/<model_name>/model.pkl` (ML).
- A full summary of performance metrics on the test set will be printed as a table and saved to `runs/results.json`.

---

## Step 4: Screen an External SMILES Library (GPU Required)
Once you have a trained model, you can screen custom libraries (e.g. TCM monomers or TargetMol collections) to identify potential hits:

1. Create/provide an input CSV file (e.g., `library.csv`) with at least a `smiles` column (an optional `name` column will be preserved in the output if present):
   ```csv
   smiles,name
   C(=O)(C(C)O)O,Lactate
   Cc1ccccc1,Toluene
   ```

2. Run the virtual screening CLI with your best checkpoint (e.g., GCN):
   ```bash
   python scripts/screen.py \
     --model runs/gcn/best.pt \
     --input library.csv \
     --output hits.csv \
     --threshold 0.9
   ```

### Screening Outputs
- The output file `hits.csv` will contain predictions for all inputs sorted descending by probability of activity.
- Compounds with predicted probabilities $\ge 0.9$ (or your custom `--threshold`) will have `hit` set to `True`.

---

## Verification: Run Unit Tests (No GPU Required)
You can run the full test suite on **any** machine (including CPU-only machines) to verify features, data preprocessing, and model math sanity:
```bash
pytest -v
```
All 31 test cases will execute and should output a full list of passing tests.
