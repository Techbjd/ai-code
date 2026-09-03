# Part 2: Preprocessing & Feature Engineering

Transforms raw ChEMBL SMILES data into multiple feature representations for downstream model training.

## What This Notebook Does

1. **Loads raw ChEMBL data** — CSV with SMILES and IC50 values
2. **Validates & deduplicates** — drops invalid SMILES and conflicting IC50 entries
3. **Labels activity** — IC50 < 500 nM = active, else inactive
4. **Splits data** — stratified 8:1:1 (train/val/test)
5. **Extracts features**:
   - Morgan fingerprints (2048-bit)
   - MACCS keys (166-bit)
   - Molecular graphs (32-dim atom features)
   - Enriched graphs (2246-dim = atom + Morgan + MACCS)
6. **Visualizes** feature representations

## Prerequisites

- `data/chembl_vegfr2.csv` with columns: `smiles`, `ic50_nM`
- Install dependencies: `pip install -r requirements.txt`

## Output

| File | Description |
|------|-------------|
| `data/train.csv` | Training split |
| `data/val.csv` | Validation split |
| `data/test.csv` | Test split |
| `images/feature_visualization.png` | Feature heatmaps |

## Usage

```bash
cd Part_2
jupyter notebook Preprocessing_Feature_Engineering.ipynb
```
