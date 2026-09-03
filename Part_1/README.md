# Part 1: Data Acquisition

## Downloading VEGFR2 IC50 Data from ChEMBL

This notebook downloads bioactivity data for VEGFR2 (CHEMBL279) from the ChEMBL database.

**Target:** VEGFR2 (VEGF Receptor 2)  
**Data Source:** ChEMBL279  
**Measurement:** IC50 (nM)

## Setup

```bash
pip install -r requirements.txt
```

## Output

- `data/chembl_vegfr2.csv` - Raw IC50 data with SMILES and activity values
- `images/ic50_distribution.png` - Distribution plots of IC50 values
