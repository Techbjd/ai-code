# AttentiveFP VEGFR2 Pipeline (Colab)

Complete Colab-ready pipeline for VEGFR2 virtual screening using AttentiveFP.

## Quick Start

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `AttentiveFP_VEGFR2_Pipeline.py` or paste contents
3. Runtime → Change runtime type → T4 GPU
4. Runtime → Run all

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Install dependencies |
| 2 | Clone repository |
| 3 | Check GPU |
| 4 | Download VEGFR2 data from ChEMBL279 |
| 5 | Preprocess (standardize, deduplicate, label) |
| 6 | Scaffold split (train/val/test) |
| 7 | Build enriched graphs (2246-dim) |
| 8 | Load AttentiveFP model |
| 9 | Train with early stopping |
| 10 | Final test evaluation |
| 11-12 | Visualization (training curves, ROC/PR) |
| 13 | Compare with baselines (GCN, Morgan+XGBoost) |
| 14 | Save model |
| 15-17 | Screen Chinese medicine compounds (TCMSP) |
| 18 | Export hits |
| 19 | Run unit tests |

## Output Files

- `checkpoints/attentivefp_vegfr2.pt` — Trained model
- `tcm_screening_results.csv` — All TCM predictions
- `tcm_hits.csv` — Predicted active TCM compounds
- `attentivefp_training.png` — Training curves
- `attentivefp_roc_pr.png` — ROC and PR curves
- `tcm_screening_results.png` — Screening visualization

## Model Details

- **Architecture**: AttentiveFP (OpenDrugAI reference)
- **Input**: Enriched graphs [atom(32) + Morgan(2048) + MACCS(166)] = 2246-dim
- **Split**: Scaffold-based (generalization test)
- **Metrics**: ROC-AUC, PR-AUC, MCC, F1, Sensitivity, Specificity
