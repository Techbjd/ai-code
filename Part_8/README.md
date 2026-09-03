# Part 8: Virtual Screening

## Screening External Compound Libraries

Uses the best trained GNN model (GIN with enriched graphs) to screen large compound libraries for potential VEGFR2 inhibitors.

## Workflow
1. Load trained model from Part_4
2. Define screening function for batch prediction
3. Screen compounds with probability-based ranking
4. Export predicted hits for downstream analysis

## Libraries Screened
- **COCONUT** - COllective UNknown INteractions in Natural Product database
- **ZINC** - ZINC is Not Commercial (general-purpose library)

## Features
- Batch processing with configurable thresholds
- Probability distribution analysis
- Hit rate calculation and export

## Requirements
```bash
pip install -r requirements.txt
```

## Usage
```bash
jupyter notebook Virtual_Screening.ipynb
```

## Output
- Screening predictions in `data/screening_hits.csv`
- Probability distribution plots saved to `images/screening_results.png`
