# Part 6: Ensemble Models & Full Comparison

## GNN + ML Ensemble and Complete Model Ranking

Trains **GNNEnsembleClassifier** models that combine GNN embeddings with molecular fingerprints (Morgan/MACCS) using XGBoost/RF, then produces a complete ranking of ALL models across the pipeline.

### Ensemble Configurations
- **GIN + XGBoost** - GIN embeddings → XGBoost classifier
- **PNA + XGBoost** - PNA embeddings → XGBoost classifier
- **GIN + RF** - GIN embeddings → Random Forest classifier
- **PNA + RF** - PNA embeddings → Random Forest classifier

### Pipeline Coverage
- Classical ML baselines (RF, SVM, XGBoost)
- GNN models (GCN, GAT, GATv2, MPNN, GIN)
- Advanced GNNs (PNA, Graph Transformer, AttentiveFP)
- Ensemble models (GNN + ML hybrid)

### Usage
```bash
pip install -r requirements.txt
jupyter notebook Ensemble_and_Comparison.ipynb
```

### Requirements
- Training data: `data/train.csv`, `data/val.csv`, `data/test.csv`
- Source: `src/vegfr2/` (ensemble, models, features, metrics)
- Results from Part_3 and Part_4 recommended

### Output
- Trained ensembles saved to `models/` directory
- Final comparison: `data/final_results.json`
- Visualization: `images/final_comparison.png`
