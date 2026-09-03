# Part 3: Classical ML Baselines

## Random Forest, SVM, XGBoost on Molecular Fingerprints

Establishes baseline performance using traditional ML models on:
- **Morgan Fingerprints** (2048-bit)
- **MACCS Keys** (166-bit)
- **Morgan + MACCS** (2214-bit combined)

## Models
- **Random Forest** (RF)
- **Support Vector Machine** (SVM)
- **XGBoost** (XGB)

## Features
- Training and evaluation on Morgan fingerprints, MACCS keys, and combined features
- Evaluation metrics: Accuracy, Sensitivity, Specificity, MCC, AUC
- ROC curve visualization
- Confusion matrix analysis
- Feature importance analysis for Random Forest
- Model serialization and results export

## Requirements
```bash
pip install -r requirements.txt
```

## Usage
```bash
jupyter notebook Classical_ML_Baselines.ipynb
```

## Output
- Trained models saved to `models/` directory
- Results exported to `data/ml_results.csv`
- ROC curves saved to `images/roc_curves_ml.png`
- Feature importance plots saved to `images/feature_importance.png`