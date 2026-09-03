# Part 7: Hyperparameter Optimization

## Optuna-based HPO for GNN Models

Uses Optuna to systematically search for optimal hyperparameters for GNN models:

- **Hidden dimension**: [64, 128, 256]
- **Number of layers**: [2, 3, 4, 5]
- **Learning rate**: [1e-5, 1e-2] (log scale)
- **Dropout**: [0.1, 0.5]
- **Attention heads**: [4, 8] (for GAT-based models)

## Models Tuned
- **GIN** - Graph Isomorphism Network
- **PNA** - Principal Neighbourhood Aggregation
- **GraphTransformer** - Global self-attention

## Features
- TPE (Tree-structured Parzen Estimator) sampling
- 30 trials per model with 50 training epochs each
- AUC as optimization objective
- Results visualization and comparison

## Requirements
```bash
pip install -r requirements.txt
```

## Usage
```bash
jupyter notebook Hyperparameter_Optimization.ipynb
```

## Output
- Best hyperparameters for each model
- HPO convergence plots saved to `images/hpo_results.png`
