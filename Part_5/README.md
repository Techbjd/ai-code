# Part 5: Advanced GNN Models

## PNA, GraphTransformer, AttentiveFP, and Fused Variants

Trains advanced GNN architectures and explores the **fused** approach where fingerprints are combined at the graph level (not per-atom):

- **PNA** - Principal Neighbourhood Aggregation (4 aggregators + 3 scalers + residual connections)
- **GraphTransformer** - Global self-attention + edge bias
- **AttentiveFP** - Graph attention + GRU + attentive readout
- **FusedGNN** - Lightweight graph (32-dim) + fingerprint branch

### Key Comparison
| Approach | Description |
|----------|-------------|
| Enriched | Fingerprint features concatenated at atom level (2246-dim) |
| Fused | Separate graph (32-dim) + fingerprint (2214-dim) branches merged at graph level |

### Usage
```bash
pip install -r requirements.txt
jupyter notebook Advanced_GNN_Models.ipynb
```

### Requirements
- Training data: `data/train.csv`, `data/val.csv`, `data/test.csv`
- Source: `src/vegfr2/` (features, GNN models, fused models, metrics)
- GPU recommended for faster training
