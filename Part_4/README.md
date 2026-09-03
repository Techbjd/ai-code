# Part 4: Graph Neural Network Models

## GCN, GAT, GATv2, MPNN, GIN with Enriched Graphs

Trains 5 GNN architectures using **enriched graphs** where each atom node receives: `[atom_features(32) + Morgan(2048) + MACCS(166)] = 2246-dim`.

This is the key innovation: injecting fingerprint knowledge into the graph structure gives GNNs access to established chemical features during message passing.

### Models Trained
- **GCN** - Graph Convolutional Network
- **GAT** - Graph Attention Network
- **GATv2** - Dynamic Attention (GATv2)
- **MPNN** - Message Passing Neural Network
- **GIN** - Graph Isomorphism Network

### Usage
```bash
pip install -r requirements.txt
jupyter notebook GNN_Models.ipynb
```

### Requirements
- Training data: `data/train.csv`, `data/val.csv`, `data/test.csv`
- Source: `src/vegfr2/` (features, GNN models, metrics)
- GPU recommended for faster training
