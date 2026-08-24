"""GNN + ML Ensemble - Combine GNN embeddings with traditional ML.

Uses a trained GNN to extract embeddings, then feeds them to XGBoost/RF
for final prediction. This combines:
- GNN: learns molecular graph patterns
- XGBoost/RF: excels at tabular feature combination
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from vegfr2.features import mol_to_graph_with_fps, smiles_to_morgan, smiles_to_maccs, combine_features
from vegfr2.ml_models import train_ml_model, predict_ml_model


class GNNEmbeddingExtractor(nn.Module):
    """Extract embeddings from any trained GNN model."""

    def __init__(self, gnn_model: nn.Module):
        super().__init__()
        self.gnn = gnn_model

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Forward pass that returns pooled embeddings (before output layer)."""
        if hasattr(self.gnn, "node_emb"):
            x = F.relu(self.gnn.node_emb(x))

        # GIN uses gin_convs + bn
        if hasattr(self.gnn, "gin_convs"):
            for i, (conv, bn) in enumerate(zip(self.gnn.gin_convs, self.gnn.bn)):
                x = conv(x, edge_index)
                x = bn(x)
                if i < len(self.gnn.gin_convs) - 1:
                    x = F.relu(x)
        # GCN/GAT/GATv2/GATv2 use convs + norms
        elif hasattr(self.gnn, "convs"):
            for i, conv in enumerate(self.gnn.convs):
                x = conv(x, edge_index)
                if hasattr(self.gnn, "norms") and i < len(self.gnn.norms):
                    x = self.gnn.norms[i](x)
                if i < len(self.gnn.convs) - 1:
                    x = F.relu(x)

        return global_mean_pool(x, batch)


class GNNEnsembleClassifier:
    """Ensemble of GNN + ML model.

    Always uses enriched graphs (Morgan + MACCS + atom features).
    Extracts GNN embeddings, combines with fingerprints, feeds to XGBoost/RF.

    Args:
        gnn_name: GNN model name ("gin", "pna", "graph_transformer", "gcn", "gat", "gatv2")
        ml_name: ML model name ("xgb", "rf", "svm")
        hidden: GNN hidden dimension
        layers: GNN layers
        heads: Attention heads (for GAT/GATv2/Transformer)
        dropout: Dropout rate
        strategy: "concat" (feature concat) or "stack" (stacking)
    """

    def __init__(
        self,
        gnn_name: str = "gin",
        ml_name: str = "xgb",
        hidden: int = 128,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.3,
        strategy: str = "concat",
        seed: int = 42,
    ):
        self.gnn_name = gnn_name
        self.ml_name = ml_name
        self.hidden = hidden
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.strategy = strategy
        self.seed = seed

        self.gnn_model = None
        self.ml_model = None
        self._fitted = False

    def _build_gnn(self, in_dim: int, edge_dim: int = 11) -> nn.Module:
        """Build GNN model by name."""
        from vegfr2.gnn_pyg import build_pyg_model
        return build_pyg_model(
            self.gnn_name,
            in_dim=in_dim,
            hidden=self.hidden,
            layers=self.layers,
            heads=self.heads,
            edge_dim=edge_dim,
            dropout=self.dropout,
        )

    def _extract_gnn_embeddings(
        self,
        model: nn.Module,
        smiles_list: list[str],
        device: torch.device,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Extract GNN embeddings for a list of SMILES."""
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader

        model.eval()
        embeddings = []

        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i : i + batch_size]
            data_list = []
            for s in batch_smiles:
                g = mol_to_graph_with_fps(s, use_morgan=True, use_maccs=True)
                data = Data(x=g["node_feats"], edge_index=g["edge_index"], edge_attr=g["edge_feats"])
                data_list.append(data)

            loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)

            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    x = F.relu(model.node_emb(batch.x))

                    if hasattr(model, "gin_convs"):
                        for j, conv in enumerate(model.gin_convs):
                            x = conv(x, batch.edge_index)
                            x = model.bn[j](x)
                            if j < len(model.gin_convs) - 1:
                                x = F.relu(x)
                    elif hasattr(model, "convs"):
                        for j, conv in enumerate(model.convs):
                            x = conv(x, batch.edge_index)
                            if hasattr(model, "norms") and j < len(model.norms):
                                x = model.norms[j](x)
                            if j < len(model.convs) - 1:
                                x = F.relu(x)

                    pooled = global_mean_pool(x, batch.batch)
                    embeddings.append(pooled.cpu().numpy())

        return np.vstack(embeddings)

    def _extract_fingerprints(self, smiles_list: list[str]) -> np.ndarray:
        """Extract Morgan + MACCS fingerprints (always both)."""
        morgan = np.vstack([smiles_to_morgan(s) for s in smiles_list])
        maccs = np.vstack([smiles_to_maccs(s) for s in smiles_list])
        return np.hstack([morgan, maccs])

    def fit(
        self,
        train_smiles: list[str],
        train_labels: list[int],
        val_smiles: list[str] | None = None,
        val_labels: list[int] | None = None,
        device: str | torch.device = "cuda",
        gnn_epochs: int = 100,
        gnn_patience: int = 15,
        gnn_lr: float = 0.001,
        batch_size: int = 128,
    ) -> "GNNEnsembleClassifier":
        """Fit the ensemble.

        1. Train GNN on training data
        2. Extract GNN embeddings
        3. Combine with fingerprints
        4. Train ML model on combined features
        """
        device = torch.device(device)
        torch.manual_seed(self.seed)

        # Determine input dimension (always enriched: 32 + 2048 + 166)
        in_dim = 2246

        # Step 1: Train GNN
        self.gnn_model = self._build_gnn(in_dim=in_dim).to(device)
        opt = torch.optim.AdamW(self.gnn_model.parameters(), lr=gnn_lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()

        # Prepare PyG datasets
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader

        train_data = []
        for s, y in zip(train_smiles, train_labels):
            g = mol_to_graph_with_fps(s, use_morgan=True, use_maccs=True)
            train_data.append(Data(x=g["node_feats"], edge_index=g["edge_index"], edge_attr=g["edge_feats"], y=torch.tensor([y], dtype=torch.float32)))
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_smiles and val_labels:
            val_data = []
            for s, y in zip(val_smiles, val_labels):
                g = mol_to_graph_with_fps(s, use_morgan=True, use_maccs=True)
                val_data.append(Data(x=g["node_feats"], edge_index=g["edge_index"], edge_attr=g["edge_feats"], y=torch.tensor([y], dtype=torch.float32)))
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        # Training loop
        best_auc = -1.0
        best_state = None
        wait = 0

        for epoch in range(1, gnn_epochs + 1):
            self.gnn_model.train()
            for batch in train_loader:
                batch = batch.to(device)
                logits = self.gnn_model(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(logits.squeeze(), batch.y)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if val_loader is not None:
                self.gnn_model.eval()
                val_probs, val_true = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        logits = self.gnn_model(batch.x, batch.edge_index, batch.batch)
                        val_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
                        val_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

                from vegfr2.metrics import classification_metrics
                val_auc = classification_metrics(val_true, val_probs).get("auc") or 0.0

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in self.gnn_model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= gnn_patience:
                        break

        if best_state is not None:
            self.gnn_model.load_state_dict(best_state)
        self.gnn_model.to(device)

        # Step 2: Extract embeddings
        gnn_train_emb = self._extract_gnn_embeddings(self.gnn_model, train_smiles, device)
        fp_train = self._extract_fingerprints(train_smiles)

        # Step 3: Combine features
        X_train = combine_features(gnn_train_emb, fp_train) if fp_train.shape[1] > 0 else gnn_train_emb

        # Step 4: Train ML model
        self.ml_model = train_ml_model(self.ml_name, X_train, np.array(train_labels), seed=self.seed)
        self._fitted = True

        return self

    def predict_proba(self, smiles_list: list[str], device: str | torch.device = "cuda") -> np.ndarray:
        """Predict probabilities for SMILES."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        device = torch.device(device)

        gnn_emb = self._extract_gnn_embeddings(self.gnn_model, smiles_list, device)
        fp = self._extract_fingerprints(smiles_list)

        X = combine_features(gnn_emb, fp) if fp.shape[1] > 0 else gnn_emb
        return predict_ml_model(self.ml_model, X)

    def predict(self, smiles_list: list[str], device: str | torch.device = "cuda", threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        probs = self.predict_proba(smiles_list, device)
        return (probs >= threshold).astype(int)

    def save(self, path: str) -> None:
        """Save ensemble to disk."""
        import pickle
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "gnn_state": self.gnn_model.state_dict() if self.gnn_model else None,
            "gnn_init_kwargs": self.gnn_model.init_kwargs if self.gnn_model else None,
            "gnn_name": self.gnn_name,
            "ml_model": self.ml_model,
            "config": {
                "ml_name": self.ml_name,
                "hidden": self.hidden,
                "layers": self.layers,
                "heads": self.heads,
                "dropout": self.dropout,
                "strategy": self.strategy,
                "seed": self.seed,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cpu") -> "GNNEnsembleClassifier":
        """Load ensemble from disk."""
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        config = state["config"]
        ensemble = cls(
            gnn_name=state["gnn_name"],
            ml_name=config["ml_name"],
            hidden=config["hidden"],
            layers=config["layers"],
            heads=config["heads"],
            dropout=config["dropout"],
            strategy=config["strategy"],
            seed=config["seed"],
        )

        if state["gnn_state"] is not None:
            ensemble.gnn_model = ensemble._build_gnn(in_dim=2246).to(device)
            ensemble.gnn_model.load_state_dict(state["gnn_state"])

        ensemble.ml_model = state["ml_model"]
        ensemble._fitted = True

        return ensemble
