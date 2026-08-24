"""Sklearn-compatible API for GNN models.

Use GNN models exactly like sklearn classifiers:

    from vegfr2.sklearn_api import GNNClassifier

    model = GNNClassifier(model="gin", hidden=128, layers=3)
    model.fit(train_smiles, train_labels)
    probs = model.predict_proba(test_smiles)
    labels = model.predict(test_smiles)

    model.save("model.pkl")
    model = GNNClassifier.load("model.pkl")

    # Works with sklearn tools
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, all_smiles, all_labels, cv=5, scoring="roc_auc")
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from vegfr2.features import mol_to_graph, smiles_to_morgan, smiles_to_maccs, combine_features
from vegfr2.metrics import classification_metrics


AVAILABLE_MODELS = ["gin", "pna", "graph_transformer", "gcn", "gat", "gatv2", "mpnn"]
AVAILABLE_ML = ["xgb", "rf", "svm"]


class GNNClassifier:
    """Sklearn-compatible GNN classifier.

    Wraps PyTorch Geometric GNN models with a fit/predict API
    that works with sklearn tools (cross_val_score, GridSearchCV, etc.).

    ALL models use enriched graphs by default:
    [atom(32) + Morgan(2048) + MACCS(166)] = 2246-dim input per node.
    This gives the GNN fingerprint knowledge during message passing.

    Args:
        model: GNN architecture name
        hidden: Hidden dimension
        layers: Number of GNN layers
        heads: Attention heads (for GAT/GATv2/Transformer)
        dropout: Dropout rate
        lr: Learning rate
        batch_size: Batch size
        epochs: Max training epochs
        patience: Early stopping patience
        seed: Random seed
        device: Device ("cuda" or "cpu")
    """

    def __init__(
        self,
        model: str = "gin",
        hidden: int = 128,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.3,
        lr: float = 0.001,
        batch_size: int = 128,
        epochs: int = 200,
        patience: int = 15,
        seed: int = 42,
        device: str | torch.device = "cuda",
    ):
        model = model.lower()
        if model not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model}. Available: {AVAILABLE_MODELS}")

        self.model_name = model
        self.hidden = hidden
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.device = device

        self.gnn_model_ = None
        self._fitted = False

    def _get_input_dim(self) -> int:
        return 2246  # 32 atom + 2048 morgan + 166 maccs

    def _build_model(self, in_dim: int) -> nn.Module:
        from vegfr2.gnn_pyg import build_pyg_model
        return build_pyg_model(
            self.model_name,
            in_dim=in_dim,
            hidden=self.hidden,
            layers=self.layers,
            heads=self.heads,
            edge_dim=11,
            dropout=self.dropout,
        )

    def _smiles_to_data(self, smiles_list: list[str], labels: list[int] | None = None) -> list[Data]:
        from vegfr2.features import mol_to_graph_with_fps

        data_list = []
        for i, s in enumerate(smiles_list):
            g = mol_to_graph_with_fps(s, use_morgan=True, use_maccs=True)
            y = labels[i] if labels is not None else 0
            data = Data(
                x=g["node_feats"],
                edge_index=g["edge_index"],
                edge_attr=g["edge_feats"],
                y=torch.tensor([y], dtype=torch.float32),
            )
            data_list.append(data)
        return data_list

    def fit(
        self,
        X: list[str] | np.ndarray,
        y: list[int] | np.ndarray,
        val_X: list[str] | np.ndarray | None = None,
        val_y: list[int] | np.ndarray | None = None,
    ) -> "GNNClassifier":
        """Fit the GNN classifier.

        Args:
            X: List of SMILES strings
            y: List of binary labels (0/1)
            val_X: Optional validation SMILES
            val_y: Optional validation labels

        Returns:
            self
        """
        X = list(X)
        y = list(y)

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)

        # Build model
        in_dim = self._get_input_dim()
        self.gnn_model_ = self._build_model(in_dim).to(device)

        # Prepare data
        train_data = self._smiles_to_data(X, y)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if val_X is not None and val_y is not None:
            val_data = self._smiles_to_data(list(val_X), list(val_y))
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        # Training
        opt = torch.optim.AdamW(self.gnn_model_.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-6)
        loss_fn = nn.BCEWithLogitsLoss()

        best_auc = -1.0
        best_state = None
        wait = 0

        for epoch in range(1, self.epochs + 1):
            self.gnn_model_.train()
            for batch in train_loader:
                batch = batch.to(device)
                logits = self.gnn_model_(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(logits.squeeze(), batch.y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            scheduler.step()

            if val_loader is not None:
                self.gnn_model_.eval()
                val_probs, val_true = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        logits = self.gnn_model_(batch.x, batch.edge_index, batch.batch)
                        val_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
                        val_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

                val_auc = classification_metrics(val_true, val_probs).get("auc") or 0.0

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in self.gnn_model_.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        if best_state is not None:
            self.gnn_model_.load_state_dict(best_state)
        self.gnn_model_.to(device).eval()
        self._fitted = True

        return self

    def predict_proba(self, X: list[str] | np.ndarray) -> np.ndarray:
        """Predict probabilities.

        Args:
            X: List of SMILES strings

        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        X = list(X)

        data_list = self._smiles_to_data(X)
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)

        self.gnn_model_.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = self.gnn_model_(batch.x, batch.edge_index, batch.batch)
                batch_probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                if batch_probs.ndim == 0:
                    batch_probs = batch_probs.reshape(1)
                probs.extend(batch_probs)

        probs = np.array(probs)
        return np.column_stack([1 - probs, probs])

    def predict(self, X: list[str] | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels.

        Args:
            X: List of SMILES strings
            threshold: Classification threshold

        Returns:
            Array of 0/1 predictions
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)

    def score(self, X: list[str] | np.ndarray, y: list[int] | np.ndarray) -> float:
        """Compute accuracy (for sklearn compatibility)."""
        preds = self.predict(X)
        return float((preds == np.array(y)).mean())

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "gnn_state": self.gnn_model_.state_dict() if self.gnn_model_ else None,
            "gnn_init_kwargs": self.gnn_model_.init_kwargs if self.gnn_model_ else None,
            "config": {
                "model_name": self.model_name,
                "hidden": self.hidden,
                "layers": self.layers,
                "heads": self.heads,
                "dropout": self.dropout,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "patience": self.patience,
                "seed": self.seed,
                "device": str(self.device),
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cpu") -> "GNNClassifier":
        """Load model from disk.

        Args:
            path: Path to saved model
            device: Device to load model on

        Returns:
            Loaded GNNClassifier
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        config = state["config"]
        classifier = cls(
            model=config["model_name"],
            hidden=config["hidden"],
            layers=config["layers"],
            heads=config["heads"],
            dropout=config["dropout"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            patience=config["patience"],
            seed=config["seed"],
            device=device,
        )

        if state["gnn_state"] is not None:
            classifier.gnn_model_ = classifier._build_model(classifier._get_input_dim())
            classifier.gnn_model_.load_state_dict(state["gnn_state"])
            classifier.gnn_model_.to(device).eval()

        classifier._fitted = True
        return classifier

    def __repr__(self) -> str:
        return (
            f"GNNClassifier(model={self.model_name!r}, hidden={self.hidden}, "
            f"layers={self.layers}, heads={self.heads}, dropout={self.dropout})"
        )


class GNNRegressor:
    """Sklearn-compatible GNN regressor for continuous targets.

    Same API as GNNClassifier but outputs continuous values.

    Args:
        model: GNN architecture name
        hidden: Hidden dimension
        layers: Number of GNN layers
        heads: Attention heads
        dropout: Dropout rate
        lr: Learning rate
        batch_size: Batch size
        epochs: Max training epochs
        patience: Early stopping patience
        seed: Random seed
        device: Device ("cuda" or "cpu")
    """

    def __init__(
        self,
        model: str = "gin",
        hidden: int = 128,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.3,
        lr: float = 0.001,
        batch_size: int = 128,
        epochs: int = 200,
        patience: int = 15,
        seed: int = 42,
        device: str | torch.device = "cuda",
    ):
        self.model_name = model
        self.hidden = hidden
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.device = device

        self.gnn_model_ = None
        self._fitted = False

    def _build_model(self) -> nn.Module:
        from vegfr2.gnn_pyg import build_pyg_model
        return build_pyg_model(
            self.model_name,
            in_dim=2246,
            hidden=self.hidden,
            layers=self.layers,
            heads=self.heads,
            edge_dim=11,
            dropout=self.dropout,
            out_dim=1,
        )

    def _smiles_to_data(self, smiles_list: list[str], labels: list[float] | None = None) -> list[Data]:
        from vegfr2.features import mol_to_graph_with_fps

        data_list = []
        for i, s in enumerate(smiles_list):
            g = mol_to_graph_with_fps(s, use_morgan=True, use_maccs=True)
            y = labels[i] if labels is not None else 0.0
            data = Data(
                x=g["node_feats"],
                edge_index=g["edge_index"],
                edge_attr=g["edge_feats"],
                y=torch.tensor([y], dtype=torch.float32),
            )
            data_list.append(data)
        return data_list

    def fit(
        self,
        X: list[str] | np.ndarray,
        y: list[float] | np.ndarray,
        val_X: list[str] | np.ndarray | None = None,
        val_y: list[float] | np.ndarray | None = None,
    ) -> "GNNRegressor":
        """Fit the GNN regressor."""
        X = list(X)
        y = [float(v) for v in y]

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)

        self.gnn_model_ = self._build_model().to(device)

        train_data = self._smiles_to_data(X, y)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if val_X is not None and val_y is not None:
            val_data = self._smiles_to_data(list(val_X), [float(v) for v in val_y])
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        opt = torch.optim.AdamW(self.gnn_model_.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(1, self.epochs + 1):
            self.gnn_model_.train()
            for batch in train_loader:
                batch = batch.to(device)
                preds = self.gnn_model_(batch.x, batch.edge_index, batch.batch).squeeze()
                loss = loss_fn(preds, batch.y)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if val_loader is not None:
                self.gnn_model_.eval()
                val_preds, val_true = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        preds = self.gnn_model_(batch.x, batch.edge_index, batch.batch).squeeze()
                        val_preds.extend(preds.cpu().numpy())
                        val_true.extend(batch.y.cpu().numpy())

                val_loss = float(loss_fn(torch.tensor(val_preds), torch.tensor(val_true)))
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.gnn_model_.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        if best_state is not None:
            self.gnn_model_.load_state_dict(best_state)
        self.gnn_model_.to(device).eval()
        self._fitted = True

        return self

    def predict(self, X: list[str] | np.ndarray) -> np.ndarray:
        """Predict continuous values."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        data_list = self._smiles_to_data(list(X))
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)

        self.gnn_model_.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = self.gnn_model_(batch.x, batch.edge_index, batch.batch)
                preds.extend(out.squeeze().cpu().numpy())

        return np.array(preds)

    def score(self, X: list[str] | np.ndarray, y: list[float] | np.ndarray) -> float:
        """Compute R^2 score (for sklearn compatibility)."""
        from sklearn.metrics import r2_score
        preds = self.predict(X)
        return r2_score(y, preds)

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "gnn_state": self.gnn_model_.state_dict() if self.gnn_model_ else None,
            "gnn_init_kwargs": self.gnn_model_.init_kwargs if self.gnn_model_ else None,
            "config": {
                "model_name": self.model_name,
                "hidden": self.hidden,
                "layers": self.layers,
                "heads": self.heads,
                "dropout": self.dropout,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "patience": self.patience,
                "seed": self.seed,
                "device": str(self.device),
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cpu") -> "GNNRegressor":
        """Load model from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        config = state["config"]
        regressor = cls(
            model=config["model_name"],
            hidden=config["hidden"],
            layers=config["layers"],
            heads=config["heads"],
            dropout=config["dropout"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            patience=config["patience"],
            seed=config["seed"],
            device=device,
        )

        if state["gnn_state"] is not None:
            regressor.gnn_model_ = regressor._build_model()
            regressor.gnn_model_.load_state_dict(state["gnn_state"])
            regressor.gnn_model_.to(device).eval()

        regressor._fitted = True
        return regressor

    def __repr__(self) -> str:
        return (
            f"GNNRegressor(model={self.model_name!r}, hidden={self.hidden}, "
            f"layers={self.layers}, heads={self.heads}, dropout={self.dropout})"
        )


class EnsembleClassifier:
    """Sklearn-compatible GNN + ML ensemble classifier.

    Combines GNN embeddings with traditional ML (XGBoost/RF).
    Always uses enriched graphs (Morgan + MACCS + atom features).

    Args:
        gnn: GNN model name
        ml: ML model name
        hidden: GNN hidden dimension
        layers: GNN layers
        heads: Attention heads
        dropout: Dropout rate
        seed: Random seed
        device: Device
    """

    def __init__(
        self,
        gnn: str = "gin",
        ml: str = "xgb",
        hidden: int = 128,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.3,
        seed: int = 42,
        device: str | torch.device = "cuda",
    ):
        from vegfr2.models.ensemble import GNNEnsembleClassifier
        self._ensemble = GNNEnsembleClassifier(
            gnn_name=gnn,
            ml_name=ml,
            hidden=hidden,
            layers=layers,
            heads=heads,
            dropout=dropout,
            seed=seed,
        )
        self.device = device
        self._fitted = False

    def fit(
        self,
        X: list[str] | np.ndarray,
        y: list[int] | np.ndarray,
        val_X: list[str] | np.ndarray | None = None,
        val_y: list[int] | np.ndarray | None = None,
    ) -> "EnsembleClassifier":
        """Fit the ensemble."""
        self._ensemble.fit(
            train_smiles=list(X),
            train_labels=[int(v) for v in y],
            val_smiles=list(val_X) if val_X is not None else None,
            val_labels=[int(v) for v in val_y] if val_y is not None else None,
            device=self.device,
        )
        self._fitted = True
        return self

    def predict_proba(self, X: list[str] | np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        probs = self._ensemble.predict_proba(list(X), device=self.device)
        return np.column_stack([1 - probs, probs])

    def predict(self, X: list[str] | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)

    def score(self, X: list[str] | np.ndarray, y: list[int] | np.ndarray) -> float:
        """Compute accuracy."""
        preds = self.predict(X)
        return float((preds == np.array(y)).mean())

    def save(self, path: str) -> None:
        """Save model."""
        self._ensemble.save(path)

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cpu") -> "EnsembleClassifier":
        """Load model."""
        from vegfr2.models.ensemble import GNNEnsembleClassifier
        ens = GNNEnsembleClassifier.load(path, device)
        wrapper = cls.__new__(cls)
        wrapper._ensemble = ens
        wrapper.device = device
        wrapper._fitted = True
        return wrapper

    def __repr__(self) -> str:
        return f"EnsembleClassifier(gnn={self._ensemble.gnn_name!r}, ml={self._ensemble.ml_name!r})"
