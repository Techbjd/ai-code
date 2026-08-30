"""Self-supervised pre-training for VEGFR2 molecular GNNs.

Provides two pre-training approaches:
1. Contrastive learning (SimCLR-style)
2. Masked atom prediction (BERT-style)

Both approaches learn general molecular representations from unlabeled SMILES,
which can then be fine-tuned on the VEGFR2 activity prediction task.

Usage:
    from vegfr2.pretrain import SelfSupervisedPretrainer

    # Contrastive pre-training
    pretrainer = SelfSupervisedPretrainer(
        model_name="gin",
        method="contrastive",
        hidden=128,
        layers=3,
    )
    pretrainer.pretrain(train_smiles, epochs=100)
    pretrainer.save_pretrained("checkpoints/pretrained_gin.pt")

    # Fine-tune on VEGFR2 task
    from vegfr2.sklearn_api import GNNClassifier
    model = GNNClassifier(model="gin", hidden=128, layers=3)
    model.load_pretrained("checkpoints/pretrained_gin.pt")
    model.fit(train_smiles, train_labels)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from vegfr2.features import mol_to_graph_with_fps
from vegfr2.pretrain_models import (
    ContrastiveGNN,
    MaskedAtomGNN,
    GraphAugmentor,
    PretrainDataset,
)


class SelfSupervisedPretrainer:
    """Unified API for self-supervised pre-training of molecular GNNs.

    Args:
        model_name: GNN architecture name (gin, pna, graph_transformer, etc.)
        method: Pre-training method ("contrastive" or "masked")
        hidden: Hidden dimension
        layers: Number of GNN layers
        heads: Attention heads
        dropout: Dropout rate
        lr: Learning rate
        batch_size: Batch size
        temperature: Temperature for contrastive loss (contrastive only)
        mask_rate: Mask rate for atom prediction (masked only)
        projection_dim: Projection head output dim (contrastive only)
        seed: Random seed
    """

    def __init__(
        self,
        model_name: str = "gin",
        method: str = "contrastive",
        hidden: int = 128,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.3,
        lr: float = 0.001,
        batch_size: int = 128,
        temperature: float = 0.07,
        mask_rate: float = 0.15,
        projection_dim: int = 64,
        seed: int = 42,
    ):
        if method not in ("contrastive", "masked"):
            raise ValueError(f"Unknown method: {method}. Use 'contrastive' or 'masked'.")

        self.model_name = model_name
        self.method = method
        self.hidden = hidden
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask_rate = mask_rate
        self.projection_dim = projection_dim
        self.seed = seed

        # Will be initialized in _build_model
        self.model: nn.Module | None = None
        self._pretrained = False
        self.node_dim = 2246  # 32 atom + 2048 morgan + 166 maccs

    def _build_base_model(self) -> nn.Module:
        """Build the base GNN model with hidden-dim output (not 1)."""
        from vegfr2.gnn_pyg import build_pyg_model
        return build_pyg_model(
            self.model_name,
            in_dim=self.node_dim,
            hidden=self.hidden,
            layers=self.layers,
            heads=self.heads,
            out_dim=self.hidden,  # Output hidden dims for pre-training
            edge_dim=11,
            dropout=self.dropout,
        )

    def _build_pretrain_model(self) -> nn.Module:
        """Build the pre-training model with appropriate head."""
        base_gnn = self._build_base_model()

        if self.method == "contrastive":
            model = ContrastiveGNN(
                gnn=base_gnn,
                hidden_dim=self.hidden,
                projection_dim=self.projection_dim,
                temperature=self.temperature,
            )
        else:  # masked
            model = MaskedAtomGNN(
                gnn=base_gnn,
                hidden_dim=self.hidden,
                atom_feat_dim=self.node_dim,
                mask_rate=self.mask_rate,
            )

        return model

    def _get_gnn_embeddings(self, model: nn.Module, batch) -> torch.Tensor:
        """Extract GNN embeddings from a batch (before pre-training head)."""
        if hasattr(model, "gnn"):
            # ContrastiveGNN or MaskedAtomGNN wrapper
            base_gnn = model.gnn
            if hasattr(base_gnn, "gin_convs"):
                # GIN
                x = torch.relu(base_gnn.node_emb(batch.x))
                for i, (conv, bn) in enumerate(zip(base_gnn.gin_convs, base_gnn.bn)):
                    x = conv(x, batch.edge_index)
                    x = bn(x)
                    if i < len(base_gnn.gin_convs) - 1:
                        x = torch.relu(x)
                return global_mean_pool(x, batch.batch)
            elif hasattr(base_gnn, "convs"):
                # GCN/GAT/GATv2/PNA
                x = batch.x
                for i, conv in enumerate(base_gnn.convs):
                    x = conv(x, batch.edge_index)
                    if hasattr(base_gnn, "norms") and i < len(base_gnn.norms):
                        x = base_gnn.norms[i](x)
                    if i < len(base_gnn.convs) - 1:
                        x = torch.relu(x)
                return global_mean_pool(x, batch.batch)
        # Fallback: try direct forward
        return model(batch.x, batch.edge_index, batch.batch)

    def pretrain(
        self,
        smiles_list: list[str],
        epochs: int = 100,
        val_smiles: list[str] | None = None,
        patience: int = 20,
        device: str | torch.device = "cuda",
        verbose: bool = True,
    ) -> dict:
        """Run self-supervised pre-training.

        Args:
            smiles_list: List of SMILES strings (labels are ignored)
            epochs: Maximum pre-training epochs
            val_smiles: Optional validation SMILES for early stopping
            patience: Early stopping patience
            device: Target device
            verbose: Print progress

        Returns:
            Dict with pre-training history (losses, epochs, etc.)
        """
        torch.manual_seed(self.seed)
        device = torch.device(device)

        # Build model
        self.model = self._build_pretrain_model().to(device)
        n_params = sum(p.numel() for p in self.model.parameters())
        if verbose:
            print(f"  Model: {self.model_name} ({n_params:,} params)")
            print(f"  Method: {self.method}")
            print(f"  Input: {self.node_dim}-dim enriched graphs")

        # Prepare data
        train_dataset = PretrainDataset(smiles_list)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if val_smiles:
            val_dataset = PretrainDataset(val_smiles)
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        # Optimizer
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=1e-6
        )

        # Pre-training loop
        history = {"train_loss": [], "val_loss": []}
        best_loss = float("inf")
        best_state = None
        wait = 0

        if self.method == "contrastive":
            augmentor = GraphAugmentor(seed=self.seed)

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = batch.to(device)

                if self.method == "contrastive":
                    # Create two augmented views
                    aug1 = augmentor(batch)
                    aug2 = augmentor(batch)

                    z1 = self.model.encode(aug1.x, aug1.edge_index, batch.batch)
                    z2 = self.model.encode(aug2.x, aug2.edge_index, batch.batch)
                    loss = self.model.contrastive_loss(z1, z2)

                else:  # masked
                    loss, mask = self.model.masked_prediction_loss(
                        batch.x, batch.x,  # original features
                        batch.edge_index, batch.batch,
                    )

                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = total_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        if self.method == "contrastive":
                            aug1 = augmentor(batch)
                            aug2 = augmentor(batch)
                            z1 = self.model.encode(aug1.x, aug1.edge_index, batch.batch)
                            z2 = self.model.encode(aug2.x, aug2.edge_index, batch.batch)
                            loss = self.model.contrastive_loss(z1, z2)
                        else:
                            loss, _ = self.model.masked_prediction_loss(
                                batch.x, batch.x,
                                batch.edge_index, batch.batch,
                            )
                        val_loss += loss.item()
                        val_batches += 1
                avg_val_loss = val_loss / max(val_batches, 1)
                history["val_loss"].append(avg_val_loss)

                # Early stopping
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch} (best loss={best_loss:.4f})")
                        break

            if verbose and epoch % 10 == 0:
                val_str = f" val_loss={history['val_loss'][-1]:.4f}" if history["val_loss"] else ""
                print(f"  Epoch {epoch:3d} | train_loss={avg_train_loss:.4f}{val_str}")

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.to(device)
        self._pretrained = True

        if verbose:
            print(f"  Pre-training complete. Best loss: {best_loss:.4f}")

        return history

    def get_embeddings(
        self,
        smiles_list: list[str],
        batch_size: int = 256,
        device: str | torch.device = "cpu",
    ) -> np.ndarray:
        """Get pre-trained GNN embeddings for molecules.

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size
            device: Target device

        Returns:
            Array of shape (n_molecules, hidden_dim)
        """
        if not self._pretrained:
            raise RuntimeError("Model not pre-trained. Call pretrain() first.")

        device = torch.device(device)
        self.model.eval()

        dataset = PretrainDataset(smiles_list)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        embeddings = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                h = self._get_gnn_embeddings(self.model, batch)
                embeddings.append(h.cpu().numpy())

        return np.vstack(embeddings)

    def save_pretrained(self, path: str | Path) -> None:
        """Save pre-trained model to disk.

        Args:
            path: Save path
        """
        if not self._pretrained:
            raise RuntimeError("Model not pre-trained. Call pretrain() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Extract the base GNN model (without pre-training head)
        if hasattr(self.model, "gnn"):
            gnn_state = self.model.gnn.state_dict()
            gnn_init_kwargs = getattr(self.model.gnn, "init_kwargs", {})
        else:
            gnn_state = self.model.state_dict()
            gnn_init_kwargs = getattr(self.model, "init_kwargs", {})

        state = {
            "gnn_state": gnn_state,
            "gnn_init_kwargs": gnn_init_kwargs,
            "pretrain_config": {
                "model_name": self.model_name,
                "method": self.method,
                "hidden": self.hidden,
                "layers": self.layers,
                "heads": self.heads,
                "dropout": self.dropout,
                "node_dim": self.node_dim,
            },
        }
        torch.save(state, path)

    @classmethod
    def load_pretrained(
        cls,
        path: str | Path,
        model_name: str | None = None,
        device: str | torch.device = "cpu",
    ) -> "SelfSupervisedPretrainer":
        """Load pre-trained model from disk.

        Args:
            path: Path to saved checkpoint
            model_name: Override model name (optional)
            device: Target device

        Returns:
            Loaded SelfSupervisedPretrainer
        """
        path = Path(path)
        state = torch.load(path, map_location=device, weights_only=False)

        config = state["pretrain_config"]
        pretrainer = cls(
            model_name=model_name or config["model_name"],
            method=config["method"],
            hidden=config["hidden"],
            layers=config["layers"],
            heads=config["heads"],
            dropout=config["dropout"],
        )

        # Rebuild and load GNN
        pretrainer.model = pretrainer._build_pretrain_model()
        if hasattr(pretrainer.model, "gnn"):
            pretrainer.model.gnn.load_state_dict(state["gnn_state"])
        else:
            pretrainer.model.load_state_dict(state["gnn_state"])

        pretrainer.model.to(device).eval()
        pretrainer._pretrained = True

        return pretrainer

    def __repr__(self) -> str:
        return (
            f"SelfSupervisedPretrainer(model={self.model_name!r}, "
            f"method={self.method!r}, hidden={self.hidden}, "
            f"layers={self.layers})"
        )
