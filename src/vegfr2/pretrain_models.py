"""Self-supervised pre-training models and augmentations for molecular graphs.

Two approaches:
1. Contrastive Learning (SimCLR-style): learn by contrasting augmented views
2. Masked Atom Prediction (BERT-style): reconstruct masked atom features

Both wrap existing GNN models (GIN, PNA, etc.) and add pre-training heads.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from vegfr2.features import mol_to_graph_with_fps


# ============================================================
# Graph Augmentations
# ============================================================

class GraphAugmentor:
    """Apply random augmentations to molecular graphs for contrastive learning.

    Supported augmentations:
        - Atom feature masking: randomly zero out atom features
        - Edge dropping: randomly remove edges
        - Subgraph extraction: extract a random subgraph
        - Feature permutation: randomly permute atom features across nodes
    """

    def __init__(
        self,
        atom_mask_prob: float = 0.15,
        edge_drop_prob: float = 0.1,
        subgraph_ratio: float = 0.8,
        feature_permute_prob: float = 0.1,
        seed: int = 42,
    ):
        self.atom_mask_prob = atom_mask_prob
        self.edge_drop_prob = edge_drop_prob
        self.subgraph_ratio = subgraph_ratio
        self.feature_permute_prob = feature_permute_prob
        self.rng = random.Random(seed)

    def __call__(self, data: Data) -> Data:
        """Apply a random augmentation to a PyG Data object.

        Only uses augmentations that preserve node count (atom_mask, feature_permute)
        to keep batch vectors valid.
        """
        aug_type = self.rng.choice(["atom_mask", "feature_permute"])

        if aug_type == "atom_mask":
            return self._atom_mask(data)
        elif aug_type == "edge_drop":
            return self._edge_drop(data)
        elif aug_type == "subgraph":
            return self._subgraph(data)
        elif aug_type == "feature_permute":
            return self._feature_permute(data)
        return data

    def _atom_mask(self, data: Data) -> Data:
        """Randomly mask atom features (set to zero)."""
        data = data.clone()
        mask = torch.rand(data.x.shape[0]) < self.atom_mask_prob
        data.x[mask] = 0.0
        return data

    def _edge_drop(self, data: Data) -> Data:
        """Randomly drop edges."""
        data = data.clone()
        n_edges = data.edge_index.shape[1]
        keep = torch.rand(n_edges) >= self.edge_drop_prob
        if keep.sum() < 2:
            keep[:2] = True  # ensure at least one edge
        data.edge_index = data.edge_index[:, keep]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[keep]
        return data

    def _subgraph(self, data: Data) -> Data:
        """Extract a random subgraph by keeping a fraction of nodes."""
        data = data.clone()
        n_nodes = data.x.shape[0]
        keep_ratio = self.subgraph_ratio + self.rng.uniform(-0.2, 0.2)
        keep_ratio = max(0.3, min(1.0, keep_ratio))
        n_keep = max(2, int(n_nodes * keep_ratio))
        keep_idx = torch.tensor(
            self.rng.sample(range(n_nodes), n_keep), dtype=torch.long
        )
        keep_idx, _ = keep_idx.sort()

        # Map old node indices to new
        old_to_new = {old.item(): new for new, old in enumerate(keep_idx)}
        data.x = data.x[keep_idx]

        # Filter and remap edges
        mask_src = torch.isin(data.edge_index[0], keep_idx)
        mask_dst = torch.isin(data.edge_index[1], keep_idx)
        mask = mask_src & mask_dst
        data.edge_index = data.edge_index[:, mask]
        data.edge_index[0] = torch.tensor(
            [old_to_new[x.item()] for x in data.edge_index[0]]
        )
        data.edge_index[1] = torch.tensor(
            [old_to_new[x.item()] for x in data.edge_index[1]]
        )
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[mask]

        return data

    def _feature_permute(self, data: Data) -> Data:
        """Randomly permute atom features across nodes."""
        data = data.clone()
        n_nodes = data.x.shape[0]
        perm = torch.randperm(n_nodes)
        mask = torch.rand(n_nodes) < self.feature_permute_prob
        data.x[mask] = data.x[perm[mask]]
        return data


# ============================================================
# Contrastive Learning (SimCLR-style)
# ============================================================

class ContrastiveGNN(nn.Module):
    """GNN with projection head for contrastive pre-training.

    Wraps any existing GNN (GIN, PNA, etc.) and adds:
    - A projection MLP head for contrastive learning
    - NT-Xent loss computation

    Args:
        gnn: Base GNN model (must accept x, edge_index, batch)
        hidden_dim: GNN output dimension
        projection_dim: Projection head output dimension
        temperature: Temperature for NT-Xent loss
    """

    def __init__(
        self,
        gnn: nn.Module,
        hidden_dim: int = 128,
        projection_dim: int = 64,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.gnn = gnn
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Projection MLP (used only during pre-training)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Get GNN embeddings (without projection)."""
        return self.gnn(x, edge_index, batch)

    def encode(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Get projected embeddings for contrastive learning."""
        h = self.gnn(x, edge_index, batch)
        z = self.projector(h)
        return F.normalize(z, dim=1)

    def contrastive_loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        """NT-Xent contrastive loss for two augmented views.

        Args:
            z1: Projected embeddings from view 1, shape [B, projection_dim]
            z2: Projected embeddings from view 2, shape [B, projection_dim]

        Returns:
            Scalar loss
        """
        batch_size = z1.shape[0]
        device = z1.device

        # Concatenate both views: [2B, projection_dim]
        z = torch.cat([z1, z2], dim=0)
        n = z.shape[0]

        # Compute cosine similarity matrix
        z_norm = F.normalize(z, dim=1)
        sim = torch.mm(z_norm, z_norm.t()) / self.temperature  # [2B, 2B]

        # Mask out self-similarity (diagonal)
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        sim_masked = sim.masked_select(mask).view(n, n - 1)

        # For each sample i, its positive pair is at index (i + batch_size) % n
        # After removing diagonal, we need to adjust the positive index
        labels = torch.zeros(n, dtype=torch.long, device=device)
        for i in range(n):
            pos_idx = (i + batch_size) % n
            # After removing column i from row i, adjust index
            adjusted = pos_idx - 1 if pos_idx > i else pos_idx
            labels[i] = adjusted

        return F.cross_entropy(sim_masked, labels)


# ============================================================
# Masked Atom Prediction (BERT-style)
# ============================================================

class MaskedAtomGNN(nn.Module):
    """GNN with reconstruction head for masked atom prediction.

    Works at the NODE level (no pooling) - reconstructs each atom's features.

    Args:
        gnn: Base GNN model
        hidden_dim: GNN hidden dimension
        atom_feat_dim: Atom feature dimension (2246 for enriched)
        mask_rate: Fraction of atoms to mask
    """

    def __init__(
        self,
        gnn: nn.Module,
        hidden_dim: int = 128,
        atom_feat_dim: int = 2246,
        mask_rate: float = 0.15,
    ):
        super().__init__()
        self.gnn = gnn
        self.hidden_dim = hidden_dim
        self.mask_rate = mask_rate

        # Extract node embeddings BEFORE pooling
        self.node_encoder = None  # Will use gnn's layers directly

        # Decoder MLP to reconstruct atom features (node-level)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, atom_feat_dim),
        )

    def _get_node_embeddings(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Get node-level embeddings from GNN (before pooling)."""
        if hasattr(self.gnn, "node_emb"):
            # GIN, PNA, MPNN, GraphTransformer
            h = F.relu(self.gnn.node_emb(x))
        else:
            # GCN, GAT, GATv2 - first conv is in convs list
            h = x

        if hasattr(self.gnn, "gin_convs"):
            # GIN
            for i, (conv, bn) in enumerate(zip(self.gnn.gin_convs, self.gnn.bn)):
                h = conv(h, edge_index)
                h = bn(h)
                if i < len(self.gnn.gin_convs) - 1:
                    h = F.relu(h)
        elif hasattr(self.gnn, "convs"):
            # GCN, GAT, GATv2, PNA
            for i, conv in enumerate(self.gnn.convs):
                h = conv(h, edge_index)
                if hasattr(self.gnn, "norms") and i < len(self.gnn.norms):
                    h = self.gnn.norms[i](h)
                if i < len(self.gnn.convs) - 1:
                    h = F.relu(h)
        elif hasattr(self.gnn, "edge_nets"):
            # MPNN - uses edge_attr
            # For masked prediction, we skip edge features for simplicity
            h = F.relu(self.gnn.node_emb(x))

        return h  # [N_nodes, hidden_dim]

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Get graph-level embeddings (for downstream tasks)."""
        node_h = self._get_node_embeddings(x, edge_index)
        from torch_geometric.nn import global_mean_pool
        return global_mean_pool(node_h, batch)

    def create_mask(
        self, x: Tensor, mask_rate: float | None = None
    ) -> tuple[Tensor, Tensor]:
        """Create mask for atom features."""
        rate = mask_rate or self.mask_rate
        n_nodes = x.shape[0]
        mask = torch.rand(n_nodes, device=x.device) < rate
        masked_x = x.clone()
        masked_x[mask] = 0.0
        return masked_x, mask

    def masked_prediction_loss(
        self, x_orig: Tensor, x_masked: Tensor, edge_index: Tensor, batch: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute MSE loss on masked atom positions (node-level).

        Args:
            x_orig: Original atom features [N_nodes, feat_dim]
            x_masked: Masked atom features
            edge_index: Graph connectivity
            batch: Batch assignment vector

        Returns:
            loss: MSE loss on masked positions
            mask: Boolean mask used
        """
        # Create mask
        n_nodes = x_orig.shape[0]
        mask = torch.rand(n_nodes, device=x_orig.device) < self.mask_rate

        # Zero out masked positions
        x_input = x_orig.clone()
        x_input[mask] = 0.0

        # Get node-level embeddings (NO pooling)
        h = self._get_node_embeddings(x_input, edge_index)

        # Decode to original features (node-level)
        x_recon = self.decoder(h)

        # Compute loss only on masked positions
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x_orig.device, requires_grad=True), mask

        loss = F.mse_loss(x_recon[mask], x_orig[mask])
        return loss, mask


# ============================================================
# Pre-training Dataset
# ============================================================

class PretrainDataset(torch.utils.data.Dataset):
    """Dataset for self-supervised pre-training.

    Converts SMILES to PyG Data objects with enriched features.
    """

    def __init__(
        self,
        smiles_list: list[str],
        morgan_radius: int = 2,
        morgan_n_bits: int = 2048,
        maccs_n_bits: int = 166,
    ):
        self.data_list = []
        for s in smiles_list:
            try:
                g = mol_to_graph_with_fps(
                    s,
                    use_morgan=True,
                    use_maccs=True,
                    morgan_radius=morgan_radius,
                    morgan_n_bits=morgan_n_bits,
                    maccs_n_bits=maccs_n_bits,
                )
                data = Data(
                    x=g["node_feats"],
                    edge_index=g["edge_index"],
                    edge_attr=g["edge_feats"],
                )
                self.data_list.append(data)
            except Exception:
                continue  # skip invalid molecules

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]
