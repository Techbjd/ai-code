"""GNN models using DGL (Deep Graph Library) - matching paper's architecture.

Paper: Hou et al. (2025) used DGL library for molecular graph construction
and GNN training. This module implements GCN, GAT, and MPNN using DGL.

Reference: "Identification of potent inhibitors of potential VEGFR2:
a graph neural network-based virtual screening and in vitro study"
Journal of Enzyme Inhibition and Medicinal Chemistry, 40:1, 2518192
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl.data import DGLDataset


# ---------------------------------------------------------------------------
# DGL Dataset
# ---------------------------------------------------------------------------

class DGLMolDataset(DGLDataset):
    """DGL dataset for molecular graphs with 32-dim atom features."""

    def __init__(self, smiles_list: list[str], labels: list[int]):
        self.smiles_list = smiles_list
        self.labels = labels
        self.graphs = []
        self._build_graphs()
        super().__init__(name="vegfr2_mols")

    def _smiles_to_dgl(self, smiles: str):
        """Convert SMILES to DGL graph with 32-dim atom features."""
        from rdkit import Chem
        from vegfr2.features import _atom_features, _bond_features, ATOM_FEAT_DIM, BOND_FEAT_DIM

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Atom features (32-dim)
        atom_feats = []
        for atom in mol.GetAtoms():
            atom_feats.append(_atom_features(atom))

        # Bond features and edge list
        src_nodes = []
        dst_nodes = []
        bond_feats = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = _bond_features(bond)
            src_nodes.extend([i, j])
            dst_nodes.extend([j, i])
            bond_feats.extend([bf, bf])

        if not src_nodes:
            return None

        g = dgl.graph((src_nodes, dst_nodes))
        g.ndata["feat"] = torch.tensor(atom_feats, dtype=torch.float32)
        g.edata["feat"] = torch.tensor(bond_feats, dtype=torch.float32)

        return g

    def _build_graphs(self):
        for s, y in zip(self.smiles_list, self.labels):
            g = self._smiles_to_dgl(s)
            if g is not None:
                g.ndata["label"] = torch.tensor([y] * g.num_nodes(), dtype=torch.float32)
                self.graphs.append(g)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


def collate_dgl(batch):
    """Collate function for DGL DataLoader."""
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return batched_graph, labels


# ---------------------------------------------------------------------------
# GCN Model (Paper: Equation 1)
# H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
# ---------------------------------------------------------------------------

class GCN_DGL(nn.Module):
    """Graph Convolutional Network using DGL.

    Matches paper's GCN architecture with LayerNorm and dropout.
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden: int = 64,
        layers: int = 3,
        out_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.init_kwargs = {
            "in_dim": in_dim,
            "hidden": hidden,
            "layers": layers,
            "out_dim": out_dim,
            "dropout": dropout,
        }
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.convs.append(dglnn.GraphConv(in_dim, hidden, allow_zero_in_degree=True))
        for _ in range(layers - 1):
            self.convs.append(dglnn.GraphConv(hidden, hidden, allow_zero_in_degree=True))
        for _ in range(layers):
            self.norms.append(nn.LayerNorm(hidden))
            self.dropouts.append(nn.Dropout(dropout))

        self.output = nn.Linear(hidden, out_dim)

    def forward(self, g, x):
        for i, (conv, norm, drop) in enumerate(
            zip(self.convs, self.norms, self.dropouts)
        ):
            x = conv(g, x)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = drop(x)
        # Global mean pooling
        g.ndata["h"] = x
        hg = dgl.mean_nodes(g, "h")
        return self.output(hg)


# ---------------------------------------------------------------------------
# GAT Model (Paper: attention mechanism with LeakyReLU + softmax)
# ---------------------------------------------------------------------------

class GAT_DGL(nn.Module):
    """Graph Attention Network using DGL.

    Matches paper's GAT with multi-head attention.
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden: int = 64,
        layers: int = 3,
        heads: int = 4,
        out_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.init_kwargs = {
            "in_dim": in_dim,
            "hidden": hidden,
            "layers": layers,
            "heads": heads,
            "out_dim": out_dim,
            "dropout": dropout,
        }
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.convs.append(dglnn.GATConv(in_dim, hidden // heads, heads=heads, allow_zero_in_degree=True))
        for _ in range(layers - 1):
            self.convs.append(dglnn.GATConv(hidden, hidden // heads, heads=heads, allow_zero_in_degree=True))
        for _ in range(layers):
            self.norms.append(nn.LayerNorm(hidden))
            self.dropouts.append(nn.Dropout(dropout))

        self.output = nn.Linear(hidden, out_dim)

    def forward(self, g, x):
        for i, (conv, norm, drop) in enumerate(
            zip(self.convs, self.norms, self.dropouts)
        ):
            x = conv(g, x)
            x = x.flatten(1)  # concatenate heads
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = drop(x)
        g.ndata["h"] = x
        hg = dgl.mean_nodes(g, "h")
        return self.output(hg)


# ---------------------------------------------------------------------------
# MPNN Model (Paper: Message, Aggregate, Update)
# m_v(t) = Σ M(h_u(t-1), h_v(t-1), e_uv)
# a_v(t) = AGG({m_u(t) : u ∈ N(v)})
# h_v(t) = U(h_v(t-1), a_v(t))
# ---------------------------------------------------------------------------

class MPNN_DGL(nn.Module):
    """Message Passing Neural Network using DGL.

    Matches paper's MPNN with edge features.
    Uses DGL's built-in MPNN layer.
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden: int = 64,
        layers: int = 3,
        out_dim: int = 1,
        edge_dim: int = 11,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.init_kwargs = {
            "in_dim": in_dim,
            "hidden": hidden,
            "layers": layers,
            "out_dim": out_dim,
            "edge_dim": edge_dim,
            "dropout": dropout,
        }
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.convs.append(dglnn.MPNNConv(in_dim, hidden, edge_dim=edge_dim, aggregator_type="sum"))
        for _ in range(layers - 1):
            self.convs.append(dglnn.MPNNConv(hidden, hidden, edge_dim=edge_dim, aggregator_type="sum"))
        for _ in range(layers):
            self.norms.append(nn.LayerNorm(hidden))
            self.dropouts.append(nn.Dropout(dropout))

        self.output = nn.Linear(hidden, out_dim)

    def forward(self, g, x):
        for i, (conv, norm, drop) in enumerate(
            zip(self.convs, self.norms, self.dropouts)
        ):
            x = conv(g, x)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = drop(x)
        g.ndata["h"] = x
        hg = dgl.mean_nodes(g, "h")
        return self.output(hg)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dgl_model(
    name: str,
    in_dim: int = 32,
    hidden: int = 64,
    layers: int = 3,
    heads: int = 4,
    out_dim: int = 1,
    edge_dim: int = 11,
    dropout: float = 0.3,
) -> nn.Module:
    """Build a DGL GNN model by name.

    Args:
        name: Model name ("gcn", "gat", "mpnn")
        in_dim: Input node feature dimension
        hidden: Hidden dimension
        layers: Number of GNN layers
        heads: Attention heads (for GAT)
        out_dim: Output dimension
        edge_dim: Edge feature dimension (for MPNN)
        dropout: Dropout rate

    Returns:
        DGL GNN model
    """
    name = name.lower()
    if name == "gcn":
        return GCN_DGL(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim, dropout=dropout)
    elif name == "gat":
        return GAT_DGL(in_dim=in_dim, hidden=hidden, layers=layers, heads=heads, out_dim=out_dim, dropout=dropout)
    elif name == "mpnn":
        return MPNN_DGL(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim, edge_dim=edge_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown DGL model: {name}. Available: gcn, gat, mpnn")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_dgl_model(
    name: str,
    train_smiles: list[str],
    train_labels: list[int],
    val_smiles: list[str] | None = None,
    val_labels: list[int] | None = None,
    hidden: int = 64,
    layers: int = 3,
    heads: int = 4,
    lr: float = 0.001,
    batch_size: int = 128,
    epochs: int = 200,
    patience: int = 15,
    seed: int = 42,
    device: str | torch.device = "cuda",
) -> nn.Module:
    """Train a DGL GNN model.

    Args:
        name: Model name ("gcn", "gat", "mpnn")
        train_smiles: Training SMILES strings
        train_labels: Training binary labels (0/1)
        val_smiles: Validation SMILES (optional)
        val_labels: Validation labels
        hidden: Hidden dimension
        layers: Number of GNN layers
        heads: Attention heads
        lr: Learning rate
        batch_size: Batch size
        epochs: Maximum epochs
        patience: Early stopping patience
        seed: Random seed
        device: Target device

    Returns:
        Trained model
    """
    torch.manual_seed(seed)
    device = torch.device(device)

    # Create datasets
    train_ds = DGLMolDataset(train_smiles, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_dgl
    )

    val_loader = None
    if val_smiles is not None:
        val_ds = DGLMolDataset(val_smiles, val_labels)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_dgl
        )

    # Build model
    model = build_dgl_model(name, in_dim=32, hidden=hidden, layers=layers, heads=heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_graph, batch_labels in train_loader:
            batch_graph = batch_graph.to(device)
            batch_labels = batch_labels.to(device)
            x = batch_graph.ndata["feat"]
            logits = model(batch_graph, x)
            loss = loss_fn(logits.squeeze(), batch_labels.squeeze())
            opt.zero_grad()
            loss.backward()
            opt.step()

        if val_loader is not None:
            model.eval()
            val_probs = []
            val_true = []
            with torch.no_grad():
                for batch_graph, batch_labels in val_loader:
                    batch_graph = batch_graph.to(device)
                    x = batch_graph.ndata["feat"]
                    logits = model(batch_graph, x)
                    val_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
                    val_true.extend(batch_labels.squeeze().numpy().astype(int))

            from vegfr2.metrics import classification_metrics
            metrics = classification_metrics(val_true, val_probs)
            val_auc = metrics.get("auc") or 0.0

            if epoch % 25 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} val_AUC={val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"  Early stop at epoch {epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_dgl_model(
    model: nn.Module,
    smiles_list: list[str],
    batch_size: int = 256,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """Predict probabilities using a trained DGL model.

    Args:
        model: Trained model
        smiles_list: SMILES strings to predict
        batch_size: Inference batch size
        device: Target device

    Returns:
        Array of probabilities
    """
    device = torch.device(device)
    model.eval()

    ds = DGLMolDataset(smiles_list, [0] * len(smiles_list))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_dgl
    )

    probs = []
    with torch.no_grad():
        for batch_graph, _ in loader:
            batch_graph = batch_graph.to(device)
            x = batch_graph.ndata["feat"]
            logits = model(batch_graph, x)
            probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())

    return np.array(probs)
