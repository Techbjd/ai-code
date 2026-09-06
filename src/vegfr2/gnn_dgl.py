"""GNN models matching paper's architecture using pure PyTorch.

Paper: Hou et al. (2025) used DGL for GCN, GAT, MPNN.
This implements the same architectures with pure PyTorch operations
(no DGL/PyG dependency) so it works on any Python version.

Reference: "Identification of potent inhibitors of potential VEGFR2:
a graph neural network-based virtual screening and in vitro study"
Journal of Enzyme Inhibition and Medicinal Chemistry, 40:1, 2518192
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Graph helper functions (pure PyTorch, no library dependency)
# ---------------------------------------------------------------------------

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum src into index positions, result has dim_size entries."""
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    return out.scatter_add(0, index.unsqueeze(1).expand_as(src), src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean of src grouped by index."""
    out_sum = scatter_add(src, index, dim_size)
    count = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index, torch.ones(src.size(0), device=src.device, dtype=src.dtype))
    count = count.clamp(min=1).unsqueeze(1)
    return out_sum / count


class BatchedGraph:
    """Simple batched graph container for plain PyTorch GNN."""

    def __init__(self, node_feats: torch.Tensor, edge_index: torch.Tensor,
                 batch_vec: torch.Tensor, num_nodes: int, num_edges: int):
        self.x = node_feats          # [total_nodes, feat_dim]
        self.edge_index = edge_index  # [2, total_edges]
        self.batch = batch_vec        # [total_nodes] — which graph each node belongs to
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_graphs = batch_vec.unique().numel()

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.batch = self.batch.to(device)
        return self


def mol_to_graph(smiles: str):
    """Convert SMILES to (node_feats, edge_index) using RDKit."""
    from rdkit import Chem
    from vegfr2.features import _atom_features, _bond_features

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_feats = [_atom_features(atom) for atom in mol.GetAtoms()]
    if not atom_feats:
        return None

    src_nodes, dst_nodes = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src_nodes.extend([i, j])
        dst_nodes.extend([j, i])

    if not src_nodes:
        return None

    x = torch.tensor(atom_feats, dtype=torch.float32)
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    return x, edge_index


def batch_graphs(graph_list: list[tuple[torch.Tensor, torch.Tensor]]) -> BatchedGraph:
    """Batch a list of (node_feats, edge_index) into one BatchedGraph."""
    x_list, ei_list, batch_list = [], [], []
    offset = 0
    for i, (x, ei) in enumerate(graph_list):
        x_list.append(x)
        ei_list.append(ei + offset)
        batch_list.append(torch.full((x.size(0),), i, dtype=torch.long))
        offset += x.size(0)

    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(ei_list, dim=1)
    batch = torch.cat(batch_list, dim=0)
    return BatchedGraph(x, edge_index, batch, x.size(0), edge_index.size(1))


def smiles_to_graph_batch(smiles_list: list[str]) -> BatchedGraph | None:
    """Convert list of SMILES to a batched graph. Skips invalid."""
    graphs = []
    for s in smiles_list:
        result = mol_to_graph(s)
        if result is not None:
            graphs.append(result)
    if not graphs:
        return None
    return batch_graphs(graphs)


# ---------------------------------------------------------------------------
# GCN Layer (Paper: H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l)))
# ---------------------------------------------------------------------------

class GCNConv(nn.Module):
    """Graph Convolutional Network layer (Kipf & Welling, 2017)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, g: BatchedGraph, x: torch.Tensor) -> torch.Tensor:
        src, dst = g.edge_index[0], g.edge_index[1]
        # Degree normalization: D^{-1/2} A D^{-1/2}
        deg = torch.zeros(g.num_nodes, device=x.device)
        deg.scatter_add_(0, src, torch.ones(src.size(0), device=x.device))
        deg_inv_sqrt = deg.clamp(min=1).pow(-0.5)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        # Aggregate normalized messages
        msg = x[src] * norm.unsqueeze(1)
        out = scatter_add(msg, dst, g.num_nodes)
        return self.W(out) + self.bias


class GCN_DGL(nn.Module):
    """Graph Convolutional Network matching paper's architecture."""

    def __init__(self, in_dim=32, hidden=64, layers=3, out_dim=1, dropout=0.3):
        super().__init__()
        self.init_kwargs = {"in_dim": in_dim, "hidden": hidden, "layers": layers,
                            "out_dim": out_dim, "dropout": dropout}
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        for _ in range(layers):
            self.norms.append(nn.LayerNorm(hidden))
            self.dropouts.append(nn.Dropout(dropout))
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, g: BatchedGraph, x: torch.Tensor) -> torch.Tensor:
        for i, (conv, norm, drop) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            x = conv(g, x)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = drop(x)
        # Global mean pooling
        hg = scatter_mean(x, g.batch, g.num_graphs)
        return self.output(hg)


# ---------------------------------------------------------------------------
# GAT Layer (Paper: attention with LeakyReLU + softmax)
# ---------------------------------------------------------------------------

class GATConv(nn.Module):
    """Graph Attention Network layer (Veličković et al., 2018)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.heads = heads
        self.head_dim = out_dim // heads
        assert out_dim == heads * self.head_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky = nn.LeakyReLU(0.2)
        self.attn_drop = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, g: BatchedGraph, x: torch.Tensor) -> torch.Tensor:
        H, D = self.heads, self.head_dim
        Wh = self.W(x).view(-1, H, D)  # [N, H, D]
        src, dst = g.edge_index[0], g.edge_index[1]

        e_src = (Wh * self.a_src).sum(dim=-1)  # [N, H]
        e_dst = (Wh * self.a_dst).sum(dim=-1)
        e = self.leaky(e_src[src] + e_dst[dst])  # [E, H]

        # Softmax per destination node
        max_e = scatter_add(F.relu(e), dst, g.num_nodes)  # approximate max
        e_max = max_e[dst]
        e = e - e_max  # numerical stability
        exp_e = torch.exp(e)
        sum_exp = scatter_add(exp_e, dst, g.num_nodes)
        attn = exp_e / (sum_exp[dst] + 1e-8)
        attn = self.attn_drop(attn)

        # Weighted aggregation
        msg = Wh[src] * attn.unsqueeze(2)  # [E, H, D]
        msg = msg.view(-1, H * D)
        out = scatter_add(msg, dst, g.num_nodes)
        return out + self.bias


class GAT_DGL(nn.Module):
    """Graph Attention Network matching paper's architecture."""

    def __init__(self, in_dim=32, hidden=64, layers=3, heads=4, out_dim=1, dropout=0.3):
        super().__init__()
        self.init_kwargs = {"in_dim": in_dim, "hidden": hidden, "layers": layers,
                            "heads": heads, "out_dim": out_dim, "dropout": dropout}
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden, heads=heads, dropout=dropout))
        for _ in range(layers - 1):
            self.convs.append(GATConv(hidden, hidden, heads=heads, dropout=dropout))
        for _ in range(layers):
            self.norms.append(nn.LayerNorm(hidden))
            self.dropouts.append(nn.Dropout(dropout))
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, g: BatchedGraph, x: torch.Tensor) -> torch.Tensor:
        for i, (conv, norm, drop) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            x = conv(g, x)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = drop(x)
        hg = scatter_mean(x, g.batch, g.num_graphs)
        return self.output(hg)


# ---------------------------------------------------------------------------
# MPNN Layer (Paper: Message, Aggregate, Update)
# m_v(t) = Σ M(h_u(t-1), h_v(t-1), e_uv)
# a_v(t) = AGG({m_u(t)})
# h_v(t) = U(h_v(t-1), a_v(t))
# ---------------------------------------------------------------------------

class MPNNConv(nn.Module):
    """Message Passing Neural Network layer (Gilmer et al., 2017)."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 11):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.msg_fn = nn.Sequential(
            nn.Linear(out_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.update_fn = nn.GRUCell(out_dim, out_dim)

    def forward(self, g: BatchedGraph, x: torch.Tensor, edge_feats: torch.Tensor) -> torch.Tensor:
        src, dst = g.edge_index[0], g.edge_index[1]
        x_proj = self.in_proj(x)
        # Message: M(h_u, e_uv)
        msg_input = torch.cat([x_proj[src], edge_feats], dim=-1)
        messages = self.msg_fn(msg_input)
        # Aggregate: sum
        agg = scatter_add(messages, dst, g.num_nodes)
        # Update: GRU(h_v, a_v)
        x_proj = self.update_fn(agg, x_proj)
        return x_proj


class MPNN_DGL(nn.Module):
    """Message Passing Neural Network matching paper's architecture."""

    def __init__(self, in_dim=32, hidden=64, layers=3, out_dim=1, edge_dim=11, dropout=0.3):
        super().__init__()
        self.init_kwargs = {"in_dim": in_dim, "hidden": hidden, "layers": layers,
                            "out_dim": out_dim, "edge_dim": edge_dim, "dropout": dropout}
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.convs.append(MPNNConv(in_dim, hidden, edge_dim=edge_dim))
        for _ in range(layers - 1):
            self.convs.append(MPNNConv(hidden, hidden, edge_dim=edge_dim))
        for _ in range(layers):
            self.norms.append(nn.LayerNorm(hidden))
            self.dropouts.append(nn.Dropout(dropout))
        self.edge_proj = nn.Linear(edge_dim, edge_dim)
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, g: BatchedGraph, x: torch.Tensor) -> torch.Tensor:
        # Edge features are not available in plain graph mode; use zeros
        edge_feats = self.edge_proj(torch.zeros(g.num_edges, 11, device=x.device))
        for i, (conv, norm, drop) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            x = conv(g, x, edge_feats)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = drop(x)
        hg = scatter_mean(x, g.batch, g.num_graphs)
        return self.output(hg)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dgl_model(name: str, in_dim=32, hidden=64, layers=3, heads=4,
                    out_dim=1, edge_dim=11, dropout=0.3) -> nn.Module:
    """Build GNN model by name (matching paper's DGL models)."""
    name = name.lower()
    if name == "gcn":
        return GCN_DGL(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim, dropout=dropout)
    elif name == "gat":
        return GAT_DGL(in_dim=in_dim, hidden=hidden, layers=layers, heads=heads, out_dim=out_dim, dropout=dropout)
    elif name == "mpnn":
        return MPNN_DGL(in_dim=in_dim, hidden=hidden, layers=layers, out_dim=out_dim, edge_dim=edge_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {name}. Available: gcn, gat, mpnn")


# ---------------------------------------------------------------------------
# Dataset and DataLoader
# ---------------------------------------------------------------------------

class MolDataset:
    """Simple dataset for SMILES + labels."""

    def __init__(self, smiles_list: list[str], labels: list[int]):
        self.smiles = smiles_list
        self.labels = labels

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]


def collate_fn(batch):
    """Collate: convert SMILES to batched graph."""
    smiles, labels = zip(*batch)
    valid_idx = []
    valid_smiles = []
    for i, s in enumerate(smiles):
        try:
            if mol_to_graph(s) is not None:
                valid_idx.append(i)
                valid_smiles.append(s)
        except Exception:
            continue

    if not valid_smiles:
        return None, None

    g = smiles_to_graph_batch(valid_smiles)
    labels = torch.tensor([labels[i] for i in valid_idx], dtype=torch.float32)
    return g, labels.unsqueeze(1)


def collate_fn_predict(batch):
    """Collate for prediction (no labels)."""
    smiles = batch
    valid_idx = []
    valid_smiles = []
    for i, s in enumerate(smiles):
        if mol_to_graph(s) is not None:
            valid_idx.append(i)
            valid_smiles.append(s)
    if not valid_smiles:
        return None, None
    g = smiles_to_graph_batch(valid_smiles)
    return g, valid_idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_dgl_model(
    name: str,
    train_smiles: list[str],
    train_labels: list[int],
    val_smiles: list[str] | None = None,
    val_labels: list[int] | None = None,
    hidden: int = 128,
    layers: int = 3,
    heads: int = 8,
    lr: float = 0.001,
    batch_size: int = 128,
    epochs: int = 200,
    patience: int = 15,
    seed: int = 42,
    device: str | torch.device = "cuda",
) -> nn.Module:
    """Train a GNN model."""
    torch.manual_seed(seed)
    device = torch.device(device)

    train_ds = MolDataset(train_smiles, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )

    val_loader = None
    if val_smiles is not None:
        val_ds = MolDataset(val_smiles, val_labels)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn, num_workers=0
        )

    model = build_dgl_model(name, in_dim=32, hidden=hidden, layers=layers, heads=heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for g_batch, labels_batch in train_loader:
            if g_batch is None:
                continue
            g_batch = g_batch.to(device)
            labels_batch = labels_batch.to(device)
            logits = model(g_batch, g_batch.x)
            loss = loss_fn(logits.squeeze(), labels_batch.squeeze())
            opt.zero_grad()
            loss.backward()
            opt.step()

        if val_loader is not None:
            model.eval()
            val_probs, val_true = [], []
            with torch.no_grad():
                for g_batch, labels_batch in val_loader:
                    if g_batch is None:
                        continue
                    g_batch = g_batch.to(device)
                    logits = model(g_batch, g_batch.x)
                    val_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
                    val_true.extend(labels_batch.squeeze().numpy().astype(int))

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
    """Predict probabilities using a trained model."""
    device = torch.device(device)
    model.eval()

    all_probs = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        g = smiles_to_graph_batch(batch)
        if g is None:
            all_probs.extend([0.5] * len(batch))
            continue
        g = g.to(device)
        with torch.no_grad():
            logits = model(g, g.x)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        if probs.ndim == 0:
            probs = [probs.item()]
        all_probs.extend(probs)

    return np.array(all_probs)
