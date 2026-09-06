"""Tests for vegfr2.gnn_pyg (PyTorch Geometric GNN models including GATv2)."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.loader import DataLoader

from vegfr2.features import ATOM_FEAT_DIM, mol_to_graph
from vegfr2.gnn_pyg import (
    build_pyg_model,
    PlainPyGDataset,
)


@pytest.fixture
def pyg_batch():
    """Create a PyG batch for testing (32-dim atom features only)."""
    dataset = PlainPyGDataset(
        ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"],
        [1, 0, 1],
    )
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    return next(iter(loader))


@pytest.mark.parametrize("name", ["gcn", "gat", "gatv2", "mpnn"])
def test_pyg_model_forward(name, pyg_batch):
    """PyG models produce [B, 1] logits with DGL-style atom features."""
    model = build_pyg_model(name, in_dim=ATOM_FEAT_DIM, hidden=32, layers=2, heads=2)
    model.eval()

    with torch.no_grad():
        if name == "mpnn":
            logits = model(pyg_batch.x, pyg_batch.edge_index, pyg_batch.edge_attr, pyg_batch.batch)
        else:
            logits = model(pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch)

    assert logits.shape == (3, 1)
    assert not torch.isnan(logits).any()


@pytest.mark.parametrize("name", ["gcn", "gat", "gatv2", "mpnn"])
def test_pyg_model_backward(name, pyg_batch):
    """PyG models have finite gradients with DGL-style atom features."""
    model = build_pyg_model(name, in_dim=ATOM_FEAT_DIM, hidden=32, layers=2, heads=2)
    model.train()

    if name == "mpnn":
        logits = model(pyg_batch.x, pyg_batch.edge_index, pyg_batch.edge_attr, pyg_batch.batch)
    else:
        logits = model(pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch)

    loss = logits.sum()
    loss.backward()

    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert not torch.isnan(p.grad).any()


def test_gatv2_vs_gat_different():
    """GATv2 and GAT produce different outputs (different architectures)."""
    torch.manual_seed(42)
    gat = build_pyg_model("gat", in_dim=ATOM_FEAT_DIM, hidden=32, layers=2, heads=2)
    torch.manual_seed(42)
    gatv2 = build_pyg_model("gatv2", in_dim=ATOM_FEAT_DIM, hidden=32, layers=2, heads=2)

    dataset = PlainPyGDataset(["CCO", "c1ccccc1"], [1, 0])
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    gat.eval()
    gatv2.eval()

    with torch.no_grad():
        gat_out = gat(batch.x, batch.edge_index, batch.batch)
        gatv2_out = gatv2(batch.x, batch.edge_index, batch.batch)

    assert not torch.allclose(gat_out, gatv2_out, atol=1e-4)


def test_build_pyg_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_pyg_model("unknown_model")


def test_build_pyg_model_gatv2():
    model = build_pyg_model("gatv2", in_dim=ATOM_FEAT_DIM, hidden=128, layers=3, heads=8)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 10000  # Should have reasonable parameters
