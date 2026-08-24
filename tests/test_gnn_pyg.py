"""Tests for vegfr2.gnn_pyg (PyTorch Geometric GNN models including GATv2)."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.loader import DataLoader

from vegfr2.features import mol_to_graph, mol_to_graph_with_fps, get_enriched_node_dim
from vegfr2.gnn_pyg import (
    build_pyg_model,
    PyGDataset,
    EnrichedPyGDataset,
)


@pytest.fixture
def pyg_batch():
    """Create a PyG batch for testing."""
    dataset = PyGDataset(
        ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"],
        [1, 0, 1],
    )
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    return next(iter(loader))


@pytest.fixture
def enriched_batch():
    """Create an enriched PyG batch for testing."""
    dataset = EnrichedPyGDataset(
        ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"],
        [1, 0, 1],
        use_morgan=True,
        use_maccs=True,
    )
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    return next(iter(loader))


@pytest.mark.parametrize("name", ["gcn", "gat", "gatv2", "mpnn"])
def test_pyg_model_forward(name, pyg_batch):
    """PyG models produce [B, 1] logits."""
    model = build_pyg_model(name, in_dim=32, hidden=32, layers=2, heads=2)
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
    """PyG models have finite gradients."""
    model = build_pyg_model(name, in_dim=32, hidden=32, layers=2, heads=2)
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


@pytest.mark.parametrize("name", ["gcn", "gat", "gatv2", "mpnn"])
def test_pyg_enriched_forward(name, enriched_batch):
    """PyG models work with enriched features (2246-dim)."""
    enriched_dim = get_enriched_node_dim(use_morgan=True, use_maccs=True)
    model = build_pyg_model(name, in_dim=enriched_dim, hidden=64, layers=2, heads=2)
    model.eval()
    
    with torch.no_grad():
        if name == "mpnn":
            logits = model(enriched_batch.x, enriched_batch.edge_index, enriched_batch.edge_attr, enriched_batch.batch)
        else:
            logits = model(enriched_batch.x, enriched_batch.edge_index, enriched_batch.batch)
    
    assert logits.shape == (3, 1)
    assert not torch.isnan(logits).any()


def test_gatv2_vs_gat_different():
    """GATv2 and GAT produce different outputs (different architectures)."""
    torch.manual_seed(42)
    gat = build_pyg_model("gat", in_dim=32, hidden=32, layers=2, heads=2)
    torch.manual_seed(42)
    gatv2 = build_pyg_model("gatv2", in_dim=32, hidden=32, layers=2, heads=2)
    
    dataset = PyGDataset(["CCO", "c1ccccc1"], [1, 0])
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    
    gat.eval()
    gatv2.eval()
    
    with torch.no_grad():
        gat_out = gat(batch.x, batch.edge_index, batch.batch)
        gatv2_out = gatv2(batch.x, batch.edge_index, batch.batch)
    
    # They should produce different outputs due to different attention mechanisms
    assert not torch.allclose(gat_out, gatv2_out, atol=1e-4)


def test_build_pyg_model_unknown_raises():
    """Unknown model name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_pyg_model("unknown_model")


def test_build_pyg_model_gatv2():
    """GATv2 model builds correctly."""
    model = build_pyg_model("gatv2", in_dim=2246, hidden=128, layers=3, heads=8)
    assert model is not None
    # Check parameter count is reasonable
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 100000  # Should have substantial parameters
