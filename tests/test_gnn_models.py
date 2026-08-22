"""Tests for vegfr2.gnn_models (pure-PyTorch GCN, GAT, MPNN forward/backward/checkpoints)."""

from __future__ import annotations

import pytest
import torch

from vegfr2.features import collate_graphs, mol_to_graph
from vegfr2.gnn_models import build_model, load_checkpoint, save_checkpoint


@pytest.fixture
def batch():
    g1 = mol_to_graph("CCO")
    g2 = mol_to_graph("c1ccccc1")
    g3 = mol_to_graph("CC(=O)Oc1ccccc1C(=O)O")
    return collate_graphs([g1, g2, g3], [1, 0, 1])


@pytest.mark.parametrize("name", ["gcn", "gat", "mpnn"])
def test_gnn_forward_backward(name, batch):
    """Every GNN produces [B, 1] logits and has finite backward gradients."""
    model = build_model(name, in_dim=28, hidden=32, layers=2, heads=2)
    logits = model(batch, device="cpu")

    assert logits.shape == (3, 1)
    assert not torch.isnan(logits).any()

    loss = logits.sum()
    loss.backward()

    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert not torch.isnan(p.grad).any()


@pytest.mark.parametrize("name", ["gcn", "gat", "mpnn"])
def test_gnn_checkpoint_roundtrip(name, batch, tmp_path):
    """Saving and loading checkpoint reproduces identical predictions."""
    model = build_model(name, in_dim=28, hidden=32, layers=2, heads=2)
    model.eval()
    with torch.no_grad():
        orig_logits = model(batch, device="cpu")

    ckpt_path = tmp_path / f"{name}.pt"
    save_checkpoint(model, ckpt_path)

    loaded = load_checkpoint(ckpt_path, device="cpu")
    with torch.no_grad():
        new_logits = loaded(batch, device="cpu")

    assert torch.allclose(orig_logits, new_logits, atol=1e-6)


def test_unknown_gnn_raises():
    with pytest.raises(ValueError):
        build_model("unknown_model")
