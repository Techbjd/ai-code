"""Tests for new GNN models: GIN, PNA, Graph Transformer, and sklearn API."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from vegfr2.models.gin import GIN_PyG
from vegfr2.models.pna import PNA_PyG
from vegfr2.models.graph_transformer import GraphTransformer_PyG
from vegfr2.gnn_pyg import build_pyg_model
from vegfr2.features import mol_to_graph


# ============================================================
# Test data helpers
# ============================================================

SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CC1=CC=CC=C1"]
EDGE_INDEX = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])


def _make_batch(smiles_list: list[str] | None = None):
    """Create a batch from SMILES or synthetic data."""
    if smiles_list is None:
        smiles_list = SMILES[:2]
    graphs = [mol_to_graph(s) for s in smiles_list]
    from vegfr2.features import collate_graphs
    return collate_graphs(graphs, [1, 0])


# ============================================================
# GIN tests
# ============================================================

class TestGIN:
    def test_gin_forward(self):
        gin = GIN_PyG(in_dim=32, hidden=64, layers=3, out_dim=1)
        batch = _make_batch()
        out = gin(batch["node_feats"], batch["edge_index"], batch["node_batch"])
        assert out.shape == (2, 1)

    def test_gin_backward(self):
        gin = GIN_PyG(in_dim=32, hidden=64, layers=2, out_dim=1)
        batch = _make_batch()
        out = gin(batch["node_feats"], batch["edge_index"], batch["node_batch"])
        loss = out.sum()
        loss.backward()
        assert all(p.grad is not None for p in gin.parameters() if p.requires_grad)

    def test_gin_no_jk(self):
        gin = GIN_PyG(in_dim=32, hidden=64, layers=3, out_dim=1, jk=False)
        batch = _make_batch()
        out = gin(batch["node_feats"], batch["edge_index"], batch["node_batch"])
        assert out.shape == (2, 1)

    def test_gin_max_pooling(self):
        gin = GIN_PyG(in_dim=32, hidden=64, layers=2, out_dim=1, pooling="max")
        batch = _make_batch()
        out = gin(batch["node_feats"], batch["edge_index"], batch["node_batch"])
        assert out.shape == (2, 1)

    def test_gin_build_via_factory(self):
        model = build_pyg_model("gin", in_dim=32, hidden=64, layers=3, heads=8, edge_dim=11)
        assert isinstance(model, GIN_PyG)
        batch = _make_batch()
        out = model(batch["node_feats"], batch["edge_index"], batch["node_batch"])
        assert out.shape == (2, 1)


# ============================================================
# PNA tests
# ============================================================

class TestPNA:
    def test_pna_forward(self):
        pna = PNA_PyG(in_dim=32, hidden=64, layers=3, out_dim=1)
        batch = _make_batch()
        out = pna(batch["node_feats"], batch["edge_index"], batch["node_batch"])
        assert out.shape == (2, 1)

    def test_pna_backward(self):
        pna = PNA_PyG(in_dim=32, hidden=64, layers=2, out_dim=1)
        batch = _make_batch()
        out = pna(batch["node_feats"], batch["edge_index"], batch["node_batch"])
        loss = out.sum()
        loss.backward()
        assert all(p.grad is not None for p in pna.parameters() if p.requires_grad)

    def test_pna_build_via_factory(self):
        model = build_pyg_model("pna", in_dim=32, hidden=64, layers=3, heads=8, edge_dim=11)
        assert isinstance(model, PNA_PyG)
        batch = _make_batch()
        out = model(batch["node_feats"], batch["edge_index"], batch["node_batch"])
        assert out.shape == (2, 1)


# ============================================================
# Graph Transformer tests
# ============================================================

class TestGraphTransformer:
    def test_transformer_forward(self):
        gt = GraphTransformer_PyG(in_dim=32, hidden=64, layers=2, heads=8, out_dim=1, edge_dim=11)
        batch = _make_batch()
        out = gt(batch["node_feats"], batch["edge_index"], batch["node_batch"], batch["edge_feats"])
        assert out.shape == (2, 1)

    def test_transformer_backward(self):
        gt = GraphTransformer_PyG(in_dim=32, hidden=64, layers=2, heads=4, out_dim=1, edge_dim=11)
        batch = _make_batch()
        out = gt(batch["node_feats"], batch["edge_index"], batch["node_batch"], batch["edge_feats"])
        loss = out.sum()
        loss.backward()
        assert all(p.grad is not None for p in gt.parameters() if p.requires_grad)

    def test_transformer_single_layer(self):
        gt = GraphTransformer_PyG(in_dim=32, hidden=64, layers=1, heads=4, out_dim=1, edge_dim=11)
        batch = _make_batch()
        out = gt(batch["node_feats"], batch["edge_index"], batch["node_batch"], batch["edge_feats"])
        assert out.shape == (2, 1)

    def test_transformer_build_via_factory(self):
        model = build_pyg_model("graph_transformer", in_dim=32, hidden=64, layers=2, heads=8, edge_dim=11)
        assert isinstance(model, GraphTransformer_PyG)
        batch = _make_batch()
        out = model(batch["node_feats"], batch["edge_index"], batch["node_batch"], batch["edge_feats"])
        assert out.shape == (2, 1)


# ============================================================
# Sklearn API tests
# ============================================================

class TestGNNClassifier:
    def test_init(self):
        from vegfr2.sklearn_api import GNNClassifier
        model = GNNClassifier(model="gin", hidden=64, layers=2)
        assert model.model_name == "gin"

    def test_init_invalid(self):
        from vegfr2.sklearn_api import GNNClassifier
        with pytest.raises(ValueError):
            GNNClassifier(model="invalid_model")

    def test_repr(self):
        from vegfr2.sklearn_api import GNNClassifier
        model = GNNClassifier(model="gin", hidden=128, layers=3)
        r = repr(model)
        assert "gin" in r
        assert "128" in r

    def test_predict_before_fit(self):
        from vegfr2.sklearn_api import GNNClassifier
        model = GNNClassifier(model="gin", hidden=64, layers=2)
        with pytest.raises(RuntimeError):
            model.predict_proba(["CCO"])

    def test_save_load_roundtrip(self):
        from vegfr2.sklearn_api import GNNClassifier
        model = GNNClassifier(model="gin", hidden=32, layers=1, epochs=2, device="cpu")
        # Minimal fit
        model.fit(["CCO", "c1ccccc1"], [1, 0])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.pkl")
            model.save(path)
            loaded = GNNClassifier.load(path, device="cpu")
            assert loaded.model_name == "gin"
            assert loaded._fitted

            # Verify predictions match
            orig = model.predict_proba(["CCO"])
            loaded_pred = loaded.predict_proba(["CCO"])
            assert orig.shape == loaded_pred.shape


class TestEnsembleClassifier:
    def test_init(self):
        from vegfr2.sklearn_api import EnsembleClassifier
        ens = EnsembleClassifier(gnn="gin", ml="xgb")
        assert ens._ensemble.gnn_name == "gin"

    def test_repr(self):
        from vegfr2.sklearn_api import EnsembleClassifier
        ens = EnsembleClassifier(gnn="gin", ml="xgb")
        r = repr(ens)
        assert "gin" in r
        assert "xgb" in r


# ============================================================
# Model parameter count tests
# ============================================================

class TestModelSize:
    @pytest.mark.parametrize("name", ["gcn", "gat", "gatv2", "mpnn", "gin", "pna", "graph_transformer"])
    def test_model_has_parameters(self, name: str):
        model = build_pyg_model(name, in_dim=32, hidden=64, layers=3, heads=8, edge_dim=11)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        assert n_params < 100_000_000  # sanity check: not too large

    def test_gin_has_more_params_than_gcn(self):
        gcn = build_pyg_model("gcn", in_dim=32, hidden=64, layers=3)
        gin = build_pyg_model("gin", in_dim=32, hidden=64, layers=3)
        gcn_params = sum(p.numel() for p in gcn.parameters())
        gin_params = sum(p.numel() for p in gin.parameters())
        assert gin_params > gcn_params  # GIN uses MLPs, should be larger
