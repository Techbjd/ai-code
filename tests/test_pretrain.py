"""Tests for vegfr2.pretrain module (self-supervised pre-training)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from vegfr2.pretrain import SelfSupervisedPretrainer
from vegfr2.pretrain_models import (
    ContrastiveGNN,
    MaskedAtomGNN,
    GraphAugmentor,
    PretrainDataset,
)
from vegfr2.features import mol_to_graph, ATOM_FEAT_DIM


SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CC1=CC=CC=C1"]


def _make_simple_batch():
    from torch_geometric.data import Data as PyGData
    from torch_geometric.loader import DataLoader
    graphs = [mol_to_graph(s) for s in SMILES[:2]]
    data_list = []
    for i, g in enumerate(graphs):
        data = PyGData(
            x=g["node_feats"],
            edge_index=g["edge_index"],
            edge_attr=g["edge_feats"],
            y=torch.tensor([i], dtype=torch.float32),
        )
        data_list.append(data)
    loader = DataLoader(data_list, batch_size=2, shuffle=False)
    return next(iter(loader))


def _make_enriched_batch():
    from torch_geometric.loader import DataLoader
    dataset = PretrainDataset(SMILES[:2])
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    return next(iter(loader))


class TestGraphAugmentor:
    def test_atom_mask(self):
        data = _make_enriched_batch()
        augmentor = GraphAugmentor(atom_mask_prob=0.5, seed=42)
        aug = augmentor._atom_mask(data)
        assert aug.x.shape == data.x.shape

    def test_edge_drop(self):
        data = _make_enriched_batch()
        augmentor = GraphAugmentor(edge_drop_prob=0.5, seed=42)
        aug = augmentor._edge_drop(data)
        assert aug.edge_index.shape[1] <= data.edge_index.shape[1]

    def test_subgraph(self):
        data = _make_enriched_batch()
        augmentor = GraphAugmentor(subgraph_ratio=0.8, seed=42)
        aug = augmentor._subgraph(data)
        assert aug.x.shape[0] <= data.x.shape[0]

    def test_feature_permute(self):
        data = _make_enriched_batch()
        augmentor = GraphAugmentor(feature_permute_prob=0.5, seed=42)
        aug = augmentor._feature_permute(data)
        assert aug.x.shape == data.x.shape

    def test_call_picks_augmentation(self):
        data = _make_enriched_batch()
        augmentor = GraphAugmentor(seed=42)
        aug = augmentor(data)
        assert aug.x.shape[0] > 0
        assert aug.edge_index.shape[1] > 0

    def test_deterministic(self):
        data = _make_enriched_batch()
        aug1 = GraphAugmentor(atom_mask_prob=0.3, seed=42)
        aug2 = GraphAugmentor(atom_mask_prob=0.3, seed=42)
        torch.manual_seed(42)
        result1 = aug1(data.clone())
        torch.manual_seed(42)
        result2 = aug2(data.clone())
        assert torch.allclose(result1.x, result2.x)


class TestContrastiveGNN:
    def test_encode_shape(self):
        from vegfr2.gnn_pyg import build_pyg_model
        base_gnn = build_pyg_model("gcn", in_dim=32, hidden=64, layers=2, out_dim=64)
        model = ContrastiveGNN(base_gnn, hidden_dim=64, projection_dim=32)

        batch = _make_simple_batch()
        z = model.encode(batch.x, batch.edge_index, batch.batch)

        assert z.shape == (2, 32)
        norms = torch.norm(z, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_contrastive_loss(self):
        from vegfr2.gnn_pyg import build_pyg_model
        base_gnn = build_pyg_model("gcn", in_dim=32, hidden=64, layers=2, out_dim=64)
        model = ContrastiveGNN(base_gnn, hidden_dim=64, projection_dim=32)

        z1 = torch.randn(4, 32)
        z1 = F.normalize(z1, dim=1)
        z2 = torch.randn(4, 32)
        z2 = F.normalize(z2, dim=1)

        loss = model.contrastive_loss(z1, z2)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_forward_shape(self):
        from vegfr2.gnn_pyg import build_pyg_model
        base_gnn = build_pyg_model("gcn", in_dim=32, hidden=64, layers=2, out_dim=64)
        model = ContrastiveGNN(base_gnn, hidden_dim=64, projection_dim=32)

        batch = _make_simple_batch()
        out = model(batch.x, batch.edge_index, batch.batch)
        assert out.shape == (2, 64)


class TestMaskedAtomGNN:
    def test_create_mask(self):
        from vegfr2.gnn_pyg import build_pyg_model
        base_gnn = build_pyg_model("gcn", in_dim=32, hidden=64, layers=2, out_dim=64)
        model = MaskedAtomGNN(base_gnn, hidden_dim=64, atom_feat_dim=32, mask_rate=0.3)

        x = torch.randn(10, 32)
        masked_x, mask = model.create_mask(x, mask_rate=0.3)

        assert masked_x.shape == x.shape
        assert mask.shape == (10,)
        assert mask.dtype == torch.bool
        assert (masked_x[mask] == 0).all()

    def test_masked_prediction_loss(self):
        from vegfr2.gnn_pyg import build_pyg_model
        base_gnn = build_pyg_model("gcn", in_dim=32, hidden=64, layers=2, out_dim=64)
        model = MaskedAtomGNN(base_gnn, hidden_dim=64, atom_feat_dim=32, mask_rate=0.3)

        batch = _make_simple_batch()
        x = batch.x
        loss, mask = model.masked_prediction_loss(x, x, batch.edge_index, batch.batch)

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_forward_shape(self):
        from vegfr2.gnn_pyg import build_pyg_model
        base_gnn = build_pyg_model("gcn", in_dim=32, hidden=64, layers=2, out_dim=64)
        model = MaskedAtomGNN(base_gnn, hidden_dim=64, atom_feat_dim=32)

        batch = _make_simple_batch()
        out = model(batch.x, batch.edge_index, batch.batch)
        assert out.shape == (2, 64)


class TestPretrainDataset:
    def test_dataset_length(self):
        dataset = PretrainDataset(SMILES)
        assert len(dataset) == len(SMILES)

    def test_dataset_item(self):
        dataset = PretrainDataset(SMILES[:1])
        data = dataset[0]
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert data.x.shape[1] == ATOM_FEAT_DIM  # 32-dim plain graph

    def test_dataset_skip_invalid(self):
        dataset = PretrainDataset(["CCO", "INVALID_SMILES", "c1ccccc1"])
        assert len(dataset) == 2


class TestSelfSupervisedPretrainer:
    def test_init_contrastive(self):
        pretrainer = SelfSupervisedPretrainer(
            model_name="gcn", method="contrastive", hidden=32, layers=2
        )
        assert pretrainer.model_name == "gcn"
        assert pretrainer.method == "contrastive"

    def test_init_masked(self):
        pretrainer = SelfSupervisedPretrainer(
            model_name="gcn", method="masked", hidden=32, layers=2
        )
        assert pretrainer.model_name == "gcn"
        assert pretrainer.method == "masked"

    def test_init_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            SelfSupervisedPretrainer(method="invalid")

    def test_repr(self):
        pretrainer = SelfSupervisedPretrainer(
            model_name="gin", method="contrastive", hidden=64
        )
        r = repr(pretrainer)
        assert "gin" in r
        assert "contrastive" in r

    def test_pretrain_few_epochs(self):
        pretrainer = SelfSupervisedPretrainer(
            model_name="gcn", method="contrastive", hidden=32, layers=2,
            batch_size=2, seed=42,
        )
        history = pretrainer.pretrain(
            smiles_list=["CCO", "c1ccccc1", "CC(=O)O", "CC1=CC=CC=C1"],
            epochs=3,
            device="cpu",
            verbose=False,
        )
        assert len(history["train_loss"]) > 0
        assert all(isinstance(l, float) for l in history["train_loss"])

    def test_pretrain_masked(self):
        pretrainer = SelfSupervisedPretrainer(
            model_name="gcn", method="masked", hidden=32, layers=2,
            batch_size=2, seed=42,
        )
        history = pretrainer.pretrain(
            smiles_list=["CCO", "c1ccccc1", "CC(=O)O", "CC1=CC=CC=C1"],
            epochs=3,
            device="cpu",
            verbose=False,
        )
        assert len(history["train_loss"]) > 0

    def test_save_load_roundtrip(self):
        pretrainer = SelfSupervisedPretrainer(
            model_name="gcn", method="contrastive", hidden=32, layers=2,
            batch_size=2, seed=42,
        )
        pretrainer.pretrain(
            smiles_list=["CCO", "c1ccccc1", "CC(=O)O"],
            epochs=2,
            device="cpu",
            verbose=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "pretrained.pt")
            pretrainer.save_pretrained(path)

            loaded = SelfSupervisedPretrainer.load_pretrained(path, device="cpu")
            assert loaded._pretrained
            assert loaded.model_name == "gcn"

            orig_emb = pretrainer.get_embeddings(["CCO"], device="cpu")
            loaded_emb = loaded.get_embeddings(["CCO"], device="cpu")
            assert orig_emb.shape == loaded_emb.shape

    def test_get_embeddings_before_pretrain(self):
        pretrainer = SelfSupervisedPretrainer(model_name="gcn", method="contrastive")
        with pytest.raises(RuntimeError, match="not pre-trained"):
            pretrainer.get_embeddings(["CCO"])

    def test_build_model_factory(self):
        for method in ["contrastive", "masked"]:
            pretrainer = SelfSupervisedPretrainer(
                model_name="gcn", method=method, hidden=32, layers=2
            )
            model = pretrainer._build_pretrain_model()
            assert model is not None


class TestPretrainModelIntegration:
    @pytest.mark.parametrize("name", ["gcn", "gat", "gin", "pna"])
    def test_contrastive_with_each_gnn(self, name: str):
        pretrainer = SelfSupervisedPretrainer(
            model_name=name, method="contrastive", hidden=32, layers=2
        )
        model = pretrainer._build_pretrain_model()
        assert isinstance(model, ContrastiveGNN)

    @pytest.mark.parametrize("name", ["gcn", "gat", "gin", "pna"])
    def test_masked_with_each_gnn(self, name: str):
        pretrainer = SelfSupervisedPretrainer(
            model_name=name, method="masked", hidden=32, layers=2
        )
        model = pretrainer._build_pretrain_model()
        assert isinstance(model, MaskedAtomGNN)
