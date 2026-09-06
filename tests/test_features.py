"""Tests for vegfr2.features module (fingerprints, molecular graphs, batching, GNN embeddings, combinations)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vegfr2.features import (
    ATOM_FEAT_DIM,
    BOND_FEAT_DIM,
    collate_graphs,
    mol_to_graph,
    smiles_to_morgan,
    extract_gnn_embedding,
    extract_gnn_embeddings_batch,
    combine_features,
    get_feature_dim,
    MORGAN_ONLY,
    GNN_ONLY,
    GNN_MORGAN,
)


def test_morgan_shape_dtype():
    fp = smiles_to_morgan("CCO", radius=2, n_bits=2048)
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (2048,)
    assert fp.dtype == np.uint8
    assert fp.sum() > 0


def test_morgan_deterministic():
    fp1 = smiles_to_morgan("CC(=O)Oc1ccccc1C(=O)O", radius=2, n_bits=2048)
    fp2 = smiles_to_morgan("CC(=O)Oc1ccccc1C(=O)O", radius=2, n_bits=2048)
    assert np.array_equal(fp1, fp2)


def test_morgan_invalid_raises():
    with pytest.raises(ValueError):
        smiles_to_morgan("NOT_A_VALID_SMILES")


def test_mol_to_graph_dimensions():
    g = mol_to_graph("CCO")
    assert g["num_nodes"] == 3
    assert g["node_feats"].shape == (3, ATOM_FEAT_DIM)
    assert g["node_feats"].dtype == torch.float32
    assert g["edge_index"].shape == (2, 4)
    assert g["edge_index"].dtype == torch.int64
    assert g["edge_feats"].shape == (4, BOND_FEAT_DIM)
    assert g["edge_feats"].dtype == torch.float32


def test_mol_to_graph_invalid_raises():
    with pytest.raises(ValueError):
        mol_to_graph("NOT_SMILES")


def test_collate_graphs_batching():
    g1 = mol_to_graph("CCO")
    g2 = mol_to_graph("c1ccccc1")
    batch = collate_graphs([g1, g2], [1, 0])

    assert batch["num_graphs"] == 2
    assert batch["node_feats"].shape == (9, ATOM_FEAT_DIM)
    assert batch["edge_index"].shape == (2, 16)
    assert batch["edge_feats"].shape == (16, BOND_FEAT_DIM)
    assert batch["labels"].shape == (2, 1)
    assert batch["node_batch"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1]

    g2_edges_in_batch = batch["edge_index"][:, 4:]
    assert g2_edges_in_batch.min().item() >= 3
    assert g2_edges_in_batch.max().item() <= 8


def test_combine_features_basic():
    arr1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    arr2 = np.array([[5, 6, 7], [8, 9, 10]], dtype=np.float32)
    combined = combine_features(arr1, arr2)
    assert combined.shape == (2, 5)
    assert np.array_equal(combined[:, :2], arr1)
    assert np.array_equal(combined[:, 2:], arr2)


def test_combine_features_single():
    arr = np.array([[1, 2, 3]], dtype=np.float32)
    combined = combine_features(arr)
    assert combined.shape == arr.shape
    assert np.array_equal(combined, arr)


def test_combine_features_mismatched_samples():
    arr1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    arr2 = np.array([[5, 6]], dtype=np.float32)
    with pytest.raises(ValueError, match="Feature array 0 has 2 samples"):
        combine_features(arr1, arr2)


def test_combine_features_empty():
    with pytest.raises(ValueError, match="At least one feature array"):
        combine_features()


def test_get_feature_dim():
    assert get_feature_dim(MORGAN_ONLY) == 2048
    assert get_feature_dim(GNN_ONLY) == 64
    assert get_feature_dim(GNN_MORGAN) == 64 + 2048


def test_get_feature_dim_custom():
    assert get_feature_dim(MORGAN_ONLY, morgan_bits=1024) == 1024
    assert get_feature_dim(GNN_ONLY, gnn_hidden=128) == 128


def test_get_feature_dim_invalid():
    with pytest.raises(ValueError, match="Unknown method"):
        get_feature_dim("invalid_method")


def test_extract_gnn_embedding_shape():
    from vegfr2.gnn_models import build_model

    model = build_model("gcn", in_dim=32, hidden=64, layers=3)
    embedding = extract_gnn_embedding(model, "CCO", device="cpu")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (64,)
    assert embedding.dtype == np.float32


def test_extract_gnn_embedding_deterministic():
    from vegfr2.gnn_models import build_model

    model = build_model("gcn", in_dim=32, hidden=64, layers=3)
    emb1 = extract_gnn_embedding(model, "CCO", device="cpu")
    emb2 = extract_gnn_embedding(model, "CCO", device="cpu")

    assert np.allclose(emb1, emb2)


def test_extract_gnn_embeddings_batch_shape():
    from vegfr2.gnn_models import build_model

    model = build_model("gcn", in_dim=32, hidden=64, layers=3)
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    embeddings = extract_gnn_embeddings_batch(model, smiles_list, device="cpu")

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 64)
    assert embeddings.dtype == np.float32


def test_extract_gnn_embeddings_batch_empty():
    from vegfr2.gnn_models import build_model

    model = build_model("gcn", in_dim=32, hidden=64, layers=3)
    embeddings = extract_gnn_embeddings_batch(model, [], device="cpu")

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, 64)


def test_combined_features_gnn_morgan():
    from vegfr2.gnn_models import build_model

    model = build_model("gcn", in_dim=32, hidden=64, layers=3)
    gnn_emb = extract_gnn_embedding(model, "CCO", device="cpu")
    fp_morgan = smiles_to_morgan("CCO", radius=2, n_bits=2048)
    combined = combine_features(gnn_emb.reshape(1, -1), fp_morgan.reshape(1, -1))
    assert combined.shape == (1, 64 + 2048)


def test_mol_to_graph_deterministic():
    g1 = mol_to_graph("CC(=O)Oc1ccccc1C(=O)O")
    g2 = mol_to_graph("CC(=O)Oc1ccccc1C(=O)O")
    assert torch.allclose(g1["node_feats"], g2["node_feats"])
