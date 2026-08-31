"""Tests for vegfr2.features module (fingerprints, molecular graphs, batching, MACCS, GNN embeddings, combinations, enriched graphs)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vegfr2.features import (
    ATOM_FEAT_DIM,
    BOND_FEAT_DIM,
    collate_graphs,
    collate_enriched_graphs,
    mol_to_graph,
    mol_to_graph_with_fps,
    get_enriched_node_dim,
    smiles_to_morgan,
    smiles_to_maccs,
    extract_gnn_embedding,
    extract_gnn_embeddings_batch,
    combine_features,
    get_feature_dim,
    MORGAN_ONLY,
    MACCS_ONLY,
    GNN_ONLY,
    MORGAN_MACCS,
    GNN_MORGAN,
    GNN_MORGAN_MACCS,
)


def test_morgan_shape_dtype():
    """smiles_to_morgan produces 2048-dim uint8 array with nonzero bits."""
    fp = smiles_to_morgan("CCO", radius=2, n_bits=2048)
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (2048,)
    assert fp.dtype == np.uint8
    assert fp.sum() > 0


def test_morgan_deterministic():
    """Identical SMILES produce identical fingerprints."""
    fp1 = smiles_to_morgan("CC(=O)Oc1ccccc1C(=O)O", radius=2, n_bits=2048)
    fp2 = smiles_to_morgan("CC(=O)Oc1ccccc1C(=O)O", radius=2, n_bits=2048)
    assert np.array_equal(fp1, fp2)


def test_morgan_invalid_raises():
    """Invalid SMILES raises ValueError."""
    with pytest.raises(ValueError):
        smiles_to_morgan("NOT_A_VALID_SMILES")


def test_maccs_shape_dtype():
    """smiles_to_maccs produces 166-dim uint8 array with nonzero bits."""
    fp = smiles_to_maccs("CCO")
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (166,)
    assert fp.dtype == np.uint8
    assert fp.sum() > 0


def test_maccs_deterministic():
    """Identical SMILES produce identical MACCS keys."""
    fp1 = smiles_to_maccs("CC(=O)Oc1ccccc1C(=O)O")
    fp2 = smiles_to_maccs("CC(=O)Oc1ccccc1C(=O)O")
    assert np.array_equal(fp1, fp2)


def test_maccs_invalid_raises():
    """Invalid SMILES raises ValueError."""
    with pytest.raises(ValueError):
        smiles_to_maccs("NOT_A_VALID_SMILES")


def test_maccs_different_molecules():
    """Different molecules produce different MACCS fingerprints."""
    fp1 = smiles_to_maccs("CCO")  # ethanol
    fp2 = smiles_to_maccs("c1ccccc1")  # benzene
    assert not np.array_equal(fp1, fp2)


def test_mol_to_graph_dimensions():
    """Graph featurizer outputs correct tensor dimensions."""
    g = mol_to_graph("CCO")  # 3 heavy atoms: C, C, O; 2 single bonds
    assert g["num_nodes"] == 3
    assert g["node_feats"].shape == (3, ATOM_FEAT_DIM)
    assert g["node_feats"].dtype == torch.float32

    # Bidirectional edges: 2 bonds -> 4 directed edges
    assert g["edge_index"].shape == (2, 4)
    assert g["edge_index"].dtype == torch.int64
    assert g["edge_feats"].shape == (4, BOND_FEAT_DIM)
    assert g["edge_feats"].dtype == torch.float32


def test_mol_to_graph_invalid_raises():
    with pytest.raises(ValueError):
        mol_to_graph("NOT_SMILES")


def test_collate_graphs_batching():
    """collate_graphs correctly offsets edges and sets up node_batch."""
    g1 = mol_to_graph("CCO")       # 3 nodes, 4 edges
    g2 = mol_to_graph("c1ccccc1")  # 6 nodes, 12 edges
    batch = collate_graphs([g1, g2], [1, 0])

    assert batch["num_graphs"] == 2
    assert batch["node_feats"].shape == (9, ATOM_FEAT_DIM)
    assert batch["edge_index"].shape == (2, 16)
    assert batch["edge_feats"].shape == (16, BOND_FEAT_DIM)
    assert batch["labels"].shape == (2, 1)

    # First graph nodes (0..2) -> 0; second graph nodes (3..8) -> 1
    assert batch["node_batch"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1]

    # Edges in second graph must be offset by 3 (num nodes in g1)
    g2_edges_in_batch = batch["edge_index"][:, 4:]
    assert g2_edges_in_batch.min().item() >= 3
    assert g2_edges_in_batch.max().item() <= 8


def test_combine_features_basic():
    """combine_features concatenates arrays correctly."""
    arr1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    arr2 = np.array([[5, 6, 7], [8, 9, 10]], dtype=np.float32)
    combined = combine_features(arr1, arr2)
    assert combined.shape == (2, 5)
    assert np.array_equal(combined[:, :2], arr1)
    assert np.array_equal(combined[:, 2:], arr2)


def test_combine_features_single():
    """combine_features with single array returns same array."""
    arr = np.array([[1, 2, 3]], dtype=np.float32)
    combined = combine_features(arr)
    assert combined.shape == arr.shape
    assert np.array_equal(combined, arr)


def test_combine_features_mismatched_samples():
    """combine_features raises ValueError when sample counts differ."""
    arr1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    arr2 = np.array([[5, 6]], dtype=np.float32)
    with pytest.raises(ValueError, match="Feature array 0 has 2 samples"):
        combine_features(arr1, arr2)


def test_combine_features_empty():
    """combine_features raises ValueError when no arrays provided."""
    with pytest.raises(ValueError, match="At least one feature array"):
        combine_features()


def test_get_feature_dim():
    """get_feature_dim returns correct dimensions for all methods."""
    assert get_feature_dim(MORGAN_ONLY) == 2048
    assert get_feature_dim(MACCS_ONLY) == 166
    assert get_feature_dim(GNN_ONLY) == 64
    assert get_feature_dim(MORGAN_MACCS) == 2048 + 166  # 2214
    assert get_feature_dim(GNN_MORGAN) == 64 + 2048  # 2112
    assert get_feature_dim(GNN_MORGAN_MACCS) == 64 + 2048 + 166  # 2278


def test_get_feature_dim_custom():
    """get_feature_dim respects custom dimension arguments."""
    assert get_feature_dim(MORGAN_ONLY, morgan_bits=1024) == 1024
    assert get_feature_dim(GNN_ONLY, gnn_hidden=128) == 128
    assert get_feature_dim(MORGAN_MACCS, morgan_bits=1024, maccs_bits=166) == 1190


def test_get_feature_dim_invalid():
    """get_feature_dim raises ValueError for unknown method."""
    with pytest.raises(ValueError, match="Unknown method"):
        get_feature_dim("invalid_method")


def test_extract_gnn_embedding_shape():
    """extract_gnn_embedding returns correct shape embedding."""
    from vegfr2.gnn_models import build_model
    
    model = build_model("gcn", in_dim=32, hidden=64, layers=3)
    embedding = extract_gnn_embedding(model, "CCO", device="cpu")
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (64,)
    assert embedding.dtype == np.float32


def test_extract_gnn_embedding_deterministic():
    """Same model produces same embedding for same molecule."""
    from vegfr2.gnn_models import build_model
    
    model = build_model("gcn", in_dim=32, hidden=64, layers=3)
    emb1 = extract_gnn_embedding(model, "CCO", device="cpu")
    emb2 = extract_gnn_embedding(model, "CCO", device="cpu")
    
    assert np.allclose(emb1, emb2)


def test_extract_gnn_embeddings_batch_shape():
    """extract_gnn_embeddings_batch returns correct shape for batch."""
    from vegfr2.gnn_models import build_model
    
    model = build_model("gcn", in_dim=32, hidden=64, layers=3)
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    embeddings = extract_gnn_embeddings_batch(model, smiles_list, device="cpu")
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 64)
    assert embeddings.dtype == np.float32


def test_extract_gnn_embeddings_batch_empty():
    """extract_gnn_embeddings_batch handles empty list."""
    from vegfr2.gnn_models import build_model
    
    model = build_model("gcn", in_dim=32, hidden=64, layers=3)
    embeddings = extract_gnn_embeddings_batch(model, [], device="cpu")
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, 64)


def test_combined_features_morgan_maccs():
    """Morgan + MACCS combination produces correct shape."""
    fp_morgan = smiles_to_morgan("CCO", radius=2, n_bits=2048)
    fp_maccs = smiles_to_maccs("CCO")
    combined = combine_features(fp_morgan.reshape(1, -1), fp_maccs.reshape(1, -1))
    assert combined.shape == (1, 2048 + 166)  # (1, 2214)


def test_combined_features_gnn_morgan():
    """GNN + Morgan combination produces correct shape."""
    from vegfr2.gnn_models import build_model
    
    model = build_model("gcn", in_dim=32, hidden=64, layers=3)
    gnn_emb = extract_gnn_embedding(model, "CCO", device="cpu")
    fp_morgan = smiles_to_morgan("CCO", radius=2, n_bits=2048)
    combined = combine_features(gnn_emb.reshape(1, -1), fp_morgan.reshape(1, -1))
    assert combined.shape == (1, 64 + 2048)  # (1, 2112)


def test_combined_features_gnn_morgan_maccs():
    """GNN + Morgan + MACCS combination produces correct shape."""
    from vegfr2.gnn_models import build_model
    
    model = build_model("gcn", in_dim=32, hidden=64, layers=3)
    gnn_emb = extract_gnn_embedding(model, "CCO", device="cpu")
    fp_morgan = smiles_to_morgan("CCO", radius=2, n_bits=2048)
    fp_maccs = smiles_to_maccs("CCO")
    combined = combine_features(
        gnn_emb.reshape(1, -1),
        fp_morgan.reshape(1, -1),
        fp_maccs.reshape(1, -1),
    )
    assert combined.shape == (1, 64 + 2048 + 166)  # (1, 2278)


def test_mol_to_graph_with_fps_morgan_only():
    """Enriched graph with Morgan only has correct dimensions."""
    g = mol_to_graph_with_fps("CCO", use_morgan=True, use_maccs=False)
    expected_dim = ATOM_FEAT_DIM + 2048  # 32 + 2048 = 2080
    assert g["node_feats"].shape == (3, expected_dim)
    assert g["num_nodes"] == 3


def test_mol_to_graph_with_fps_maccs_only():
    """Enriched graph with MACCS only has correct dimensions."""
    g = mol_to_graph_with_fps("CCO", use_morgan=False, use_maccs=True)
    expected_dim = ATOM_FEAT_DIM + 166  # 32 + 166 = 198
    assert g["node_feats"].shape == (3, expected_dim)
    assert g["num_nodes"] == 3


def test_mol_to_graph_with_fps_both():
    """Enriched graph with Morgan + MACCS has correct dimensions."""
    g = mol_to_graph_with_fps("CCO", use_morgan=True, use_maccs=True)
    expected_dim = ATOM_FEAT_DIM + 2048 + 166  # 32 + 2048 + 166 = 2246
    assert g["node_feats"].shape == (3, expected_dim)
    assert g["num_nodes"] == 3


def test_mol_to_graph_with_fps_no_fp():
    """Enriched graph with no fingerprints is same as original."""
    g_orig = mol_to_graph("CCO")
    g_enriched = mol_to_graph_with_fps("CCO", use_morgan=False, use_maccs=False)
    assert g_enriched["node_feats"].shape == g_orig["node_feats"].shape
    assert torch.allclose(g_enriched["node_feats"], g_orig["node_feats"])


def test_mol_to_graph_with_fps_invalid_raises():
    """Enriched graph raises ValueError for invalid SMILES."""
    with pytest.raises(ValueError):
        mol_to_graph_with_fps("NOT_SMILES")


def test_mol_to_graph_with_fps_all_atoms_same_fp():
    """All atoms in same molecule get identical fingerprint features (molecular-level info)."""
    g = mol_to_graph_with_fps("CCO", use_morgan=True, use_maccs=True)
    # Fingerprint part starts at index ATOM_FEAT_DIM
    fp_part = g["node_feats"][:, ATOM_FEAT_DIM:]
    # All rows should be identical (same molecule fingerprint)
    assert torch.allclose(fp_part[0], fp_part[1])
    assert torch.allclose(fp_part[0], fp_part[2])


def test_get_enriched_node_dim():
    """get_enriched_node_dim returns correct dimensions."""
    assert get_enriched_node_dim(use_morgan=True, use_maccs=True) == 32 + 2048 + 166
    assert get_enriched_node_dim(use_morgan=True, use_maccs=False) == 32 + 2048
    assert get_enriched_node_dim(use_morgan=False, use_maccs=True) == 32 + 166
    assert get_enriched_node_dim(use_morgan=False, use_maccs=False) == 32


def test_collate_enriched_graphs_batching():
    """collate_enriched_graphs works with enriched node features."""
    g1 = mol_to_graph_with_fps("CCO", use_morgan=True, use_maccs=True)
    g2 = mol_to_graph_with_fps("c1ccccc1", use_morgan=True, use_maccs=True)
    batch = collate_enriched_graphs([g1, g2], [1, 0])
    
    expected_dim = ATOM_FEAT_DIM + 2048 + 166  # 2246
    assert batch["num_graphs"] == 2
    assert batch["node_feats"].shape == (9, expected_dim)  # 3 + 6 = 9 atoms
    assert batch["labels"].shape == (2, 1)


def test_enriched_graph_deterministic():
    """Same SMILES produces same enriched graph."""
    g1 = mol_to_graph_with_fps("CC(=O)Oc1ccccc1C(=O)O", use_morgan=True, use_maccs=True)
    g2 = mol_to_graph_with_fps("CC(=O)Oc1ccccc1C(=O)O", use_morgan=True, use_maccs=True)
    assert torch.allclose(g1["node_feats"], g2["node_feats"])
