"""Tests for vegfr2.features module (fingerprints, molecular graphs, batching)."""

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
