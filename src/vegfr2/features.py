"""Morgan fingerprints and molecular graph construction/collation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

ATOM_SYMBOLS: list[str] = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Se']
SYMBOL_CHOICES: list[str] = ATOM_SYMBOLS[:11] + ['other']
HYBRIDIZATIONS: list[str] = ['S', 'SP', 'SP2', 'SP3']
DEGREE_SLOTS: list[int] = list(range(7))
CHARGE_CHOICES: list[int] = [-1, 0, 1]
ATOM_FEAT_DIM: int = 28
BOND_FEAT_DIM: int = 5


def _one_hot(value: object, choices: Sequence[object]) -> list[int]:
    """Return a 0/1 indicator list for value within choices."""
    return [1 if choice == value else 0 for choice in choices]


def smiles_to_morgan(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute a Morgan fingerprint as a uint8 bit array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    return np.fromiter((fp[i] for i in range(n_bits)), dtype=np.uint8)


def _atom_features(atom: Chem.Atom) -> list[float]:
    symbol = atom.GetSymbol()
    symbol_block = _one_hot(
        symbol if symbol in SYMBOL_CHOICES else 'other', SYMBOL_CHOICES
    )
    degree_block = _one_hot(min(atom.GetDegree(), 6), DEGREE_SLOTS)
    charge = max(-1, min(1, atom.GetFormalCharge()))
    charge_block = _one_hot(charge, CHARGE_CHOICES)
    aromatic_block = [int(atom.GetIsAromatic())]
    hybridization = str(atom.GetHybridization())
    hybridization_block = _one_hot(
        hybridization if hybridization in HYBRIDIZATIONS else 'other',
        HYBRIDIZATIONS + ['other'],
    )
    features = (
        symbol_block
        + degree_block
        + charge_block
        + aromatic_block
        + hybridization_block
    )
    assert len(features) == ATOM_FEAT_DIM, f'expected {ATOM_FEAT_DIM} atom features'
    return features


def _bond_features(bond: Chem.Bond) -> list[float]:
    btype = bond.GetBondType()
    return [
        int(btype == Chem.BondType.SINGLE),
        int(btype == Chem.BondType.DOUBLE),
        int(btype == Chem.BondType.TRIPLE),
        int(btype == Chem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
    ]


def mol_to_graph(smiles: str) -> dict[str, object]:
    """Build a bidirectional-edge molecular graph with 28-dim node and 5-dim edge features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    rows: list[list[float]] = [_atom_features(atom) for atom in mol.GetAtoms()]
    edge_index: list[list[int]] = []
    edge_rows: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i > j:
            i, j = j, i
        feats = _bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_rows.extend([feats, feats])
    return {
        'node_feats': torch.tensor(np.array(rows, dtype=np.float32)),
        'edge_index': torch.tensor(
            np.array(edge_index, dtype=np.int64).reshape(2, -1), dtype=torch.int64
        ),
        'edge_feats': torch.tensor(
            np.array(edge_rows, dtype=np.float32).reshape(-1, BOND_FEAT_DIM)
        ),
        'num_nodes': mol.GetNumAtoms(),
    }


def collate_graphs(graphs: list[dict], labels: Sequence[int]) -> dict:
    """Batch graphs with offset edges and a node-to-graph index vector."""
    num_graphs = len(graphs)
    node_feats = torch.cat([g['node_feats'] for g in graphs], dim=0)
    edge_feats = torch.cat([g['edge_feats'] for g in graphs], dim=0)
    counts = [int(g['num_nodes']) for g in graphs]
    offsets = torch.tensor([0] + counts[:-1], dtype=torch.int64).cumsum(0)
    edge_index = torch.cat(
        [g['edge_index'] + offset for g, offset in zip(graphs, offsets)], dim=1
    )
    node_batch = torch.repeat_interleave(torch.arange(num_graphs), torch.tensor(counts))
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return {
        'node_feats': node_feats,
        'edge_index': edge_index,
        'edge_feats': edge_feats,
        'node_batch': node_batch,
        'labels': labels_tensor,
        'num_graphs': num_graphs,
    }
