"""Morgan fingerprints, molecular graph construction, and feature utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

ATOM_SYMBOLS: list[str] = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Se']
SYMBOL_CHOICES: list[str] = ATOM_SYMBOLS[:11] + ['other']
HYBRIDIZATIONS: list[str] = ['S', 'SP', 'SP2', 'SP3']
DEGREE_SLOTS: list[int] = list(range(7))
CHARGE_CHOICES: list[int] = [-1, 0, 1]
CHIRAL_TAGS: list[str] = [
    'CHI_UNSPECIFIED',
    'CHI_TETRAHEDRAL_CW',
    'CHI_TETRAHEDRAL_CCW',
    'CHI_OTHER',
]
BOND_STEREO: list[str] = [
    'STEREONONE',
    'STEREOANY',
    'STEREOZ',
    'STEREOE',
    'STEREOCIS',
    'STEREOTRANS',
]
ATOM_FEAT_DIM: int = 32
BOND_FEAT_DIM: int = 11

_FP_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _one_hot(value: object, choices: Sequence[object]) -> list[int]:
    return [1 if choice == value else 0 for choice in choices]


def smiles_to_morgan(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    cache_key = f"morgan_{radius}_{n_bits}"
    if smiles not in _FP_CACHE:
        _FP_CACHE[smiles] = {}
    if cache_key not in _FP_CACHE[smiles]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'Invalid SMILES: {smiles}')
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = generator.GetFingerprint(mol)
        _FP_CACHE[smiles][cache_key] = np.fromiter((fp[i] for i in range(n_bits)), dtype=np.uint8)
    return _FP_CACHE[smiles][cache_key]


def clear_fp_cache() -> None:
    _FP_CACHE.clear()


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
    chiral_tag = str(atom.GetChiralTag())
    chiral_block = _one_hot(
        chiral_tag if chiral_tag in CHIRAL_TAGS else 'CHI_OTHER', CHIRAL_TAGS
    )
    features = (
        symbol_block
        + degree_block
        + charge_block
        + aromatic_block
        + hybridization_block
        + chiral_block
    )
    assert len(features) == ATOM_FEAT_DIM, f'expected {ATOM_FEAT_DIM} atom features'
    return features


def _bond_features(bond: Chem.Bond) -> list[float]:
    btype = bond.GetBondType()
    stereo = str(bond.GetStereo())
    stereo_block = _one_hot(
        stereo if stereo in BOND_STEREO else 'STEREONONE', BOND_STEREO
    )
    return [
        int(btype == Chem.BondType.SINGLE),
        int(btype == Chem.BondType.DOUBLE),
        int(btype == Chem.BondType.TRIPLE),
        int(btype == Chem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
    ] + stereo_block


def mol_to_graph(smiles: str) -> dict[str, object]:
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


def extract_gnn_embedding(
    model: torch.nn.Module,
    smiles: str,
    device: str | torch.device = 'cpu',
) -> np.ndarray:
    graph = mol_to_graph(smiles)
    batch = collate_graphs([graph], [0])

    batch_t = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    model.eval()
    with torch.no_grad():
        x = batch_t['node_feats'].to(device)
        edge_index = batch_t['edge_index'].to(device)
        node_batch = batch_t['node_batch'].to(device)
        num_graphs = batch_t['num_graphs']

        h = F.relu(model.input(x))

        src, dst = edge_index[0], edge_index[1]
        num_nodes = h.shape[0]

        for i, (lin, ln, drop) in enumerate(
            zip(model.linears, model.norms, model.dropouts)
        ):
            m = h[src]
            agg = torch.zeros_like(h).index_add_(0, dst, m)
            h = F.relu(drop(ln(lin(h + agg))))

        pooled = torch.zeros(num_graphs, h.shape[1], dtype=h.dtype, device=h.device)
        pooled.index_add_(0, node_batch, h)
        counts = torch.bincount(node_batch, minlength=num_graphs).clamp(min=1).unsqueeze(1)
        embedding = (pooled / counts).squeeze(0).cpu().numpy()

    return embedding


def extract_gnn_embeddings_batch(
    model: torch.nn.Module,
    smiles_list: list[str],
    device: str | torch.device = 'cpu',
    batch_size: int = 256,
) -> np.ndarray:
    if len(smiles_list) == 0:
        hidden_dim = model.input.out_features
        return np.zeros((0, hidden_dim), dtype=np.float32)

    model.eval()
    all_embeddings = []

    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i + batch_size]
        graphs = [mol_to_graph(s) for s in batch_smiles]
        batch = collate_graphs(graphs, [0] * len(graphs))

        batch_t = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            x = batch_t['node_feats'].to(device)
            edge_index = batch_t['edge_index'].to(device)
            node_batch = batch_t['node_batch'].to(device)
            num_graphs = batch_t['num_graphs']

            h = F.relu(model.input(x))

            src, dst = edge_index[0], edge_index[1]

            for lin, ln, drop in zip(model.linears, model.norms, model.dropouts):
                m = h[src]
                agg = torch.zeros_like(h).index_add_(0, dst, m)
                h = F.relu(drop(ln(lin(h + agg))))

            pooled = torch.zeros(num_graphs, h.shape[1], dtype=h.dtype, device=h.device)
            pooled.index_add_(0, node_batch, h)
            counts = torch.bincount(node_batch, minlength=num_graphs).clamp(min=1).unsqueeze(1)
            embeddings = (pooled / counts).cpu().numpy()
            all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


def combine_features(*fps_list: np.ndarray) -> np.ndarray:
    if len(fps_list) == 0:
        raise ValueError("At least one feature array must be provided")

    n_samples = fps_list[0].shape[0]
    for i, fp in enumerate(fps_list):
        if fp.shape[0] != n_samples:
            raise ValueError(
                f"Feature array 0 has {n_samples} samples, but array {i} has {fp.shape[0]}"
            )

    return np.hstack(list(fps_list))


MORGAN_ONLY = 'morgan_only'
GNN_ONLY = 'gnn_only'
GNN_MORGAN = 'gnn_morgan'

SUPPORTED_FEATURE_METHODS = [
    MORGAN_ONLY,
    GNN_ONLY,
    GNN_MORGAN,
]


def get_feature_dim(method: str, morgan_bits: int = 2048, gnn_hidden: int = 64) -> int:
    dims = {
        MORGAN_ONLY: morgan_bits,
        GNN_ONLY: gnn_hidden,
        GNN_MORGAN: gnn_hidden + morgan_bits,
    }
    if method not in dims:
        raise ValueError(f"Unknown method: {method}. Supported: {SUPPORTED_FEATURE_METHODS}")
    return dims[method]
