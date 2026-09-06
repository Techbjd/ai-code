"""Morgan fingerprints, molecular graph construction, and feature utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# ---- DGL CanonicalAtomFeaturizer (exact match) ----
# Source: github.com/awslabs/dgl-lifesci/python/dgllife/utils/featurizers.py
DGL_ATOM_TYPES: list[str] = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
    'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
    'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
]
DGL_DEGREE_SLOTS: list[int] = list(range(11))       # 0-10
DGL_IMPLICIT_VALENCE_SLOTS: list[int] = list(range(7))  # 0-6
DGL_HYBRIDIZATIONS: list[str] = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
DGL_NUM_HS_SLOTS: list[int] = list(range(5))         # 0-4
BOND_STEREO: list[str] = [
    'STEREONONE',
    'STEREOANY',
    'STEREOZ',
    'STEREOE',
    'STEREOCIS',
    'STEREOTRANS',
]
ATOM_FEAT_DIM: int = 74  # Exact DGL CanonicalAtomFeaturizer
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
    """Exact DGL CanonicalAtomFeaturizer (74-dim).

    Source: dgllife/utils/featurizers.py
    """
    # 1. atom_type_one_hot: 43 elements
    symbol = atom.GetSymbol()
    type_block = _one_hot(symbol, DGL_ATOM_TYPES)  # 43

    # 2. atom_degree_one_hot: 0-10
    degree_block = _one_hot(min(atom.GetDegree(), 10), DGL_DEGREE_SLOTS)  # 11

    # 3. atom_implicit_valence_one_hot: 0-6
    impl_val = _one_hot(min(atom.GetImplicitValence(), 6), DGL_IMPLICIT_VALENCE_SLOTS)  # 7

    # 4. atom_formal_charge: RAW int (not one-hot)
    formal_charge = [float(atom.GetFormalCharge())]  # 1

    # 5. atom_num_radical_electrons: RAW int (not one-hot)
    radical_electrons = [float(atom.GetNumRadicalElectrons())]  # 1

    # 6. atom_hybridization_one_hot: SP, SP2, SP3, SP3D, SP3D2
    hyb = str(atom.GetHybridization())
    hybrid_block = _one_hot(hyb, DGL_HYBRIDIZATIONS)  # 5

    # 7. atom_is_aromatic: bool
    aromatic = [float(atom.GetIsAromatic())]  # 1

    # 8. atom_total_num_H_one_hot: 0-4
    num_h = _one_hot(min(atom.GetTotalNumHs(), 4), DGL_NUM_HS_SLOTS)  # 5

    features = (
        type_block + degree_block + impl_val + formal_charge
        + radical_electrons + hybrid_block + aromatic + num_h
    )
    assert len(features) == ATOM_FEAT_DIM, f'expected {ATOM_FEAT_DIM} atom features, got {len(features)}'
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
