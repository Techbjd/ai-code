"""Morgan fingerprints, molecular graph construction, and feature utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.logger().setLevel(RDLogger.ERROR)

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
COMPACT_ATOM_FEAT_DIM: int = 32  # Compact version (10 common elements)
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


# ---- Compact atom features (32-dim) ----
# Only 10 most common drug elements, reduced one-hot slots
COMPACT_ATOM_TYPES: list[str] = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'I', 'Si']
COMPACT_DEGREE_SLOTS: list[int] = list(range(6))       # 0-5
COMPACT_IMPLICIT_VALENCE_SLOTS: list[int] = list(range(4))  # 0-3
COMPACT_HYBRIDIZATIONS: list[str] = ['SP', 'SP2', 'SP3']
COMPACT_NUM_HS_SLOTS: list[int] = list(range(4))       # 0-3


def _compact_atom_features(atom: Chem.Atom) -> list[float]:
    """Compact 32-dim atom features for faster training.

    Only keeps 10 most common drug elements + essential chemistry.
    Use with motif graph to avoid information loss.
    """
    # 1. atom_type_one_hot: 10 common elements
    symbol = atom.GetSymbol()
    type_block = _one_hot(symbol, COMPACT_ATOM_TYPES)  # 10

    # 2. atom_degree_one_hot: 0-5
    degree_block = _one_hot(min(atom.GetDegree(), 5), COMPACT_DEGREE_SLOTS)  # 6

    # 3. atom_implicit_valence_one_hot: 0-3
    impl_val = _one_hot(min(atom.GetImplicitValence(), 3), COMPACT_IMPLICIT_VALENCE_SLOTS)  # 4

    # 4. atom_formal_charge: RAW int
    formal_charge = [float(atom.GetFormalCharge())]  # 1

    # 5. atom_num_radical_electrons: RAW int
    radical_electrons = [float(atom.GetNumRadicalElectrons())]  # 1

    # 6. atom_hybridization_one_hot: SP, SP2, SP3
    hyb = str(atom.GetHybridization())
    hybrid_block = _one_hot(hyb, COMPACT_HYBRIDIZATIONS)  # 3

    # 7. atom_is_aromatic: bool
    aromatic = [float(atom.GetIsAromatic())]  # 1

    # 8. atom_total_num_H_one_hot: 0-3
    num_h = _one_hot(min(atom.GetTotalNumHs(), 3), COMPACT_NUM_HS_SLOTS)  # 4

    # 9. in_ring: bool (extra signal for motif graph)
    in_ring = [float(atom.IsInRing())]  # 1

    # 10. chirality (2 bits)
    from rdkit.Chem import chiral_types
    chirality = [
        float(atom.GetChiralTag() != chiral_types.ChiralType.CHI_UNSPECIFIED),
        float(atom.GetChiralTag() == chiral_types.ChiralType.CHI_TETRAHEDRAL_CW),
    ]  # 2

    features = (
        type_block + degree_block + impl_val + formal_charge
        + radical_electrons + hybrid_block + aromatic + num_h
        + in_ring + chirality
    )
    assert len(features) == COMPACT_ATOM_FEAT_DIM, \
        f'expected {COMPACT_ATOM_FEAT_DIM} compact atom features, got {len(features)}'
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


def mol_to_graph(smiles: str, compact: bool = False) -> dict[str, object]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    feat_fn = _compact_atom_features if compact else _atom_features
    rows: list[list[float]] = [feat_fn(atom) for atom in mol.GetAtoms()]
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


# ---------------------------------------------------------------------------
# Motif-level graph construction (NOT done in VEGFR2 papers before)
# Novel approach: decompose molecule into functional groups/rings,
# build a second graph where nodes = motifs, edges = connectivity
# ---------------------------------------------------------------------------

MOTIF_TYPES = [
    'ring',           # 0: cyclic substructure
    'functional',     # 1: functional group (OH, NH2, COOH, etc.)
    'chain',          # 2: aliphatic chain
    'aromatic_ring',  # 3: aromatic ring
    'heteroatom',     # 4: ring with heteroatoms
]
MOTIF_FEAT_DIM = 12  # 5 type one-hot + 7 chemical properties


def _classify_motif(fragment_mol: Chem.rdchem.Mol) -> int:
    """Classify a molecular fragment into a motif type."""
    if fragment_mol is None:
        return 2  # chain
    n_atoms = fragment_mol.GetNumAtoms()
    has_ring = any(ring for ring in fragment_mol.GetRingInfo().AtomRings())
    is_aromatic = any(a.GetIsAromatic() for a in fragment_mol.GetAtoms())
    has_hetero = any(a.GetAtomicNum() not in (6, 1) for a in fragment_mol.GetAtoms())

    if has_ring and is_aromatic:
        return 3  # aromatic_ring
    if has_ring and has_hetero:
        return 4  # heteroatom ring
    if has_ring:
        return 0  # ring
    if n_atoms <= 4:
        return 1  # functional group
    return 2  # chain


def _motif_features(fragment_mol: Chem.rdchem.Mol, atom_map: dict[int, int]) -> list[float]:
    """Featurize a motif fragment (12-dim)."""
    if fragment_mol is None:
        return [0.0] * MOTIF_FEAT_DIM

    motif_type = _classify_motif(fragment_mol)
    type_onehot = _one_hot(motif_type, range(len(MOTIF_TYPES)))  # 5

    n_atoms = fragment_mol.GetNumAtoms()
    n_bonds = fragment_mol.GetNumBonds()
    n_rings = fragment_mol.GetRingInfo().NumRings()
    n_hetero = sum(1 for a in fragment_mol.GetAtoms() if a.GetAtomicNum() not in (6, 1))
    n_aromatic_atoms = sum(1 for a in fragment_mol.GetAtoms() if a.GetIsAromatic())
    avg_degree = sum(a.GetDegree() for a in fragment_mol.GetAtoms()) / max(n_atoms, 1)
    has_donor = any(
        a.GetAtomicNum() in (7, 8) and a.GetTotalNumHs() > 0
        for a in fragment_mol.GetAtoms()
    )
    has_acceptor = any(
        a.GetAtomicNum() in (7, 8) for a in fragment_mol.GetAtoms()
    )

    props = [
        min(n_atoms / 20.0, 1.0),         # normalized atom count
        min(n_bonds / 20.0, 1.0),          # normalized bond count
        min(n_rings / 5.0, 1.0),           # normalized ring count
        min(n_hetero / 10.0, 1.0),         # normalized heteroatom count
        min(n_aromatic_atoms / 10.0, 1.0), # normalized aromatic atom count
        min(avg_degree / 4.0, 1.0),        # normalized avg degree
        float(has_donor) + float(has_acceptor),  # 0, 1, or 2
    ]

    return type_onehot + props


def decompose_to_motifs(smiles: str) -> tuple[list[dict], list[tuple[int, int]]]:
    """Decompose SMILES into motifs and inter-motif connectivity.

    Returns:
        motif_features: list of dicts with 'features' (MOTIF_FEAT_DIM-dim list)
                         and 'atom_indices' (set of original atom indices)
        motif_edges: list of (src_motif_idx, dst_motif_idx) pairs
    """
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], []

    # Strategy: use ring detection + functional group identification
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()

    # Track which atoms belong to which motif
    atom_to_motif: dict[int, int] = {}
    motifs: list[dict] = []

    # 1. Extract ring motifs
    for ring_atoms in rings:
        ring_mol = Chem.PathToSubmol(mol, [])
        # Build submol from ring atoms
        ring_atoms_set = set(ring_atoms)
        atom_map = {a: i for i, a in enumerate(ring_atoms)}
        ring_submol = Chem.RWMol()
        for a_idx in ring_atoms:
            atom = mol.GetAtomWithIdx(a_idx)
            ring_submol.AddAtom(atom)
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in ring_atoms_set and b in ring_atoms_set:
                new_a = ring_atoms.index(a)
                new_b = ring_atoms.index(b)
                ring_submol.AddBond(new_a, new_b, bond.GetBondType())

        feat = _motif_features(ring_submol.GetMol(), atom_map)
        motif_idx = len(motifs)
        motifs.append({
            'features': feat,
            'atom_indices': ring_atoms_set,
        })
        for a in ring_atoms:
            atom_to_motif[a] = motif_idx

    # 2. Extract non-ring fragments (functional groups, chains)
    # Find connected components of non-ring atoms
    ring_atoms_all = set()
    for ring in rings:
        ring_atoms_all.update(ring)

    non_ring_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in ring_atoms_all]

    if non_ring_atoms:
        # BFS to find connected components among non-ring atoms
        visited = set()
        for start in non_ring_atoms:
            if start in visited:
                continue
            component = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited or node in ring_atoms_all:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in mol.GetAtomWithIdx(node).GetNeighbors():
                    n_idx = neighbor.GetIdx()
                    if n_idx not in visited and n_idx not in ring_atoms_all:
                        queue.append(n_idx)
            if component:
                # Also include ring atoms this fragment is attached to
                attached_rings = set()
                for a_idx in component:
                    for neighbor in mol.GetAtomWithIdx(a_idx).GetNeighbors():
                        if neighbor.GetIdx() in ring_atoms_all:
                            attached_rings.add(neighbor.GetIdx())
                all_atoms = set(component) | attached_rings
                atom_map = {a: i for i, a in enumerate(all_atoms)}
                frag_mol = Chem.RWMol()
                for a_idx in all_atoms:
                    frag_mol.AddAtom(mol.GetAtomWithIdx(a_idx))
                for bond in mol.GetBonds():
                    a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if a in all_atoms and b in all_atoms:
                        new_a = list(all_atoms).index(a)
                        new_b = list(all_atoms).index(b)
                        frag_mol.AddBond(new_a, new_b, bond.GetBondType())

                feat = _motif_features(frag_mol.GetMol(), atom_map)
                motif_idx = len(motifs)
                motifs.append({
                    'features': feat,
                    'atom_indices': all_atoms,
                })
                for a in component:
                    atom_to_motif[a] = motif_idx

    # 3. Handle any remaining unassigned atoms
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in atom_to_motif:
            feat = _motif_features(None, {})
            motif_idx = len(motifs)
            motifs.append({
                'features': feat,
                'atom_indices': {idx},
            })
            atom_to_motif[idx] = motif_idx

    # 4. Build inter-motif edges (motifs connected through bonds)
    motif_edges_set = set()
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        m_a = atom_to_motif.get(a)
        m_b = atom_to_motif.get(b)
        if m_a is not None and m_b is not None and m_a != m_b:
            edge = (min(m_a, m_b), max(m_a, m_b))
            motif_edges_set.add(edge)

    motif_edges = list(motif_edges_set)

    return motifs, motif_edges


def mol_to_dual_graph(smiles: str, compact: bool = False) -> dict[str, object] | None:
    """Convert SMILES to dual graph: atom graph + motif graph.

    Args:
        compact: If True, use 32-dim compact atom features instead of 74-dim.

    Returns dict with keys:
        'atom_node_feats', 'atom_edge_index', 'atom_edge_feats', 'num_atom_nodes'
        'motif_node_feats', 'motif_edge_index', 'num_motif_nodes'
        'atom_to_motif': mapping from atom index to motif index
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom graph (74-dim or 32-dim)
    atom_graph = mol_to_graph(smiles, compact=compact)
    if atom_graph is None:
        return None

    # Motif graph
    motifs, motif_edges = decompose_to_motifs(smiles)
    if not motifs:
        return None

    # Motif node features
    motif_feats = torch.tensor(
        np.array([m['features'] for m in motifs], dtype=np.float32)
    )

    # Motif edge index
    if motif_edges:
        src, dst = zip(*motif_edges)
        # Make bidirectional
        motif_edge_index = torch.tensor(
            [list(src) + list(dst), list(dst) + list(src)], dtype=torch.long
        )
    else:
        motif_edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Atom-to-motif mapping
    atom_to_motif = {}
    for m_idx, motif in enumerate(motifs):
        for a_idx in motif['atom_indices']:
            atom_to_motif[a_idx] = m_idx

    # Ensure all atoms have a mapping
    for a in range(mol.GetNumAtoms()):
        if a not in atom_to_motif:
            atom_to_motif[a] = 0

    return {
        'atom_node_feats': atom_graph['node_feats'],
        'atom_edge_index': atom_graph['edge_index'],
        'atom_edge_feats': atom_graph['edge_feats'],
        'num_atom_nodes': atom_graph['num_nodes'],
        'motif_node_feats': motif_feats,
        'motif_edge_index': motif_edge_index,
        'num_motif_nodes': len(motifs),
        'atom_to_motif': atom_to_motif,
    }


MOTIF_GRAPH_DIM = MOTIF_FEAT_DIM
