"""Morgan fingerprints, MACCS keys, molecular graph construction, and feature combination utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys

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


def smiles_to_maccs(smiles: str, n_bits: int = 166) -> np.ndarray:
    """Compute MACCS structural keys as a uint8 bit array.
    
    MACCS keys are 166 predefined structural patterns commonly used in
    cheminformatics for similarity searching and virtual screening.
    
    Args:
        smiles: SMILES string representation of a molecule
        n_bits: Number of MACCS keys (default: 166, standard MACCS length)
    
    Returns:
        numpy array of shape (n_bits,) with dtype uint8
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    fp = MACCSkeys.GenMACCSKeys(mol)
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
    """Build a bidirectional-edge molecular graph with 32-dim node and 11-dim edge features."""
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


def mol_to_graph_with_fps(
    smiles: str,
    use_morgan: bool = True,
    use_maccs: bool = True,
    morgan_radius: int = 2,
    morgan_n_bits: int = 2048,
    maccs_n_bits: int = 166,
) -> dict[str, object]:
    """Build molecular graph with fingerprint features injected into nodes.
    
    This combines graph structure with fingerprint knowledge:
    - Original atom features (32-dim): symbol, degree, charge, etc.
    - Morgan bits (2048-dim): circular substructure patterns
    - MACCS keys (166-dim): structural patterns
    
    Each atom node gets the FULL molecular fingerprint appended.
    This lets the GNN learn message passing while knowing the fingerprint.
    
    Args:
        smiles: SMILES string
        use_morgan: Include Morgan fingerprint bits
        use_maccs: Include MACCS structural keys
        morgan_radius: Morgan fingerprint radius
        morgan_n_bits: Morgan fingerprint bits
        maccs_n_bits: MACCS key bits
    
    Returns:
        Graph dict with enriched node features
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    
    # Get original atom features (32-dim)
    rows: list[list[float]] = [_atom_features(atom) for atom in mol.GetAtoms()]
    
    # Get molecular fingerprints
    fp_features = []
    if use_morgan:
        morgan_fp = smiles_to_morgan(smiles, radius=morgan_radius, n_bits=morgan_n_bits)
        fp_features.append(morgan_fp.astype(np.float32))
    if use_maccs:
        maccs_fp = smiles_to_maccs(smiles, n_bits=maccs_n_bits)
        fp_features.append(maccs_fp.astype(np.float32))
    
    # Concatenate all fingerprint features
    if fp_features:
        mol_fp = np.concatenate(fp_features)  # shape: (n_fp_bits,)
    else:
        mol_fp = np.array([], dtype=np.float32)
    
    n_atoms = mol.GetNumAtoms()
    
    # Each atom gets the FULL molecular fingerprint (molecular-level info)
    # This works because GNN learns to use fingerprint knowledge during message passing
    # All atoms in same molecule get same fingerprint = each atom "knows" the molecule
    enriched_rows = []
    for row in rows:
        enriched_row = row + mol_fp.tolist()  # 32 + 2048 + 166 = 2246
        enriched_rows.append(enriched_row)
    
    # Build edges
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
        'node_feats': torch.tensor(np.array(enriched_rows, dtype=np.float32)),
        'edge_index': torch.tensor(
            np.array(edge_index, dtype=np.int64).reshape(2, -1), dtype=torch.int64
        ),
        'edge_feats': torch.tensor(
            np.array(edge_rows, dtype=np.float32).reshape(-1, BOND_FEAT_DIM)
        ),
        'num_nodes': n_atoms,
    }


def get_enriched_node_dim(
    use_morgan: bool = True,
    use_maccs: bool = True,
    morgan_n_bits: int = 2048,
    maccs_n_bits: int = 166,
) -> int:
    """Get the enriched node feature dimension.
    
    Returns:
        Total node feature dimension (32 + morgan + maccs)
    """
    dim = ATOM_FEAT_DIM  # 32
    if use_morgan:
        dim += morgan_n_bits
    if use_maccs:
        dim += maccs_n_bits
    return dim


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


def collate_enriched_graphs(
    graphs: list[dict],
    labels: Sequence[int],
) -> dict:
    """Batch enriched graphs (with fingerprint info in node features).
    
    Works exactly like collate_graphs but handles larger node feature
    dimensions when fingerprints are injected.
    """
    return collate_graphs(graphs, labels)


def extract_gnn_embedding(
    model: torch.nn.Module,
    smiles: str,
    device: str | torch.device = 'cpu',
) -> np.ndarray:
    """Extract GNN embedding (mean-pooled hidden representation) for a single molecule.
    
    Uses a trained GNN model to generate fixed-size embeddings that can be
    combined with other fingerprint features for enhanced prediction.
    
    Args:
        model: Trained GNN model (GCN, GAT, or MPNN)
        smiles: SMILES string
        device: Device for computation
    
    Returns:
        numpy array of shape (hidden_dim,) with float32 values
    """
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
    """Extract GNN embeddings for a batch of molecules.
    
    Args:
        model: Trained GNN model
        smiles_list: List of SMILES strings
        device: Device for computation
        batch_size: Batch size for processing
    
    Returns:
        numpy array of shape (n_molecules, hidden_dim)
    """
    if len(smiles_list) == 0:
        # Get hidden dim from first linear layer output
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
    """Concatenate multiple feature arrays along the feature dimension.
    
    Args:
        *fps_list: Variable number of numpy arrays, each with shape (n_samples, n_features_i)
    
    Returns:
        Concatenated array of shape (n_samples, sum(n_features_i))
    
    Raises:
        ValueError: If arrays have different number of samples
    """
    if len(fps_list) == 0:
        raise ValueError("At least one feature array must be provided")
    
    n_samples = fps_list[0].shape[0]
    for i, fp in enumerate(fps_list):
        if fp.shape[0] != n_samples:
            raise ValueError(
                f"Feature array 0 has {n_samples} samples, but array {i} has {fp.shape[0]}"
            )
    
    return np.hstack(list(fps_list))


# Feature combination method constants
MORGAN_ONLY = 'morgan_only'
MACCS_ONLY = 'maccs_only'
GNN_ONLY = 'gnn_only'
MORGAN_MACCS = 'morgan_maccs'
GNN_MORGAN = 'gnn_morgan'
GNN_MORGAN_MACCS = 'gnn_morgan_maccs'

SUPPORTED_FEATURE_METHODS = [
    MORGAN_ONLY,
    MACCS_ONLY,
    GNN_ONLY,
    MORGAN_MACCS,
    GNN_MORGAN,
    GNN_MORGAN_MACCS,
]


def get_feature_dim(method: str, morgan_bits: int = 2048, maccs_bits: int = 166, gnn_hidden: int = 64) -> int:
    """Get the output dimension for a given feature combination method.
    
    Args:
        method: Feature combination method name
        morgan_bits: Number of Morgan fingerprint bits
        maccs_bits: Number of MACCS keys
        gnn_hidden: GNN hidden dimension (embedding size)
    
    Returns:
        Total feature dimension
    """
    dims = {
        MORGAN_ONLY: morgan_bits,
        MACCS_ONLY: maccs_bits,
        GNN_ONLY: gnn_hidden,
        MORGAN_MACCS: morgan_bits + maccs_bits,
        GNN_MORGAN: gnn_hidden + morgan_bits,
        GNN_MORGAN_MACCS: gnn_hidden + morgan_bits + maccs_bits,
    }
    if method not in dims:
        raise ValueError(f"Unknown method: {method}. Supported: {SUPPORTED_FEATURE_METHODS}")
    return dims[method]
