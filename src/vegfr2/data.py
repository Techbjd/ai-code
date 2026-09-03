"""Dataset loading, labeling, deduplication and splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split

PathLike = Union[str, Path]

REQUIRED_COLUMNS = ('smiles', 'ic50_nM')


def load_csv(path: PathLike) -> pd.DataFrame:
    """Load a CSV with required 'smiles' and 'ic50_nM' columns."""
    df = pd.read_csv(path)
    if not set(REQUIRED_COLUMNS).issubset(df.columns):
        raise KeyError(f"CSV must contain columns {REQUIRED_COLUMNS}, got {list(df.columns)}")
    return df


def label_ic50(value: float) -> int:
    """Label an IC50 value: 1 if < 500 nM (active), else 0; NaN is invalid."""
    if pd.isna(value):
        raise ValueError('IC50 value is NaN')
    return int(value < 500)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all rows for canonical SMILES with conflicting IC50s; keep first otherwise."""
    def canonical(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else smiles

    keys = df['smiles'].map(canonical)
    keep_indices: list[int] = []
    for key in dict.fromkeys(keys):
        indices = df.index[keys == key]
        if df.loc[indices, 'ic50_nM'].nunique() > 1:
            continue
        keep_indices.append(indices[0])
    return df.loc[sorted(keep_indices)]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Validate SMILES/IC50, deduplicate, label activity; return reset-index frame."""
    out = df.copy()
    out['ic50_nM'] = pd.to_numeric(out['ic50_nM'], errors='coerce')
    out = out.dropna(subset=['ic50_nM'])
    valid = out['smiles'].map(lambda s: Chem.MolFromSmiles(s) is not None)
    out = out[valid]
    out = deduplicate(out)
    out = out.copy()
    out['active'] = out['ic50_nM'].map(label_ic50)
    return out.reset_index(drop=True)[['smiles', 'ic50_nM', 'active']]


def split(
    df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 8:1:1 split into (train, val, test).

    This is the default random stratified split.
    For scaffold-based splitting (recommended for generalization testing),
    use ``scaffold_split`` instead.
    """
    remainder, test_df = train_test_split(
        df, test_size=0.1, stratify=df['active'], random_state=seed
    )
    train_df, val_df = train_test_split(
        remainder, test_size=1 / 9, stratify=remainder['active'], random_state=seed
    )
    return train_df, val_df, test_df


def _murcko_scaffold(smiles: str) -> str:
    """Extract Murcko scaffold from a SMILES string.

    The scaffold is the longest linear chain of atoms that forms the
    backbone of the molecule. Molecules with the same scaffold share
    the same core structure.

    Falls back to the canonical SMILES if scaffold extraction fails.
    """
    try:
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
        return MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return smiles


def scaffold_split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by Murcko scaffolds (80/10/10).

    This ensures the test set contains structurally different molecules
    from the training set, providing a more rigorous evaluation of
    model generalization ability.

    Reference: AttentiveFP (OpenDrugAI) uses this for MoleculeNet tasks.

    Args:
        df: DataFrame with 'smiles' and 'active' columns
        test_size: Fraction for test set (default 0.1)
        val_size: Fraction for validation set (default 0.1)
        seed: Random seed for shuffling within scaffold groups

    Returns:
        (train_df, val_df, test_df) tuple
    """
    import numpy as np

    # Compute scaffolds
    scaffolds = df['smiles'].map(_murcko_scaffold)

    # Group indices by scaffold
    scaffold_groups: dict[str, list[int]] = {}
    for idx, scaffold in zip(df.index, scaffolds):
        scaffold_groups.setdefault(scaffold, []).append(idx)

    # Sort scaffolds by size (largest first) for balanced splitting
    np.random.seed(seed)
    sorted_scaffolds = sorted(scaffold_groups.keys(), key=lambda s: len(scaffold_groups[s]), reverse=True)

    # Assign scaffolds to train/val/test
    n_total = len(df)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)

    test_indices: list[int] = []
    val_indices: list[int] = []
    train_indices: list[int] = []

    for scaffold in sorted_scaffolds:
        indices = scaffold_groups[scaffold]
        # Shuffle within scaffold group
        np.random.shuffle(indices)

        if len(test_indices) < n_test:
            test_indices.extend(indices)
        elif len(val_indices) < n_val:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    # Trim to exact sizes if needed
    test_indices = test_indices[:n_test]
    val_indices = val_indices[:n_val]

    # Ensure no overlap
    train_indices = [i for i in train_indices if i not in set(test_indices) and i not in set(val_indices)]

    return df.loc[train_indices].reset_index(drop=True), \
           df.loc[val_indices].reset_index(drop=True), \
           df.loc[test_indices].reset_index(drop=True)
