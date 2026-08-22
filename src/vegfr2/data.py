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
    """Stratified 8:1:1 split into (train, val, test)."""
    remainder, test_df = train_test_split(
        df, test_size=0.1, stratify=df['active'], random_state=seed
    )
    train_df, val_df = train_test_split(
        remainder, test_size=1 / 9, stratify=remainder['active'], random_state=seed
    )
    return train_df, val_df, test_df
