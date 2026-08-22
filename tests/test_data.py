"""Tests for vegfr2.data module (labeling, deduplication, preprocessing, splitting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vegfr2.data import deduplicate, label_ic50, load_csv, preprocess, split


def test_label_ic50_boundary():
    """Boundary conditions: <500 is active (1), >=500 is inactive (0)."""
    assert label_ic50(499.9) == 1
    assert label_ic50(0.1) == 1
    assert label_ic50(500.0) == 0
    assert label_ic50(500.1) == 0
    assert label_ic50(10000.0) == 0


def test_label_ic50_nan_raises():
    """NaN/None raises ValueError."""
    with pytest.raises(ValueError):
        label_ic50(float("nan"))


def test_dedup_conflicting_dropped():
    """If multiple distinct IC50 exist for a compound, ALL its rows must be deleted."""
    df = pd.DataFrame({
        "smiles": ["CCO", "CCO", "c1ccccc1"],
        "ic50_nM": [100.0, 200.0, 50.0],
    })
    out = deduplicate(df)
    assert "CCO" not in out["smiles"].values
    assert len(out) == 1
    assert out.iloc[0]["smiles"] == "c1ccccc1"


def test_dedup_identical_kept_once():
    """If multiple rows have the SAME IC50, exactly one row is retained."""
    df = pd.DataFrame({
        "smiles": ["CCO", "CCO", "CCO"],
        "ic50_nM": [100.0, 100.0, 100.0],
    })
    out = deduplicate(df)
    assert len(out) == 1
    assert out.iloc[0]["ic50_nM"] == 100.0


def test_preprocess_drops_invalid(tiny_df):
    """Preprocessing removes invalid SMILES, drops non-numeric IC50, dedups, and adds active."""
    out = preprocess(tiny_df)
    assert "INVALID_SMILES" not in out["smiles"].values
    assert set(out.columns) == {"smiles", "ic50_nM", "active"}
    assert all(out["active"].isin([0, 1]))


def test_split_shapes_and_stratification():
    """Stratified split produces ~8:1:1 train:val:test with preserved class balance."""
    # 90 active, 90 inactive = 180 compounds
    df = pd.DataFrame({
        "smiles": [f"C{'C'*i}O" for i in range(180)],
        "ic50_nM": [100.0] * 90 + [1000.0] * 90,
        "active": [1] * 90 + [0] * 90,
    })
    train_df, val_df, test_df = split(df, seed=42)
    total = len(df)
    assert len(test_df) == pytest.approx(total * 0.10, abs=2)
    assert len(val_df) == pytest.approx(total * 0.10, abs=2)
    assert len(train_df) == pytest.approx(total * 0.80, abs=2)

    # Class balance check in each split
    for split_df in [train_df, val_df, test_df]:
        pos_ratio = split_df["active"].mean()
        assert pos_ratio == pytest.approx(0.5, abs=0.08)


def test_load_csv_missing_column(tmp_path):
    """load_csv raises KeyError if required columns are missing."""
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(bad_csv, index=False)
    with pytest.raises(KeyError):
        load_csv(bad_csv)
