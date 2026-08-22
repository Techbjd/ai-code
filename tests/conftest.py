"""Shared pytest fixtures for VEGFR2 tests."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def tiny_df():
    """10-row DataFrame with valid/invalid SMILES, duplicates, conflicting IC50."""
    data = {
        "smiles": [
            "CCO",                    # ethanol - valid
            "c1ccccc1",              # benzene - valid
            "CC(=O)Oc1ccccc1C(=O)O", # aspirin - valid
            "INVALID_SMILES",        # invalid
            "CCO",                   # duplicate exact
            "CCO",                   # duplicate exact (same IC50)
            "CCO",                   # duplicate conflicting IC50
            "c1ccccc1",              # duplicate conflicting IC50
            "CC(C)O",                # isopropanol - valid
            "CC(=O)O",               # acetic acid - valid
        ],
        "ic50_nM": [
            100.0,   # active
            1000.0,  # inactive
            50.0,    # active
            200.0,   # will be dropped (invalid smiles)
            100.0,   # exact duplicate of row 0
            100.0,   # exact duplicate of row 0
            200.0,   # conflicting IC50 for CCO
            50.0,    # conflicting IC50 for benzene
            800.0,   # inactive
            400.0,   # active
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def clean_df():
    """Small clean DataFrame for split/metric tests."""
    return pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1", "CC(C)O", "CC(=O)O", "CCOC", "c1ccccc1C", "CCCCO", "CCCCCCO"],
        "ic50_nM": [100, 1000, 200, 800, 50, 1500, 300, 900],
    })