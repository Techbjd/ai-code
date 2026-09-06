#!/usr/bin/env python3
"""Generate a large TCM compound database CSV with 500+ real compounds.

Writes to data/tcm_monomer_library.csv with columns:
    molecule_name, canonical_smiles, herb_name, class

Each SMILES is verified using rdkit.Chem.MolFromSmiles.
"""

import csv
import os
import sys
from pathlib import Path

from rdkit import Chem

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "tcm_monomer_library.csv"


def get_compounds():
    """Return the full compound list: (name, smiles, herb, class)."""
    from compounds_data import COMPOUNDS
    return COMPOUNDS


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    from compounds_data import COMPOUNDS
    print(f"Raw compound entries: {len(COMPOUNDS)}")

    # Deduplicate by (name, canonical_smiles)
    seen = set()
    unique = []
    for name, smi, herb, cls in COMPOUNDS:
        key = (name, smi)
        if key not in seen:
            seen.add(key)
            unique.append((name, smi, herb, cls))

    print(f"After deduplication: {len(unique)}")

    # Validate SMILES
    valid_rows = []
    invalid = 0
    for name, smi, herb, cls in unique:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_rows.append({
                "molecule_name": name,
                "canonical_smiles": Chem.MolToSmiles(mol),
                "herb_name": herb,
                "class": cls,
            })
        else:
            invalid += 1
            print(f"  INVALID SMILES: {name} | {smi}")

    print(f"Valid SMILES: {len(valid_rows)}, Invalid: {invalid}")

    # Write CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["molecule_name", "canonical_smiles", "herb_name", "class"]
        )
        writer.writeheader()
        writer.writerows(valid_rows)

    print(f"Written {len(valid_rows)} compounds to {OUTPUT_FILE}")

    # Verify by reading back
    import pandas as pd
    df = pd.read_csv(OUTPUT_FILE)
    print(f"Verification: {len(df)} rows, {df['class'].nunique()} classes, {df['herb_name'].nunique()} herbs")
    print(f"Class distribution:\n{df['class'].value_counts().to_string()}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
