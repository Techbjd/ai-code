#!/usr/bin/env python3
"""Fetch canonical SMILES from PubChem for TCMSP ingredients and merge into monomer library."""

import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openpyxl
import pandas as pd
import requests
from rdkit import Chem

WORKDIR = Path("/home/bijay/Desktop/already read paper/ai-code")
DATA_DIR = WORKDIR / "data"

XLSX_PATH = DATA_DIR / "tcmsp_ingredients.xlsx"
TCMSP_SMILES_PATH = DATA_DIR / "tcmsp_with_smiles.csv"
MERGED_PATH = DATA_DIR / "tcm_monomer_library.csv"

PROGRESS_EVERY = 500
MAX_WORKERS = 10
DELAY_BETWEEN_BATCHES = 0.1

PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}"
    "/property/SMILES/JSON"
)


def load_tcmsp_names() -> list[str]:
    wb = openpyxl.load_workbook(XLSX_PATH, read_only=True)
    ws = wb.active
    names = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        name = row[2]  # molecule_name is column C (index 2)
        if name:
            names.append(str(name).strip())
    wb.close()
    print(f"Loaded {len(names)} compound names from {XLSX_PATH.name}", flush=True)
    return names


def fetch_one(name: str) -> tuple[str, str | None]:
    """Fetch SMILES for a single compound name."""
    url = PUBCHEM_URL.format(name=name)
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                smi = props[0].get("SMILES")
                if smi:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        return (name, smi)
    except Exception:
        pass
    return (name, None)


def fetch_all_smiles(names: list[str]) -> list[tuple[str, str]]:
    """Fetch SMILES for all names using concurrent workers."""
    results = []
    total = len(names)
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, n): n for n in names}
        for future in as_completed(futures):
            name, smi = future.result()
            done += 1
            if smi:
                results.append((name, smi))
            if done % PROGRESS_EVERY == 0 or done == total:
                print(
                    f"  [{done}/{total}] fetched, {len(results)} valid SMILES",
                    flush=True,
                )

    return results


def save_tcmsp_smiles(results: list[tuple[str, str]]):
    with open(TCMSP_SMILES_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["molecule_name", "canonical_smiles"])
        for name, smi in results:
            w.writerow([name, smi])
    print(f"Saved {len(results)} compounds to {TCMSP_SMILES_PATH.name}", flush=True)


def merge_libraries():
    existing = pd.read_csv(MERGED_PATH)
    tcmsp = pd.read_csv(TCMSP_SMILES_PATH)
    tcmsp["herb_name"] = "TCMSP"
    tcmsp["class"] = "Unknown"
    tcmsp = tcmsp[["molecule_name", "canonical_smiles", "herb_name", "class"]]

    combined = pd.concat([existing, tcmsp], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset="canonical_smiles", keep="first")
    after = len(combined)

    combined.to_csv(MERGED_PATH, index=False)
    print(f"\nMerge results:", flush=True)
    print(f"  Existing library: {len(existing)} compounds", flush=True)
    print(f"  TCMSP new:        {len(tcmsp)} compounds", flush=True)
    print(f"  Combined before dedup: {before}", flush=True)
    print(f"  Combined after dedup:  {after}", flush=True)
    print(f"  Removed duplicates:    {before - after}", flush=True)
    print(f"  Saved to: {MERGED_PATH.name}", flush=True)


def main():
    names = load_tcmsp_names()

    print(
        f"\nFetching SMILES from PubChem (workers={MAX_WORKERS})...", flush=True
    )
    t0 = time.time()
    results = fetch_all_smiles(names)
    elapsed = time.time() - t0
    print(
        f"\nFetched {len(results)} compounds with valid SMILES "
        f"out of {len(names)} total in {elapsed:.0f}s",
        flush=True,
    )

    save_tcmsp_smiles(results)
    merge_libraries()


if __name__ == "__main__":
    main()
