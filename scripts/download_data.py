"""Download ChEMBL279 VEGFR2 IC50 data to CSV (stdlib only)."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
TARGET_ID = "CHEMBL279"
STANDARD_TYPE = "IC50"
PAGE_SIZE = 1000
SLEEP_SECS = 0.2


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def download_chembl(out_path: Path) -> int:
    rows: list[dict[str, str]] = []
    page = 1
    total = 0
    print(f"Fetching activities for {TARGET_ID} ({STANDARD_TYPE})...", flush=True)

    while True:
        url = f"{BASE_URL}/activity.json?target_chembl_id={TARGET_ID}&standard_type={STANDARD_TYPE}&limit={PAGE_SIZE}&page={page}"
        try:
            data = fetch_json(url)
        except urllib.error.URLError as e:
            raise RuntimeError(f"ChEMBL API error: {e}") from e

        activities = data.get("activities", [])
        if not activities:
            break

        for act in activities:
            mol_id = act.get("molecule_chembl_id")
            val = act.get("standard_value")
            if not mol_id or val is None:
                continue
            try:
                ic50 = float(val)
                if ic50 <= 0:
                    continue
            except (ValueError, TypeError):
                continue

            mol_url = f"{BASE_URL}/molecule/{mol_id}.json"
            try:
                mol_data = fetch_json(mol_url)
                smiles = mol_data.get("molecule_structures", {}).get("canonical_smiles")
            except urllib.error.URLError:
                continue
            if not smiles:
                continue

            rows.append({"smiles": smiles, "ic50_nM": str(ic50)})

        total += len(activities)
        print(f"  Page {page}: {len(activities)} activities, {len(rows)} with SMILES", flush=True)

        if len(activities) < PAGE_SIZE:
            break
        page += 1
        time.sleep(SLEEP_SECS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles", "ic50_nM"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download VEGFR2 (CHEMBL279) IC50 data")
    parser.add_argument("--out", default="data/raw/chembl_vegfr2.csv", help="Output CSV path")
    parser.add_argument("--fallback", help="Local CSV to copy if download fails")
    args = parser.parse_args()

    out_path = Path(args.out)
    try:
        download_chembl(out_path)
    except Exception as e:
        print(f"ChEMBL download failed: {e}", file=sys.stderr)
        if args.fallback:
            fallback = Path(args.fallback)
            if fallback.exists():
                shutil.copy2(fallback, out_path)
                print(f"Copied fallback {fallback} -> {out_path}")
                return 0
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())