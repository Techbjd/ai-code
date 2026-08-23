"""Fast download ChEMBL279 VEGFR2 IC50 data using direct API (no extra molecule fetches)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
import urllib.error
import urllib.request


BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
TARGET_ID = "CHEMBL279"
STANDARD_TYPE = "IC50"
PAGE_SIZE = 1000


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def download_chembl(out_path: Path) -> int:
    rows: list[dict[str, str]] = []
    offset = 0
    total = 0
    print(f"Fetching activities for {TARGET_ID} ({STANDARD_TYPE})...", flush=True)

    while True:
        url = f"{BASE_URL}/activity.json?target_chembl_id={TARGET_ID}&standard_type={STANDARD_TYPE}&limit={PAGE_SIZE}&offset={offset}"
        try:
            data = fetch_json(url)
        except urllib.error.URLError as e:
            raise RuntimeError(f"ChEMBL API error at offset {offset}: {e}") from e

        activities = data.get("activities", [])
        
        if not activities:
            break

        page_rows = []
        for act in activities:
            # Filter: only exact IC50 values (standard_relation "=") and standard_flag=1
            if act.get("standard_relation") != "=":
                continue
            if act.get("standard_flag") != 1:
                continue
                
            smiles = act.get("canonical_smiles")
            val = act.get("standard_value")
            if not smiles or val is None:
                continue
            try:
                ic50 = float(val)
                if ic50 <= 0:
                    continue
            except (ValueError, TypeError):
                continue
            
            page_rows.append({"smiles": smiles, "ic50_nM": str(ic50)})

        if not page_rows:
            break

        rows.extend(page_rows)
        total += len(page_rows)
        print(f"  Offset {offset}: {len(page_rows)} compounds with SMILES (total: {total})", flush=True)

        if len(activities) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles", "ic50_nM"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download VEGFR2 (CHEMBL279) IC50 data (fast)")
    parser.add_argument("--out", default="data/raw/chembl_vegfr2.csv", help="Output CSV path")
    parser.add_argument("--fallback", help="Local CSV to copy if download fails")
    args = parser.parse_args()

    out_path = Path(args.out)
    try:
        download_chembl(out_path)
    except Exception as e:
        print(f"ChEMBL download failed: {e}", file=sys.stderr)
        if args.fallback:
            import shutil
            fallback = Path(args.fallback)
            if fallback.exists():
                shutil.copy2(fallback, out_path)
                print(f"Copied fallback {fallback} -> {out_path}")
                return 0
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())