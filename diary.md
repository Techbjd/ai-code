# Diary

## Session 3: TCMSP Bulk SMILES Fetch & Library Merge (2026-09-06)

### Goal
Fetch canonical SMILES from PubChem for all 13,729 TCMSP ingredients and merge into the existing compound library.

### Tasks Completed
- [x] Created `scripts/fetch_tcmsp_smiles.py` with concurrent PubChem API fetching
- [x] Tested PubChem batch endpoints — found comma-separated names unsupported; switched to individual lookups with 10 workers
- [x] Fetched SMILES for 13,729 TCMSP compounds (1,875 valid, 13.7% hit rate)
- [x] Merged with existing 437-compound library, deduplicated by canonical SMILES
- [x] Saved final library: 2,152 unique compounds

### Results
- **1,875** TCMSP compounds with valid SMILES (out of 13,729)
- **2,152** total compounds in merged library (after dedup)
- **160** duplicates removed during merge
- Runtime: ~21 minutes

### Key Decisions
- Used individual name lookups (not batch) because PubChem doesn't support comma-separated names in the `/name/` endpoint
- 10 concurrent workers with RDKit validation per fetch
- TCMSP compounds tagged with herb_name="TCMSP", class="Unknown"
- Deduplicated by canonical_smiles (kept first occurrence)

### Problems Encountered
- PubChem batch endpoint (`/name/{names}/property/SMILES/JSON`) returns 404 for comma-separated names
- POST form data approach also failed for batch names
- Solution: individual concurrent lookups with `SMILES` property (not `CanonicalSMILES` which returns `ConnectivitySMILES`)
- 86.3% of TCMSP names not found in PubChem (likely IUPAC/systematic names or Unicode characters)

### Files Created/Modified
- `scripts/fetch_tcmsp_smiles.py` — New: batch fetcher + merger
- `data/tcmsp_with_smiles.csv` — New: 1,875 TCMSP compounds
- `data/tcm_monomer_library.csv` — Updated: 503 → 2,152 compounds

### Next Steps
- Use expanded library (2,152 compounds) for virtual screening
- Consider alternative SMILES sources for TCMSP compounds not found in PubChem

## Session 2: TCM Compound Library Generation (2026-09-05)

### Goal
Generate a large TCM compound database CSV (`data/tcm_monomer_library.csv`) with 500+ compounds, each having valid RDKit-verified SMILES.

### Tasks Completed
- [x] Created `scripts/compounds_data.py` with 537 raw compound entries
- [x] Created `scripts/generate_tcm_large.py` for dedup + validation + CSV output
- [x] Fetched correct SMILES from PubChem API for ~100+ compounds with problematic hand-written SMILES
- [x] Fixed all invalid SMILES (berberine, palmatine, rutin, hesperidin, etc.)
- [x] Verified 503 unique compounds with 0 invalid SMILES

### Results
- **503** valid compounds, **0** invalid
- **31** compound classes
- **193** unique herbs
- Classes: Flavonoid (51), Alkaloid (46), Flavonoid glycoside (43), Phenolic acid (40), Monoterpenoid (36), Diterpenoid (27), Coumarin (27), Triterpenoid (25), Triterpenoid glycoside (25), and more

### Key Decisions
- Used PubChem API to verify/fetch SMILES instead of hand-writing them
- Created modular script structure: data → generate → CSV
- All SMILES verified with `rdkit.Chem.MolFromSmiles` before writing

### Problems Encountered
- Many hand-written SMILES had ring closure errors (unclosed rings, wrong valence)
- Quaternary nitrogen alkaloids (berberine, palmatine) had invalid SMILES
- Complex glycosides (rutin, hesperidin) needed exact ring notation
- Some PubChem entries returned `ConnectivitySMILES` instead of `CanonicalSMILES`

### Files Created/Modified
- `scripts/compounds_data.py` — 537 raw entries
- `scripts/generate_tcm_large.py` — Main generation script
- `scripts/fetch_tcm_compounds.py` — PubChem API helper
- `data/tcm_monomer_library.csv` — Final output (503 compounds)

### Next Steps
- Use library for virtual screening against VEGFR2
- Integrate with docking pipeline
