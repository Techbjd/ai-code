# Diary

## Session 4: NOVEL Dual-Graph GNN (Atom + Motif) for VEGFR2 (2026-09-06)

### Goal
Add a novel motif-level graph feature to improve GNN pattern finding for VEGFR2 virtual screening. This approach is NOT done in any VEGFR2 paper before — it captures higher-level chemical semantics (functional groups, rings) that pure atom-level GNNs miss.

### Tasks Completed
- [x] Added motif decomposition to `src/vegfr2/features.py` — decomposes SMILES into functional groups/rings using RDKit
- [x] Added motif featurization (12-dim): 5 motif type one-hot + 7 chemical properties
- [x] Added `mol_to_dual_graph()` — builds both atom graph AND motif graph from SMILES
- [x] Added `DualGraphGNN` model to `src/vegfr2/gnn_dgl.py` — processes both graphs with cross-level attention fusion
- [x] Added `DualMolDataset`, `dual_collate_fn`, `build_dual_model`, `predict_dual_model`
- [x] Updated `colab_screening.py` to train and screen with 7 models (was 6)
- [x] Fixed RDKit deprecation warnings in all files that import rdkit

### Novel Approach (NOT done in VEGFR2 papers)
- **Motif decomposition**: Extract rings, aromatic rings, heteroatom rings, functional groups, chains
- **Motif graph**: Build second graph where nodes = motifs, edges = inter-motif connectivity
- **Cross-level attention**: Fuse atom embeddings and motif embeddings via learned attention
- **Dual encoding**: GCN processes atom graph separately from motif graph, then combines

### Architecture
```
SMILES → Mol → [Atom Graph (74-dim)] → GCN → atom embeddings
         ↓
         [Motif Graph (12-dim)] → GCN → motif embeddings
         ↓
         Cross-level attention fusion → prediction
```

### Files Modified
- `src/vegfr2/features.py` — Added: motif decomposition, dual graph construction, compact atom features (32-dim)
- `src/vegfr2/gnn_dgl.py` — Added: DualGraphGNN model, dataset, collate functions, compact mode support
- `colab_screening.py` — Updated: 7 models (was 6), dual model training + screening, USE_COMPACT toggle

### Key Decisions
- Used 12-dim motif features (5 type + 7 properties) to keep computation fast
- Motif types: ring, functional group, chain, aromatic ring, heteroatom ring
- Cross-level attention rather than simple concatenation for better fusion
- Smaller batch size (64) for dual model due to motif decomposition overhead

### What's Different from Other Papers
- **FnGATGCN (Wang 2024)**: Morgan + GAT-GCN (no motifs)
- **GEM-GNN (Xu 2024)**: 3D geometric features (no motifs)
- **HimNet**: Hierarchical but uses PyTorch Geometric (we use pure PyTorch)
- **DFusMol**: Motif graph but not for VEGFR2
- **Our approach**: Motif-level graph + cross-level attention, pure PyTorch, VEGFR2-specific

### Next Steps
- Run `colab_screening.py` to test dual model performance
- Compare DualGraphGNN AUC/MCC vs GCN/GAT/MPNN
- If dual model performs well, it's a novel contribution for VEGFR2

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
