# REVIEW.md — Code Quality & Paper Fidelity Audit

**Target Paper:** Hou et al. 2025, VEGFR2 GNN Virtual Screening (DOI 10.1080/14756366.2025.2518192)

An audit of the implementation in `src/vegfr2/` and `scripts/` was conducted to verify accuracy, safety, and strict alignment with paper constraints and standards.

---

## 1. Paper Fidelity Audit

*   **Target and Preprocessing:**
    *   **CHEMBL279 (VEGFR2):** Correctly targeted in `download_data.py`.
    *   **Deduplication Rules:** Rigorously implemented in `data.deduplicate()`:
        *   If multiple distinct IC50 values exist for a compound, *all* its rows are deleted.
        *   If identical duplicate IC50 values exist, *exactly one* is kept.
    *   **Activity Boundary:** Enforces `ic50 < 500 nM => active=1` else `0`.
*   **Stratified Splitting:** `data.split()` correctly uses stratified `train_test_split` to create a deterministic, stratified `8:1:1` train/val/test split (verified by test suite).
*   **Model Architectures (Pure PyTorch, DGL-Free):**
    *   **GCN:** Implements Equation (1) symmetric degree normalization correctly with bidirectional edges.
    *   **GAT:** Implements multi-head attention scores, softmax normalization per neighborhood, and LayerNorm correctly.
    *   **MPNN:** Implements message-passing MLPs and GRU updates per step correctly.
*   **Evaluation Metrics:**
    *   `metrics.classification_metrics()` implements ACC, SEN (recall), SPE (specificity), MCC, and AUC identically to the paper equations. None-safe behavior is correctly implemented when only a single class is present.

---

## 2. Hard Constraints & Standards

*   **GPU Enforcement:** `get_device()` in `src/vegfr2/device.py` acts as a strict guard. Both `scripts/train.py` and `scripts/screen.py` call `get_device()` as the first line of execution logic, aborting with a `RuntimeError` on CPU-only machines.
*   **CI Testability:** Pure PyTorch implementation accepts a `device` parameter in all GNN modules (defaulting to `'cpu'`), allowing unit tests to run and verify graph computation graphs on CPU (verified: 31/31 tests passed).
*   **Standards:** Full PEP8 styling, absolute type-hints, Pathlib-driven IO, and rigorous error handling are preserved.

---

## 3. Improvements Applied

*   *Bugfix:* Updated `rdkit.Chem.rdFingerprintGenerator` call to use `GetMorganGenerator` instead of `MorganGenerator` (since the latter was deprecated/removed in the current RDKit release).
*   *Bugfix:* Corrected GAT multi-head tensor multiplication broadcasting from `alpha.unsqueeze(-1)` to `alpha[:, None, None]` to prevent shape mismatches during batched graph computation.

---

**Audit Status: APPROVED (31/31 passing tests)**