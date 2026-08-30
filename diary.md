# Development Diary

> Track all sessions, tasks, results, and problems in this VEGFR2 project.

---

## Session 1: Feature Combinations + Enriched Graphs (2026-08-24)

### Goal
Improve GNN performance by combining molecular fingerprints (Morgan, MACCS) with graph features.

### Tasks Completed
- [x] Added Morgan + MACCS + GNN embedding combinations
- [x] Discovered enriched graph approach: inject fingerprints into node features
- [x] GATv2 achieved ~0.91 AUC

### Results
- Pure GNN: AUC ~0.63 (failed)
- Enriched GNN: AUC ~0.91 (excellent)

### Key Decisions
- Every atom gets the full molecular fingerprint appended (2246-dim)
- This gives GNN fingerprint knowledge during message passing

### Problems Encountered
- GNN alone fails with small data (~10K molecules)
- Solution: inject domain knowledge via fingerprints

---

## Session 2: Advanced GNN Models + Sklearn API (2026-08-24)

### Goal
Make GNN models usable like traditional ML (fit/predict API) and add advanced architectures.

### Tasks Completed
- [x] Created GIN, PNA, Graph Transformer architectures
- [x] Created sklearn-compatible API (GNNClassifier, GNNRegressor, EnsembleClassifier)
- [x] Added Ensemble: GNN embeddings + XGBoost

### Results
- GIN: Most expressive MPNN (provably WL-1 optimal)
- PNA: Multi-aggregator (4 aggregators + 3 scalers)
- Graph Transformer: Global attention over all atoms

### Key Decisions
- GIN uses MLP aggregation + learnable epsilon + JK concatenation
- PNA uses 4 aggregators (mean, min, max, std) + 3 scalers
- Graph Transformer uses TransformerConv with edge bias

---

## Session 3: Data Pipeline + Universal Enrichment (2026-08-24)

### Goal
Make ALL GNN models always use enriched graphs (no opt-out).

### Tasks Completed
- [x] Created full data pipeline: SMILES → enriched graphs → PyTorch tensors
- [x] Made ALL GNN models ALWAYS use enriched graphs
- [x] Every model gets [atom(32) + Morgan(2048) + MACCS(166)] = 2246-dim input

### Results
- Pipeline saves preprocessed tensors to disk
- Any model can load ready-to-train data

### Key Decisions
- Preprocess once, train many approach
- No opt-out from enrichment

---

## Session 4: Refactoring + Bug Fixes + Code Review (2026-08-24)

### Goal
Clean up codebase, extract models into separate files, fix critical bugs.

### Tasks Completed
- [x] Extracted all 7 models into separate files under `src/vegfr2/models/`
- [x] Removed all Colab notebooks, scripts, and orphaned code
- [x] Fixed 4 critical bugs (dimension mismatches, undefined attributes)
- [x] Updated README with new structure and test results
- [x] 9 organized commits pushed to GitHub

### Results
- 100 tests passing, 0 failures
- Clean, structured codebase

### Problems Encountered
- GNNClassifier.predict_proba: squeeze() on single sample → fixed with ndim check
- GNNClassifier.load: wrong parameter name → fixed
- GIN/PNA dimension mismatches → fixed
- GraphTransformer edge_dim shape → fixed

---

## Session 5: Self-Supervised Pre-Training + Diary + Skill System (2026-08-30)

### Goal
Implement self-supervised pre-training (Step 5), create diary tracking, and add OpenCode skill for auto-updating summary.

### Tasks Completed
- [x] Created `src/vegfr2/pretrain_models.py` - Augmentations + contrastive/masked wrappers
- [x] Created `src/vegfr2/pretrain.py` - SelfSupervisedPretrainer unified API
- [x] Created `scripts/pretrain.py` - CLI for pre-training
- [x] Created `tests/test_pretrain.py` - Tests for pre-training module
- [x] Created `diary.md` - Session/task tracking journal
- [x] Created `.opencode/skills/update-summary/SKILL.md` - Auto-update conversation summary skill
- [x] Updated `CONVERSATION_SUMMARY.md` with new session info

### Results
- Contrastive learning: NT-Xent loss with 4 augmentation strategies
- Masked atom prediction: MSE loss on masked positions
- Both approaches wrap existing GNN models (GIN, PNA, etc.)
- Pre-trained models can be saved/loaded for fine-tuning

### Key Decisions
- Use SimCLR-style contrastive learning with graph augmentations
- Use BERT-style masked atom prediction (15% mask rate)
- Unified SelfSupervisedPretrainer API for both methods
- Pre-trained models save base GNN weights (without pre-training head)
- Fine-tuning uses 10x lower learning rate

### Problems Encountered
- `test_deterministic` failed — augmentor used same RNG instance (state advanced). Fixed by using two separate augmentor instances + seeding PyTorch RNG
- `test_contrastive_loss` — missing `F` import in test file. Fixed
- `test_masked_prediction_loss` — GNN pooled to graph-level but mask was per-atom. Rewrote `MaskedAtomGNN` to work at node-level with custom `_get_node_embeddings()`
- `test_pretrain_masked` — `masked_prediction_loss` returned `tensor(0.0)` without `requires_grad` when no nodes masked. Fixed with `requires_grad=True`
- `global_mean_pool` not imported in `pretrain.py` — added import
- Base GNN built with `out_dim=1` instead of `out_dim=hidden` — fixed in `_build_base_model`
- Subgraph augmentation changed node count breaking batch vectors — limited to node-count-preserving augmentations only

### Next Steps
- Run pre-training on actual ChEMBL data
- Fine-tune on VEGFR2 and measure AUC improvement
- Compare contrastive vs masked approach
- Update results in CONVERSATION_SUMMARY.md

---

## Session 6: (Next Session - TODO)

### Goal
(To be filled)

### Tasks Completed
- [ ] (To be filled)

### Results
(To be filled)

### Key Decisions
(To be filled)

### Problems Encountered
(To be filled)

### Next Steps
(To be filled)
