# PLAN.md — GPU-only PyTorch reproduction of the ML/GNN part of Hou et al. 2025 (VEGFR2 virtual screening, DOI 10.1080/14756366.2025.2518192)

## 1. Architecture

```
 ChEMBL279 VEGFR2 IC50 CSV (download_data.py)
        |
        v
 data.preprocess: valid SMILES (RDKit), numeric ic50_nM, deduplicate, label
 active = 1 if ic50 < 500 nM else 0
        |
        v
 data.split: stratified 8:1:1 (train / val / test, seed=42)
        |
        +-----------------------------+
        |                             |
        v                             v
 fingerprints                    molecular graphs
 Morgan r=2, nBits=2048          mol_to_graph (28-dim atom feats,
        |                        bidirectional edges, 5-dim bond feats)
        v                             |
 RF / SVM / XGBoost            GCN / GAT / MPNN (pure PyTorch,
 (scikit-learn / xgboost)      paper Eq.(1)-(4): message passing ->
        |                      neighbor aggregation -> mean scatter pooling -> MLP head)
        +-------------+---------------+
                      v
        test-set metrics: ACC / SEN / SPE / MCC / AUC
                      |
                      v
 screen.py: external SMILES library CSV -> predicted P(active) -> hits.csv ranked desc
```

One-paragraph flow: `scripts/download_data.py` fetches the ChEMBL279 target-279 (VEGFR2) IC50 dataset to `data/raw/chembl_vegfr2.csv` (with an optional local fallback path). `data.preprocess` validates SMILES with RDKit, coerces IC50 to numeric nanometers, deduplicates (conflicting IC50 per SMILES dropped entirely, identical duplicates kept once), and adds the binary `active` column using the <500 nM threshold. The frame is split stratified 8:1:1 into train/val/test. Fingerprint rows feed RF/SVM/XGBoost; RDKit-derived graphs feed three from-scratch message-passing networks (GCN/GAT/MPNN) implementing the paper's Eq.(1)-(4) with mean pooling over nodes and a BCE-with-logits objective. Best checkpoints (by val AUC/loss) are saved under `runs/<name>/best.pt`; final metrics are computed once on the held-out test set. `scripts/screen.py` reloads any trained model and ranks an external SMILES CSV by predicted probability of activity above `--threshold`.

**GPU-only rule:** `scripts/train.py` and `scripts/screen.py` begin by calling `vegfr2.device.get_device()`, which raises `RuntimeError('GPU required...')` when `torch.cuda.is_available()` is False — enforcing that training/screening never silently runs on CPU. Library functions in `src/vegfr2/*` accept an optional `device` argument defaulting to CPU, so pytest can exercise tiny synthetic graphs without a GPU. (CI machine has torch 2.13.0+cpu and no CUDA; dgl/torch_geometric are forbidden/not installed.)

## 2. File tree

```
README.md
requirements.txt
configs/config.yaml
src/vegfr2/__init__.py
src/vegfr2/device.py
src/vegfr2/data.py
src/vegfr2/features.py
src/vegfr2/metrics.py
src/vegfr2/ml_models.py
src/vegfr2/gnn_models.py
src/vegfr2/hpo.py
scripts/download_data.py
scripts/train.py
scripts/screen.py
tests/conftest.py
tests/test_data.py
tests/test_features.py
tests/test_metrics.py
tests/test_gnn_models.py
tests/test_ml_models.py
```

## 3. Interface contracts (exact signatures)

- `device.get_device() -> torch.device`  # raises RuntimeError('GPU required...') if no CUDA
- `data.load_csv(path) -> pd.DataFrame`  # expects columns smiles,ic50_nM
- `data.label_ic50(ic50_nm: float) -> int`  # <500 ->1 else 0
- `data.deduplicate(df) -> pd.DataFrame`  # multiple distinct ic50 per smiles -> drop all rows of that smiles; identical duplicates -> keep first
- `data.preprocess(df) -> pd.DataFrame`  # valid SMILES via RDKit MolFromSmiles not None, numeric ic50, then deduplicate, then label column 'active'
- `data.split(df, seed=42) -> (train_df, val_df, test_df)`  # stratified 8:1:1 via two sklearn train_test_split calls (first 90/10 test, then 8/1 of remainder => val_size=1/9)
- `features.smiles_to_morgan(smiles, radius=2, n_bits=2048) -> np.ndarray[uint8]`
- `features.mol_to_graph(smiles) -> dict(node_feats: FloatTensor[N,F_atom], edge_index: LongTensor[2,E] bidirectional, edge_feats: FloatTensor[E,F_bond], num_nodes: int)`
  - atom feats one-hot: symbol [C,N,O,S,P,F,Cl,Br,I,B,Si,Se,other](12) + degree<=6(7) + formal charge -1..1(3) + aromatic(1) + hybridization S/SP/SP2/SP3/other(5) = 28 dims
  - bond feats: single/double/triple/aromatic(4) + conjugated(1) = 5 dims
- `features.collate_graphs(list_of_graph_dicts, labels) -> dict` of stacked padded batch tensors incl. `node_batch` index vector for scatter mean pooling
- `metrics.classification_metrics(y_true, y_prob, threshold=0.5) -> dict(acc,sen,spe,mcc,auc,confusion_matrix={'tp','tn','fp','fn'})`  # auc None-safe if single class present
- `ml_models.train_ml_model(name:'rf'|'svm'|'xgb', X_train,y_train,X_val,y_val,seed=42) -> fitted estimator`
  - rf: n_estimators=300, random_state=seed
  - svm: SVC(probability=True, kernel='rbf', C=10, gamma='scale')
  - xgb: XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, tree_method='hist'); GPU device param handled by caller note
- `gnn_models.GCN(in_dim=28, hidden=64, layers=3, out_dim=1)`; `GAT(..., heads=4)`; `MPNN(...)` — all `nn.Module` with `forward(batch_dict, device) -> logits[batch_size]` using mean scatter pooling; `gnn_models.build_model(name, in_dim) -> module`
- `hpo.optimize_gnn(name, train_fn, n_trials=20)` lazy-imports optuna; raises `ImportError('pip install optuna')` if missing
- CLIs:
  - `download_data.py --out data/raw/chembl_vegfr2.csv [--fallback path]`
  - `train.py --config configs/config.yaml [--model gcn|gat|mpnn|rf|svm|xgb|all] [--hpo]`
  - `screen.py --model runs/<name>/best.pt --input library.csv --output hits.csv [--threshold 0.9]`
- `configs/config.yaml` keys:
  ```yaml
  seed: 42
  paths: {raw_csv, test_csv, output_dir}
  label: {threshold_nM: 500}
  split: {test_size: 0.1, val_frac_of_remaining: 0.111111}
  fingerprint: {radius: 2, n_bits: 2048}
  gnn: {hidden: 64, layers: 3, heads: 4, batch: 128, lr: 0.001, epochs: 200, patience: 15}
  hpo: {n_trials: 20}
  ```

## 4. Test plan (pytest, CPU-only synthetic)

`tests/conftest.py`: fixture `tiny_df` — 10 rows mixing valid SMILES (`CCO`, `c1ccccc1`, `CC(=O)Oc1ccccc1C(=O)O`) and invalid ones, plus duplicates with same and different IC50 values spanning both classes.

- `test_data.py`: `test_label_ic50_boundary` (499->1, 500->0); `test_dedup_identical_kept_once`; `test_dedup_conflicting_dropped`; `test_preprocess_drops_invalid`; `test_split_shapes_stratified` (seed deterministic).
- `test_features.py`: `test_morgan_shape_dtype` (2048, uint8, nonzero for CCO); `test_morgan_deterministic`; `test_mol_to_graph_dims` (ethanol node_feats dim==28, edge count==2*num_bonds symmetric, edge dim==5); `test_collate_pads_and_batches`.
- `test_metrics.py`: perfect predictions -> acc=spe=sen=mcc=auc=1.0; hand-computed confusion case (tp=3,tn=2,fp=1,fn=1 -> acc=.714, sen=.75, spe=.667, mcc~.4164, tolerance 1e-3); auc single-class returns None.
- `test_gnn_models.py`: build each model via `build_model`, forward tiny collated batch with `device='cpu'`, assert logits shape==(batch,1); `backward()` yields finite grads; save/load state_dict roundtrip equality.
- `test_ml_models.py`: fit rf+xgb on 40 synthetic fingerprint rows, check predict_proba shape and range [0,1]; svm tested with the contract SVC (probability=True, rbf) on the same 40 rows — fast enough, no kernel override.

All tests MUST pass without GPU and without internet.

## 5. Decisions & risks

- Fingerprint parameters (r=2, nBits=2048) assumed — the paper does not specify radius/bit length explicitly.
- GNNs implemented from scratch in pure PyTorch instead of DGL/PyG (environment constraint: neither installed, both forbidden).
- SVM uses probability=True (Platt scaling) which is slower but required for AUC/probability-based screening output.
- Class imbalance is mild (~51/49 active/inactive) so no resampling or class weighting is applied.
- Paper's exact hyperparameters are unpublished; standard values (hidden=64, layers=3, heads=4, lr=1e-3, epochs=200, patience=15, optuna n_trials=20) fixed above.
