#!/usr/bin/env python3
"""
VEGFR2 Activity Prediction - Colab Workflow Script
Run from VS Code terminal: python run_colab_workflow.py --help
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd=None, env=None):
    """Run command and stream output."""
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  No GPU detected. GNN training will be slow on CPU.")
            return False
    except ImportError:
        print("PyTorch not installed")
        return False


def install_deps():
    """Install dependencies."""
    cmds = [
        "pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "pip install -q numpy pandas pyyaml scikit-learn xgboost rdkit optuna",
        "pip install -e . -q",
    ]
    for cmd in cmds:
        run_cmd(cmd)


def download_data(output_path="data/raw/chembl_vegfr2.csv"):
    """Download ChEMBL VEGFR2 data."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    run_cmd(f"python scripts/download_data.py --out {output_path}")


def train_model(model_name, config="configs/config.yaml", hpo=False):
    """Train a single model."""
    cmd = f"python scripts/train.py --model {model_name} --config {config}"
    if hpo:
        cmd += " --hpo"
    run_cmd(cmd)


def train_all(config="configs/config.yaml", hpo=False):
    """Train all models."""
    cmd = f"python scripts/train.py --model all --config {config}"
    if hpo:
        cmd += " --hpo"
    run_cmd(cmd)


def show_results():
    """Display training results."""
    import json
    results_path = Path("runs/results.json")
    if not results_path.exists():
        print("No results found. Run training first.")
        return
    with open(results_path) as f:
        results = json.load(f)
    print(f"{'Model':<8} {'ACC':>6} {'SEN':>6} {'SPE':>6} {'MCC':>6} {'AUC':>6}")
    print("-" * 44)
    for name, m in results.items():
        auc_str = f"{m['auc']:.4f}" if m["auc"] is not None else "N/A"
        print(f"{name:<8} {m['acc']:.4f} {m['sen']:.4f} {m['spe']:.4f} {m['mcc']:.4f} {auc_str:>6}")


def prepare_library(library_path="data/screen_library.csv", smiles_list=None):
    """Create screening library CSV."""
    import pandas as pd
    if smiles_list is None:
        smiles_list = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CCO",                       # Ethanol
            "C1=CC=CC=C1",               # Benzene
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # Caffeine
            "C[C@H](O)CC1=CC=CC=C1",     # Chiral (R)
            "C[C@@H](O)CC1=CC=CC=C1",    # Chiral (S)
            "C/C=C/C",                   # E-alkene
            "C/C=C\\C",                  # Z-alkene
        ]
    df = pd.DataFrame({"smiles": smiles_list})
    Path(library_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(library_path, index=False)
    print(f"Created library with {len(df)} compounds at {library_path}")
    return library_path


def screen_model(model_path, input_csv, output_csv, threshold=0.5, batch_size=32):
    """Screen library with trained model."""
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    cmd = (f"python scripts/screen.py "
           f"--model {model_path} "
           f"--input {input_csv} "
           f"--output {output_csv} "
           f"--threshold {threshold} "
           f"--batch-size {batch_size}")
    run_cmd(cmd)
    
    # Show results
    import pandas as pd
    results = pd.read_csv(output_csv)
    print(results[['smiles', 'probability', 'hit']].to_string(index=False))


def screen_multiple(models, input_csv, threshold=0.5):
    """Screen with multiple models and create ensemble."""
    import pandas as pd
    import numpy as np
    
    library_df = pd.read_csv(input_csv)
    all_results = library_df[['smiles']].copy()
    
    for model_path, model_name in models:
        output_csv = f"results/temp_{model_name}.csv"
        batch_size = 32 if model_path.endswith('.pt') else 256
        try:
            screen_model(model_path, input_csv, output_csv, threshold, batch_size)
            res = pd.read_csv(output_csv)
            all_results[f"prob_{model_name}"] = res["probability"]
            all_results[f"hit_{model_name}"] = res["hit"]
            print(f"✅ {model_name} done")
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
    
    # Ensemble
    ensemble_cols = [c for c in all_results.columns if c.startswith('prob_')]
    if ensemble_cols:
        all_results['prob_ensemble'] = all_results[ensemble_cols].mean(axis=1)
        all_results['hit_ensemble'] = all_results['prob_ensemble'] >= threshold
        all_results = all_results.sort_values('prob_ensemble', ascending=False)
        
        Path("results").mkdir(exist_ok=True)
        all_results.to_csv("results/combined_screening.csv", index=False)
        print("\nEnsemble results:")
        display_cols = ['smiles', 'prob_ensemble', 'hit_ensemble'] + ensemble_cols
        print(all_results[display_cols].to_string(index=False))


def create_custom_config(output_path="configs/custom_config.yaml"):
    """Create custom training config."""
    import yaml
    custom_config = {
        "seed": 42,
        "paths": {"raw_csv": "data/raw/chembl_vegfr2.csv", "output_dir": "runs_custom"},
        "label": {"threshold_nM": 500},
        "split": {"test_size": 0.1, "val_frac_of_remaining": 0.111111},
        "fingerprint": {"radius": 2, "n_bits": 2048},
        "gnn": {
            "hidden": 128, "layers": 4, "heads": 8,
            "batch": 64, "lr": 0.0005, "epochs": 300, "patience": 20
        },
        "hpo": {"n_trials": 30}
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(custom_config, f)
    print(f"Custom config saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VEGFR2 Colab Workflow")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Setup
    subparsers.add_parser("setup", help="Install dependencies and check GPU")
    
    # Data
    data_parser = subparsers.add_parser("download", help="Download ChEMBL data")
    data_parser.add_argument("--out", default="data/raw/chembl_vegfr2.csv")
    
    # Training
    train_parser = subparsers.add_parser("train", help="Train model(s)")
    train_parser.add_argument("--model", choices=["rf", "svm", "xgb", "gcn", "gat", "mpnn", "all"], required=True)
    train_parser.add_argument("--config", default="configs/config.yaml")
    train_parser.add_argument("--hpo", action="store_true")
    
    # Results
    subparsers.add_parser("results", help="Show training results")
    
    # Screening
    screen_parser = subparsers.add_parser("screen", help="Screen library with model")
    screen_parser.add_argument("--model", required=True, help="Path to model (.pt or .pkl)")
    screen_parser.add_argument("--input", default="data/screen_library.csv")
    screen_parser.add_argument("--output", default="results/screen_results.csv")
    screen_parser.add_argument("--threshold", type=float, default=0.5)
    screen_parser.add_argument("--batch-size", type=int, default=32)
    
    # Multi-model screening
    multi_parser = subparsers.add_parser("screen-all", help="Screen with all trained models")
    multi_parser.add_argument("--input", default="data/screen_library.csv")
    multi_parser.add_argument("--threshold", type=float, default=0.5)
    
    # Library
    lib_parser = subparsers.add_parser("prepare-library", help="Create screening library")
    lib_parser.add_argument("--out", default="data/screen_library.csv")
    lib_parser.add_argument("--smiles", nargs="+", help="SMILES strings")
    
    # Custom config
    config_parser = subparsers.add_parser("custom-config", help="Create custom training config")
    config_parser.add_argument("--out", default="configs/custom_config.yaml")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        check_gpu()
        install_deps()
        print("\n✅ Setup complete!")
        
    elif args.command == "download":
        download_data(args.out)
        
    elif args.command == "train":
        if args.model == "all":
            train_all(args.config, args.hpo)
        else:
            train_model(args.model, args.config, args.hpo)
        show_results()
        
    elif args.command == "results":
        show_results()
        
    elif args.command == "screen":
        screen_model(args.model, args.input, args.output, args.threshold, args.batch_size)
        
    elif args.command == "screen-all":
        models = [
            ("runs/gcn/best.pt", "GCN"),
            ("runs/gat/best.pt", "GAT"),
            ("runs/mpnn/best.pt", "MPNN"),
            ("runs/xgb/model.pkl", "XGBoost"),
            ("runs/rf/model.pkl", "RandomForest"),
        ]
        # Filter existing
        existing = [(p, n) for p, n in models if Path(p).exists()]
        if not existing:
            print("No trained models found. Run training first.")
            return
        screen_multiple(existing, args.input, args.threshold)
        
    elif args.command == "prepare-library":
        smiles = args.smiles if args.smiles else None
        prepare_library(args.out, smiles)
        
    elif args.command == "custom-config":
        create_custom_config(args.out)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()