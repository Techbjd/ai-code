"""
VEGFR2 Self-Supervised Pre-Training Pipeline
=============================================
Pre-trains GNN models on unlabeled molecular data using:
1. Contrastive learning (SimCLR-style)
2. Masked atom prediction (BERT-style)

Usage:
    # Contrastive pre-training
    python scripts/pretrain.py --method contrastive --model gin --epochs 100

    # Masked atom prediction
    python scripts/pretrain.py --method masked --model gin --epochs 100

    # Compare both approaches
    python scripts/pretrain.py --method both --model gin --compare

    # Pre-train and then fine-tune on VEGFR2
    python scripts/pretrain.py --method contrastive --model gin --finetune

    # Use pre-trained model for downstream task
    python scripts/pretrain.py --load-pretrained checkpoints/pretrained_gin.pt --finetune
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        dev = torch.device("cpu")
        print("  GPU not available, using CPU")
    return dev


def load_smiles(raw_csv: str) -> list[str]:
    """Load SMILES from raw CSV."""
    import pandas as pd
    from vegfr2.data import load_csv, preprocess

    df = load_csv(raw_csv)
    df = preprocess(df)
    return df["smiles"].tolist()


def run_pretraining(
    method: str,
    model_name: str,
    smiles: list[str],
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    compare: bool = False,
) -> dict:
    """Run pre-training with specified method."""
    from vegfr2.pretrain import SelfSupervisedPretrainer

    results = {}

    methods = ["contrastive", "masked"] if compare else [method]

    for m in methods:
        print(f"\n{'=' * 60}")
        print(f"PRE-TRAINING: {m.upper()} with {model_name.upper()}")
        print(f"{'=' * 60}")

        start_time = time.time()

        pretrainer = SelfSupervisedPretrainer(
            model_name=model_name,
            method=m,
            hidden=cfg["gnn"]["hidden"],
            layers=cfg["gnn"]["layers"],
            heads=cfg["gnn"]["heads"],
            dropout=cfg["gnn"]["dropout"],
            lr=cfg["gnn"]["lr"],
            batch_size=cfg["gnn"]["batch"],
            seed=cfg["seed"],
        )

        history = pretrainer.pretrain(
            smiles_list=smiles,
            epochs=cfg["pretrain"]["epochs"],
            patience=cfg["pretrain"]["patience"],
            device=device,
            verbose=True,
        )

        elapsed = time.time() - start_time

        # Save pre-trained model
        save_path = output_dir / f"pretrained_{model_name}_{m}.pt"
        pretrainer.save_pretrained(save_path)
        print(f"\n  Saved to: {save_path}")

        results[m] = {
            "history": history,
            "time": elapsed,
            "final_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "save_path": str(save_path),
        }

        print(f"  Time: {elapsed:.1f}s")
        print(f"  Final loss: {results[m]['final_loss']:.4f}" if results[m]['final_loss'] else "")

    return results


def run_finetuning(
    pretrained_path: str,
    model_name: str,
    raw_csv: str,
    cfg: dict,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Fine-tune a pre-trained model on VEGFR2 task."""
    from vegfr2.pretrain import SelfSupervisedPretrainer
    from vegfr2.data import load_csv, preprocess, split
    from vegfr2.features import mol_to_graph_with_fps
    from vegfr2.gnn_pyg import build_pyg_model, save_checkpoint
    from vegfr2.metrics import classification_metrics
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    print(f"\n{'=' * 60}")
    print(f"FINE-TUNING: {model_name.upper()}")
    print(f"{'=' * 60}")

    # Load data
    df = load_csv(raw_csv)
    df = preprocess(df)
    train_df, val_df, test_df = split(df, seed=cfg["seed"])

    # Load pre-trained model
    pretrainer = SelfSupervisedPretrainer.load_pretrained(pretrained_path, device=device)

    # Get the base GNN from pre-trained model
    if hasattr(pretrainer.model, "gnn"):
        base_gnn = pretrainer.model.gnn
    else:
        base_gnn = pretrainer.model

    # Create classification head
    class FinetuneModel(torch.nn.Module):
        def __init__(self, gnn, hidden_dim, out_dim=1):
            super().__init__()
            self.gnn = gnn
            self.classifier = torch.nn.Linear(hidden_dim, out_dim)

        def forward(self, x, edge_index, batch):
            h = self.gnn(x, edge_index, batch)
            return self.classifier(h)

    model = FinetuneModel(base_gnn, cfg["gnn"]["hidden"]).to(device)

    # Prepare data
    def make_loader(smiles_list, labels, shuffle):
        data_list = []
        for s, y in zip(smiles_list, labels):
            g = mol_to_graph_with_fps(s, use_morgan=True, use_maccs=True)
            data = Data(
                x=g["node_feats"], edge_index=g["edge_index"],
                edge_attr=g["edge_feats"],
                y=torch.tensor([y], dtype=torch.float32),
            )
            data_list.append(data)
        return DataLoader(data_list, batch_size=cfg["gnn"]["batch"], shuffle=shuffle)

    train_loader = make_loader(
        train_df["smiles"].tolist(),
        train_df["active"].astype(int).tolist(),
        True,
    )
    val_loader = make_loader(
        val_df["smiles"].tolist(),
        val_df["active"].astype(int).tolist(),
        False,
    )
    test_loader = make_loader(
        test_df["smiles"].tolist(),
        test_df["active"].astype(int).tolist(),
        False,
    )

    # Training setup
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["gnn"]["lr"] * 0.1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["gnn"]["epochs"], eta_min=1e-6
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Training loop
    best_auc = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, cfg["gnn"]["epochs"] + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(logits.squeeze(), batch.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

        # Validation
        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                val_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
                val_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

        val_auc = classification_metrics(val_true, val_probs).get("auc") or 0.0

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg["gnn"]["patience"]:
                print(f"  Early stopping at epoch {epoch} (best AUC={best_auc:.4f})")
                break

        if epoch % 25 == 0:
            print(f"  Epoch {epoch:3d} | val_AUC={val_auc:.4f}")

    # Evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_probs, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            test_probs.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
            test_true.extend(batch.y.squeeze().cpu().numpy().astype(int))

    metrics = classification_metrics(test_true, test_probs)

    # Save checkpoint
    ckpt_dir = output_dir / f"finetuned_{model_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_dir / "best.pt")

    print(f"\n  Test AUC: {metrics.get('auc', 0):.4f}")
    print(f"  Test ACC: {metrics['acc']:.4f}")
    print(f"  Test MCC: {metrics['mcc']:.4f}")

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VEGFR2 Self-Supervised Pre-Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data options
    parser.add_argument(
        "--raw-csv", default="data/raw/chembl_vegfr2.csv",
        help="Raw CSV with SMILES (default: data/raw/chembl_vegfr2.csv)",
    )

    # Pre-training options
    parser.add_argument(
        "--method", choices=["contrastive", "masked", "both"],
        default="contrastive", help="Pre-training method",
    )
    parser.add_argument(
        "--model", default="gin",
        choices=["gcn", "gat", "gatv2", "mpnn", "gin", "pna", "graph_transformer"],
        help="GNN architecture",
    )
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Pre-training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")

    # Fine-tuning options
    parser.add_argument("--finetune", action="store_true", help="Fine-tune on VEGFR2 task after pre-training")
    parser.add_argument("--load-pretrained", help="Load pre-trained model for fine-tuning")

    # Comparison
    parser.add_argument("--compare", action="store_true", help="Compare contrastive vs masked")

    # Output
    parser.add_argument("--output-dir", default="runs/pretrain", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Config
    cfg = {
        "seed": args.seed,
        "gnn": {
            "hidden": args.hidden,
            "layers": args.layers,
            "heads": args.heads,
            "batch": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "dropout": args.dropout,
        },
        "pretrain": {
            "epochs": args.epochs,
            "patience": args.patience,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    print("\n" + "=" * 60)
    print("VEGFR2 SELF-SUPERVISED PRE-TRAINING")
    print("=" * 60)

    # Load SMILES
    print("\nLoading data...")
    smiles = load_smiles(args.raw_csv)
    print(f"  Loaded {len(smiles)} molecules")

    # Device
    print("\nSetting up device...")
    device = get_device()

    # Run pre-training or fine-tuning
    if args.load_pretrained:
        # Fine-tune existing pre-trained model
        metrics = run_finetuning(
            args.load_pretrained, args.model, args.raw_csv, cfg, device, output_dir
        )
        results = {"finetune": metrics}
    else:
        # Pre-train
        results = run_pretraining(
            args.method, args.model, smiles, cfg, device, output_dir,
            compare=args.compare,
        )

        # Fine-tune if requested
        if args.finetune:
            for method_name, result in results.items():
                if "save_path" in result:
                    finetune_metrics = run_finetuning(
                        result["save_path"], args.model, args.raw_csv, cfg, device, output_dir
                    )
                    results[method_name]["finetune_metrics"] = finetune_metrics

    # Save results
    results_path = output_dir / "pretrain_results.json"
    with open(results_path, "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to: {results_path}")
    print("\nDone!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
