"""
Preprocess raw SMILES data into enriched graph datasets.

This script converts raw CSV (SMILES + IC50) into preprocessed PyTorch
tensors that can be directly loaded by any GNN model.

Output structure:
    data/processed/
        train/
            node_feats.pt      # [N_nodes, 2246] enriched node features
            edge_index.pt      # [2, E] edge connections
            edge_feats.pt      # [E, 11] bond features
            labels.pt          # [B] binary labels (0/1)
            node_batch.pt      # [N_nodes] graph index per node
            morgan_fps.pt      # [B, 2048] Morgan fingerprints
            maccs_fps.pt       # [B, 166] MACCS keys
            metadata.json      # Stats
        val/
            (same files)
        test/
            (same files)
        pipeline_metadata.json  # Global config

Usage:
    # Basic usage
    python scripts/preprocess_data.py

    # Custom paths
    python scripts/preprocess_data.py --input data/raw/my_data.csv --output data/my_processed

    # Custom fingerprint params
    python scripts/preprocess_data.py --morgan-bits 4096 --morgan-radius 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vegfr2.data_pipeline import VEGFR2Pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess SMILES data into enriched graph datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This converts raw SMILES → enriched graphs → PyTorch tensors.

Each molecule becomes:
  [atom_features(32) + Morgan(2048) + MACCS(166)] = 2246-dim per node

Output files:
  node_feats.pt   - Enriched node features for all molecules
  edge_index.pt   - Graph connectivity
  edge_feats.pt   - Bond features
  labels.pt       - Binary activity labels
  node_batch.pt   - Node-to-graph mapping
  morgan_fps.pt   - Morgan fingerprints (for ML models)
  maccs_fps.pt    - MACCS keys (for ML models)

Example:
  python scripts/preprocess_data.py --input data/raw/chembl_vegfr2.csv
  python scripts/preprocess_data.py --output data/my_data --morgan-bits 4096
        """,
    )
    parser.add_argument(
        "--input", "-i",
        default="data/raw/chembl_vegfr2.csv",
        help="Input CSV with 'smiles' and 'ic50_nM' columns",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument("--morgan-radius", type=int, default=2, help="Morgan fingerprint radius")
    parser.add_argument("--morgan-bits", type=int, default=2048, help="Morgan fingerprint bits")
    parser.add_argument("--maccs-bits", type=int, default=166, help="MACCS key bits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Please provide a CSV with 'smiles' and 'ic50_nM' columns.")
        return 1

    # Run pipeline
    pipeline = VEGFR2Pipeline(
        morgan_radius=args.morgan_radius,
        morgan_n_bits=args.morgan_bits,
        maccs_n_bits=args.maccs_bits,
        seed=args.seed,
    )

    results = pipeline.run(input_path, args.output)

    print("\nPipeline complete! You can now train models with:")
    print(f"  python scripts/train_all.py --raw-csv {args.input}")
    print()
    print("Or load directly in Python:")
    print("  from vegfr2.data_pipeline import VEGFR2Pipeline")
    print("  pipeline = VEGFR2Pipeline()")
    print(f"  train_data = pipeline.load_split('{args.output}', 'train')")

    return 0


if __name__ == "__main__":
    sys.exit(main())
