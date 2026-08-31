"""VEGFR2 GNN model registry.

Every model lives in its own file under ``vegfr2.models``.
Import all models here for convenient access.
"""

from vegfr2.models.gcn import GCN_PyG
from vegfr2.models.gat import GAT_PyG
from vegfr2.models.gatv2 import GATv2_PyG
from vegfr2.models.mpnn import MPNN_PyG
from vegfr2.models.gin import GIN_PyG
from vegfr2.models.pna import PNA_PyG
from vegfr2.models.graph_transformer import GraphTransformer_PyG
from vegfr2.models.ensemble import GNNEnsembleClassifier
from vegfr2.models.fused_gnn import FusedGIN, FusedGAT

__all__ = [
    "GCN_PyG",
    "GAT_PyG",
    "GATv2_PyG",
    "MPNN_PyG",
    "GIN_PyG",
    "PNA_PyG",
    "GraphTransformer_PyG",
    "GNNEnsembleClassifier",
    "FusedGIN",
    "FusedGAT",
]

MODEL_REGISTRY: dict[str, type] = {
    "gcn": GCN_PyG,
    "gat": GAT_PyG,
    "gatv2": GATv2_PyG,
    "mpnn": MPNN_PyG,
    "gin": GIN_PyG,
    "pna": PNA_PyG,
    "graph_transformer": GraphTransformer_PyG,
}
