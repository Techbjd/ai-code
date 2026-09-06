"""VEGFR2 GNN model registry."""

from functools import partial

from vegfr2.models.gcn import GCN_PyG
from vegfr2.models.gat import GAT_PyG
from vegfr2.models.gatv2 import GATv2_PyG
from vegfr2.models.mpnn import MPNN_PyG
from vegfr2.models.gin import GIN_PyG
from vegfr2.models.pna import PNA_PyG
from vegfr2.models.graph_transformer import GraphTransformer_PyG
from vegfr2.models.attentive_fp import AttentiveFP
from vegfr2.models.fused_gnn import FusedGIN, FusedGAT
from vegfr2.models.fused_variants import FusedVariant

__all__ = [
    "GCN_PyG",
    "GAT_PyG",
    "GATv2_PyG",
    "MPNN_PyG",
    "GIN_PyG",
    "PNA_PyG",
    "GraphTransformer_PyG",
    "AttentiveFP",
    "FusedGIN",
    "FusedGAT",
    "FusedVariant",
]

MODEL_REGISTRY: dict[str, type] = {
    "gcn": GCN_PyG,
    "gat": GAT_PyG,
    "gatv2": GATv2_PyG,
    "mpnn": MPNN_PyG,
    "gin": GIN_PyG,
    "pna": PNA_PyG,
    "graph_transformer": GraphTransformer_PyG,
    "attentive_fp": AttentiveFP,
}

_VARIANT_GNNS = ["gcn", "gat", "gatv2", "gin", "mpnn", "attentive_fp"]
_VARIANT_FP_TYPES = {
    "graph_only": 0,
    "morgan": 2048,
}

for _gnn in _VARIANT_GNNS:
    for _fp_name, _fp_dim in _VARIANT_FP_TYPES.items():
        _key = f"{_gnn}_{_fp_name}"
        if _fp_name == "graph_only":
            MODEL_REGISTRY[_key] = partial(FusedVariant, gnn_type=_gnn, fp_type="none", fp_dim=0)
        else:
            MODEL_REGISTRY[_key] = partial(FusedVariant, gnn_type=_gnn, fp_type=_fp_name, fp_dim=_fp_dim)
