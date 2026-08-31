"""Fused GNN - Lightweight graph processing + fingerprint at graph level.

This approach:
1. GNN processes graph (32-dim atoms) → learns structural patterns
2. After pooling, CONCATENATE fingerprint (graph-level)
3. Final classifier uses BOTH graph patterns + fingerprint knowledge

Benefits:
- Lightweight (32-dim input, not 2246)
- Accurate (fingerprint adds molecular knowledge)
- Fast (no repeated FP per atom)
- Low memory (no enriched graphs)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool


class FusedGIN(nn.Module):
    """GIN with fingerprint fusion at graph level.
    
    Architecture:
        Node features (32-dim) → GIN layers → graph embedding (hidden)
        Fingerprint (2048+166) → linear projection → FP embedding (hidden)
        Concatenate [graph_embed, fp_embed] → classifier → output
    
    Args:
        in_dim: Node feature dimension (32 for graph-only)
        hidden: Hidden dimension
        layers: Number of GIN layers
        fp_dim: Fingerprint dimension (2214 for Morgan+MACCS)
        out_dim: Output dimension (1 for binary)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_dim: int = 32,
        hidden: int = 128,
        layers: int = 3,
        fp_dim: int = 2214,
        out_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden = hidden
        self.fp_dim = fp_dim
        
        # GIN backbone (lightweight, 32-dim input)
        self.node_emb = nn.Linear(in_dim, hidden)
        self.gin_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
            )
            self.gin_convs.append(GINConv(mlp, train_eps=True))
            self.bn.append(nn.BatchNorm1d(hidden))
        
        # Fingerprint projection (compress FP to hidden dim)
        self.fp_proj = nn.Sequential(
            nn.Linear(fp_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        
        # Graph projection (pool output → hidden dim)
        self.graph_proj = nn.Linear(hidden * 3, hidden)
        
        # Fusion classifier (graph_embed + fp_embed → output)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        
        self.dropout = dropout
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        fingerprint: Tensor,
    ) -> Tensor:
        """Forward pass.
        
        Args:
            x: Node features [N_nodes, 32]
            edge_index: Graph connectivity [2, E]
            batch: Batch assignment [N_nodes]
            fingerprint: Molecular fingerprints [B, fp_dim]
        
        Returns:
            logits: [B, 1]
        """
        # GIN backbone (graph structure)
        h = F.relu(self.node_emb(x))
        
        for i, (conv, bn) in enumerate(zip(self.gin_convs, self.bn)):
            h = conv(h, edge_index)
            h = bn(h)
            if i < len(self.gin_convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Pool to graph level
        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        g_add = global_add_pool(h, batch)
        g_emb = torch.cat([g_mean, g_max, g_add], dim=-1)  # [B, hidden*3]
        
        # Project to hidden dim
        g_emb = F.relu(self.graph_proj(g_emb))  # [B, hidden]
        
        # Fingerprint branch (molecular knowledge)
        fp_emb = self.fp_proj(fingerprint)  # [B, hidden]
        
        # Fuse: concatenate graph + fingerprint
        fused = torch.cat([g_emb, fp_emb], dim=-1)  # [B, hidden*2]
        
        # Classify
        return self.classifier(fused)  # [B, 1]
    
    def get_embeddings(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        fingerprint: Tensor,
    ) -> Tensor:
        """Get graph embeddings (before classifier)."""
        h = F.relu(self.node_emb(x))
        
        for i, (conv, bn) in enumerate(zip(self.gin_convs, self.bn)):
            h = conv(h, edge_index)
            h = bn(h)
            if i < len(self.gin_convs) - 1:
                h = F.relu(h)
        
        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        g_add = global_add_pool(h, batch)
        g_emb = torch.cat([g_mean, g_max, g_add], dim=-1)
        g_emb = F.relu(self.graph_proj(g_emb))
        
        fp_emb = self.fp_proj(fingerprint)
        
        return torch.cat([g_emb, fp_emb], dim=-1)


class FusedGAT(nn.Module):
    """GAT with fingerprint fusion at graph level."""
    
    def __init__(
        self,
        in_dim: int = 32,
        hidden: int = 128,
        layers: int = 3,
        heads: int = 4,
        fp_dim: int = 2214,
        out_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden = hidden
        self.fp_dim = fp_dim
        
        from torch_geometric.nn import GATConv
        
        self.node_emb = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(layers):
            self.convs.append(GATConv(hidden, hidden // heads, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden))
        
        self.fp_proj = nn.Sequential(
            nn.Linear(fp_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        
        self.graph_proj = nn.Linear(hidden * 3, hidden)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        
        self.dropout = dropout
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        fingerprint: Tensor,
    ) -> Tensor:
        h = F.relu(self.node_emb(x))
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = h + conv(h, edge_index)  # residual
            h = norm(h)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        g_add = global_add_pool(h, batch)
        g_emb = torch.cat([g_mean, g_max, g_add], dim=-1)
        g_emb = F.relu(self.graph_proj(g_emb))
        
        fp_emb = self.fp_proj(fingerprint)
        fused = torch.cat([g_emb, fp_emb], dim=-1)
        
        return self.classifier(fused)
    
    def get_embeddings(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        fingerprint: Tensor,
    ) -> Tensor:
        h = F.relu(self.node_emb(x))
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = h + conv(h, edge_index)
            h = norm(h)
            if i < len(self.convs) - 1:
                h = F.relu(h)
        
        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        g_add = global_add_pool(h, batch)
        g_emb = torch.cat([g_mean, g_max, g_add], dim=-1)
        g_emb = F.relu(self.graph_proj(g_emb))
        
        fp_emb = self.fp_proj(fingerprint)
        
        return torch.cat([g_emb, fp_emb], dim=-1)
