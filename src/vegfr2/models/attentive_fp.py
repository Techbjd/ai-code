"""AttentiveFP - Graph Attention with GRU for Molecular Property Prediction.

Faithful implementation following OpenDrugAI/AttentiveFP reference:
https://github.com/OpenDrugAI/AttentiveFP

Key innovations from the paper (Xiong et al., J. Med. Chem. 2020):
1. Graph Attention: learns which neighboring atoms matter most
2. GRU Update: controls information flow (remember/forget)
3. Attentive Readout: T-step GRU learns which atoms matter for prediction

Architecture:
    1. atom_fc: Linear(in_dim → hidden) + LeakyReLU
    2. neighbor_fc: Linear(in_dim + bond_dim → hidden) + LeakyReLU
    3. For each radius layer (R times):
       a. Compute attention: α_ij = softmax(a^T [h_i || neighbor_j])
       b. Context: ctx_i = Σ α_ij * W_attend(neighbor_j)
       c. GRU update: h_i = GRU(ctx_i, h_i)
    4. Initial mol feature: mol_h = Σ h_i (sum of all atoms)
    5. For each readout step (T times):
       a. Mol attention: β_i = softmax(a^T [mol_h || h_i])
       b. Mol context: mol_ctx = Σ β_i * W_mol(h_i)
       c. GRU update: mol_h = GRU(mol_ctx, mol_h)
    6. Output: Linear(hidden → out_dim)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool


class AttentiveFP(nn.Module):
    """AttentiveFP: Graph Attention + GRU + Attentive Readout.

    Faithful to OpenDrugAI/AttentiveFP reference implementation.

    Args:
        in_dim: Input node feature dimension
        hidden: Hidden dimension (fingerprint_dim in reference)
        layers: Number of graph attention layers (radius in reference)
        out_dim: Output dimension (1 for binary classification)
        dropout: Dropout rate
        num_timesteps: Number of attentive readout steps (T in reference)
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden: int = 200,
        layers: int = 3,
        out_dim: int = 1,
        dropout: float = 0.2,
        num_timesteps: int = 2,
    ):
        super().__init__()
        self.init_kwargs = {
            "in_dim": in_dim,
            "hidden": hidden,
            "layers": layers,
            "out_dim": out_dim,
            "dropout": dropout,
            "num_timesteps": num_timesteps,
        }
        self.hidden = hidden
        self.radius = layers
        self.T = num_timesteps
        self.dropout_rate = dropout

        # ---- Atom embedding (reference: atom_fc) ----
        self.atom_fc = nn.Linear(in_dim, hidden)

        # ---- Neighbor embedding (reference: neighbor_fc) ----
        # Concatenate atom features + bond features → hidden
        # This is the KEY difference from standard GAT: neighbor features
        # include both the neighbor atom AND the connecting bond
        self.neighbor_fc = nn.Linear(in_dim + 11, hidden)  # 11 = bond_feat_dim

        # ---- Graph attention layers (reference: align + attend per radius) ----
        self.align = nn.ModuleList()
        self.attend = nn.ModuleList()
        self.gru_layers = nn.ModuleList()

        for _ in range(self.radius):
            # Attention alignment: score = a^T [h_i || h_j]
            self.align.append(nn.Linear(hidden * 2, 1))
            # Feature transform for context
            self.attend.append(nn.Linear(hidden, hidden))
            # GRU update
            self.gru_layers.append(nn.GRUCell(hidden, hidden))

        # ---- Attentive readout (reference: mol_align + mol_attend) ----
        self.mol_align = nn.Linear(hidden * 2, 1)
        self.mol_attend = nn.Linear(hidden, hidden)
        self.mol_gru = nn.GRUCell(hidden, hidden)

        # ---- Output ----
        self.output = nn.Linear(hidden, out_dim)

        self.dropout = nn.Dropout(dropout)

    def _leaky_relu(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, negative_slope=0.2)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Forward pass following OpenDrugAI reference.

        Args:
            x: Node features [N_nodes, in_dim]
            edge_index: Graph connectivity [2, E]
            batch: Batch assignment [N_nodes]
            edge_attr: Bond features [E, bond_dim] (required for neighbor_fc)

        Returns:
            logits: [B, out_dim]
        """
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.shape[0]

        # ---- Step 1: Atom feature transformation (reference: atom_fc + LeakyReLU) ----
        atom_feature = self._leaky_relu(self.atom_fc(x))  # [N, hidden]

        # ---- Step 2: Neighbor feature (reference: neighbor_fc + LeakyReLU) ----
        # Concatenate neighbor atom features with bond features
        if edge_attr is not None:
            neighbor_input = torch.cat([x[dst], edge_attr], dim=-1)  # [E, in_dim + bond_dim]
        else:
            # Fallback: use zeros for bond features
            neighbor_input = torch.cat([
                x[dst],
                torch.zeros(x.shape[0], 11, device=x.device)[:x[dst].shape[0]]
            ], dim=-1) if x[dst].shape[0] > 0 else x[dst]
            # If no edge_attr, just use atom features repeated
            neighbor_input = x[dst]  # [E, in_dim]

        neighbor_feature = self._leaky_relu(self.neighbor_fc(neighbor_input))  # [E, hidden]

        # ---- Step 3: Graph attention layers (reference: radius times) ----
        for d in range(self.radius):
            # Compute attention scores: a^T [h_src || h_dst]
            h_src = atom_feature[src]  # [E, hidden]
            h_dst = atom_feature[dst]  # [E, hidden]

            # Attention alignment
            align_input = torch.cat([h_src, h_dst], dim=-1)  # [E, hidden*2]
            align_score = self._leaky_relu(self.align[d](self.dropout(align_input)))  # [E, 1]

            # Softmax per destination node (scatter_softmax)
            align_score = align_score.squeeze(-1)  # [E]
            scores_max = torch.zeros(num_nodes, device=x.device, dtype=torch.float32).scatter_reduce_(
                0, dst, align_score.float(), reduce="amax"
            )
            align_score = align_score - scores_max[dst]
            exp_scores = align_score.exp()
            scores_sum = torch.zeros(num_nodes, device=x.device, dtype=torch.float32).scatter_add_(0, dst, exp_scores.float())
            attention_weight = exp_scores / (scores_sum[dst] + 1e-8)  # [E]

            # Context: weighted sum of transformed neighbor features
            neighbor_transformed = self.attend[d](self.dropout(neighbor_feature))  # [E, hidden]
            context = neighbor_transformed * attention_weight.unsqueeze(-1)  # [E, hidden]
            context = torch.zeros(num_nodes, self.hidden, device=x.device, dtype=torch.float32).scatter_add_(
                0, dst.unsqueeze(1).expand_as(context), context.float()
            )  # [N, hidden]
            context = F.elu(context)

            # GRU update
            atom_feature = self.gru_layers[d](context, atom_feature)  # [N, hidden]

        # ---- Step 4: Initial molecule feature (reference: sum of atoms) ----
        activated_features = F.relu(atom_feature)
        mol_feature = global_add_pool(activated_features, batch)  # [B, hidden]

        # ---- Step 5: Attentive readout (reference: T steps with GRU) ----
        for t in range(self.T):
            # Expand mol feature to each atom
            mol_expand = mol_feature[batch]  # [N, hidden]

            # Compute atom importance relative to molecule
            mol_align_input = torch.cat([mol_expand, activated_features], dim=-1)  # [N, hidden*2]
            mol_align_score = self._leaky_relu(self.mol_align(self.dropout(mol_align_input)))  # [N, 1]
            mol_align_score = mol_align_score.squeeze(-1)  # [N]

            # Softmax per molecule (scatter_softmax over batch)
            batch_max = torch.zeros(num_nodes, device=x.device, dtype=torch.float32).scatter_reduce_(
                0, batch, mol_align_score.float(), reduce="amax"
            )
            mol_align_score = mol_align_score - batch_max[batch]
            mol_exp_scores = mol_align_score.exp()
            mol_scores_sum = torch.zeros(num_nodes, device=x.device, dtype=torch.float32).scatter_add_(
                0, batch, mol_exp_scores.float()
            )
            mol_attention_weight = mol_exp_scores / (mol_scores_sum[batch] + 1e-8)  # [N]

            # Context: weighted sum of transformed atom features
            mol_context = self.mol_attend(self.dropout(activated_features))  # [N, hidden]
            mol_context = mol_context * mol_attention_weight.unsqueeze(-1)  # [N, hidden]
            mol_context = global_add_pool(mol_context, batch)  # [B, hidden]
            mol_context = F.elu(mol_context)

            # GRU update
            mol_feature = self.mol_gru(mol_context, mol_feature)  # [B, hidden]

        # ---- Step 6: Output ----
        return self.output(mol_feature)  # [B, out_dim]

    def get_embeddings(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Get molecule embeddings before output layer."""
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.shape[0]

        atom_feature = F.relu(self.atom_fc(x))

        if edge_attr is not None:
            neighbor_input = torch.cat([x[dst], edge_attr], dim=-1)
        else:
            neighbor_input = x[dst]

        neighbor_feature = F.relu(self.neighbor_fc(neighbor_input))

        for d in range(self.radius):
            h_src = atom_feature[src]
            h_dst = atom_feature[dst]
            align_input = torch.cat([h_src, h_dst], dim=-1)
            align_score = F.leaky_relu(self.align[d](self.dropout(align_input)), negative_slope=0.2)
            align_score = align_score.squeeze(-1)
            scores_max = torch.zeros(num_nodes, device=x.device, dtype=torch.float32).scatter_reduce_(
                0, dst, align_score.float(), reduce="amax"
            )
            align_score = align_score - scores_max[dst]
            exp_scores = align_score.exp()
            scores_sum = torch.zeros(num_nodes, device=x.device, dtype=torch.float32).scatter_add_(0, dst, exp_scores.float())
            attention_weight = exp_scores / (scores_sum[dst] + 1e-8)

            neighbor_transformed = self.attend[d](self.dropout(neighbor_feature))
            context = neighbor_transformed * attention_weight.unsqueeze(-1)
            context = torch.zeros(num_nodes, self.hidden, device=x.device, dtype=torch.float32).scatter_add_(
                0, dst.unsqueeze(1).expand_as(context), context.float()
            )
            context = F.elu(context)
            atom_feature = self.gru_layers[d](context, atom_feature)

        activated_features = F.relu(atom_feature)
        mol_feature = global_add_pool(activated_features, batch)

        for t in range(self.T):
            mol_expand = mol_feature[batch]
            mol_align_input = torch.cat([mol_expand, activated_features], dim=-1)
            mol_align_score = F.leaky_relu(self.mol_align(self.dropout(mol_align_input)), negative_slope=0.2)
            mol_align_score = mol_align_score.squeeze(-1)
            batch_max = torch.zeros(num_nodes, device=x.device, dtype=torch.float32).scatter_reduce_(
                0, batch, mol_align_score.float(), reduce="amax"
            )
            mol_align_score = mol_align_score - batch_max[batch]
            mol_exp_scores = mol_align_score.exp()
            mol_scores_sum = torch.zeros(num_nodes, device=x.device, dtype=torch.float32).scatter_add_(
                0, batch, mol_exp_scores.float()
            )
            mol_attention_weight = mol_exp_scores / (mol_scores_sum[batch] + 1e-8)

            mol_context = self.mol_attend(self.dropout(activated_features))
            mol_context = mol_context * mol_attention_weight.unsqueeze(-1)
            mol_context = global_add_pool(mol_context, batch)
            mol_context = F.elu(mol_context)
            mol_feature = self.mol_gru(mol_context, mol_feature)

        return mol_feature
