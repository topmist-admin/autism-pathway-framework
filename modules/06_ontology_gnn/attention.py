"""
Biological Attention Mechanisms for Ontology-Aware GNN

Provides attention mechanisms that incorporate biological knowledge:
- Gene constraint scores (pLI, LOEUF)
- Pathway co-membership
- GO term semantic similarity
- Expression correlation
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using numpy fallback.")


if TORCH_AVAILABLE:

    class BiologicalAttention(nn.Module):
        """
        Multi-head attention with biological inductive biases.

        Incorporates prior knowledge into attention computation:
        1. Structural attention: Based on node features
        2. Biological attention: Modulated by constraint scores, expression
        3. Semantic attention: Based on GO term similarity

        Example:
            >>> attn = BiologicalAttention(
            ...     hidden_dim=128,
            ...     num_heads=8,
            ...     prior_types=["pli", "expression"],
            ... )
            >>> # Compute attention-weighted aggregation
            >>> out = attn(query, key, value, bio_priors={"pli": pli_scores})
        """

        def __init__(
            self,
            hidden_dim: int,
            num_heads: int = 8,
            dropout: float = 0.1,
            prior_types: Optional[List[str]] = None,
            use_bias_term: bool = True,
        ):
            """
            Initialize biological attention.

            Args:
                hidden_dim: Hidden dimension (must be divisible by num_heads)
                num_heads: Number of attention heads
                dropout: Dropout rate
                prior_types: List of biological priors to incorporate
                use_bias_term: Whether to add learned bias to attention
            """
            super().__init__()
            assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads
            self.prior_types = prior_types or []
            self.use_bias_term = use_bias_term

            # Standard QKV projections
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)

            # Biological prior projections (convert priors to attention biases)
            if self.prior_types:
                self.prior_proj = nn.ModuleDict({
                    ptype: nn.Sequential(
                        nn.Linear(1, num_heads),
                        nn.Tanh(),  # Bounded bias
                    )
                    for ptype in self.prior_types
                })

            # Optional learned bias term
            if use_bias_term:
                self.attn_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))

            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.head_dim)

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            bio_priors: Optional[Dict[str, torch.Tensor]] = None,
            edge_index: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute biological attention.

            Args:
                query: Query tensor [batch, seq_q, hidden_dim] or [num_nodes, hidden_dim]
                key: Key tensor [batch, seq_k, hidden_dim] or [num_nodes, hidden_dim]
                value: Value tensor [batch, seq_k, hidden_dim] or [num_nodes, hidden_dim]
                attention_mask: Optional mask [batch, seq_q, seq_k] or [num_edges]
                bio_priors: Dict of biological prior values for nodes
                edge_index: Optional edge indices for sparse attention [2, num_edges]

            Returns:
                Tuple of (output tensor, attention weights)
            """
            # Handle both batched and unbatched inputs
            is_batched = query.dim() == 3
            if not is_batched:
                query = query.unsqueeze(0)
                key = key.unsqueeze(0)
                value = value.unsqueeze(0)

            batch_size, seq_q, _ = query.shape
            seq_k = key.size(1)

            # Project to Q, K, V
            Q = self.q_proj(query)  # [batch, seq_q, hidden_dim]
            K = self.k_proj(key)    # [batch, seq_k, hidden_dim]
            V = self.v_proj(value)  # [batch, seq_k, hidden_dim]

            # Reshape for multi-head attention
            Q = Q.view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
            # Now: [batch, num_heads, seq, head_dim]

            # Compute attention scores
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            # [batch, num_heads, seq_q, seq_k]

            # Add biological prior biases
            if bio_priors and self.prior_types:
                bio_bias = self._compute_bio_bias(bio_priors, seq_k, query.device)
                if bio_bias is not None:
                    # bio_bias: [num_heads, 1, seq_k] -> broadcast to [batch, num_heads, seq_q, seq_k]
                    attn_scores = attn_scores + bio_bias.unsqueeze(0)

            # Add learned bias
            if self.use_bias_term:
                attn_scores = attn_scores + self.attn_bias

            # Apply attention mask
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    # [batch, seq_k] -> [batch, 1, 1, seq_k]
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                elif attention_mask.dim() == 3:
                    # [batch, seq_q, seq_k] -> [batch, 1, seq_q, seq_k]
                    attention_mask = attention_mask.unsqueeze(1)
                attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

            # Softmax and dropout
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Apply attention to values
            out = torch.matmul(attn_weights, V)
            # [batch, num_heads, seq_q, head_dim]

            # Reshape and project output
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_q, self.hidden_dim)
            out = self.out_proj(out)

            # Remove batch dim if input was unbatched
            if not is_batched:
                out = out.squeeze(0)
                attn_weights = attn_weights.squeeze(0)

            return out, attn_weights

        def _compute_bio_bias(
            self,
            bio_priors: Dict[str, torch.Tensor],
            seq_len: int,
            device: torch.device,
        ) -> Optional[torch.Tensor]:
            """Compute attention bias from biological priors."""
            biases = []

            for ptype in self.prior_types:
                if ptype in bio_priors:
                    prior_val = bio_priors[ptype]
                    if prior_val.dim() == 1:
                        prior_val = prior_val.unsqueeze(-1)  # [seq_len, 1]
                    # Project to per-head bias
                    bias = self.prior_proj[ptype](prior_val)  # [seq_len, num_heads]
                    bias = bias.transpose(0, 1).unsqueeze(1)  # [num_heads, 1, seq_len]
                    biases.append(bias)

            if not biases:
                return None

            # Combine biases (sum)
            return sum(biases)


    class EdgeTypeAttention(nn.Module):
        """
        Attention mechanism that considers edge types.

        Different edge types (PPI, pathway, GO) get different
        attention heads or attention patterns.
        """

        def __init__(
            self,
            hidden_dim: int,
            edge_types: List[str],
            heads_per_type: int = 2,
            dropout: float = 0.1,
        ):
            """
            Initialize edge-type-aware attention.

            Args:
                hidden_dim: Hidden dimension
                edge_types: List of edge type names
                heads_per_type: Number of attention heads per edge type
                dropout: Dropout rate
            """
            super().__init__()
            self.hidden_dim = hidden_dim
            self.edge_types = edge_types
            self.num_types = len(edge_types)
            self.heads_per_type = heads_per_type
            self.total_heads = self.num_types * heads_per_type

            assert hidden_dim % self.total_heads == 0, \
                f"hidden_dim ({hidden_dim}) must be divisible by total_heads ({self.total_heads})"

            self.head_dim = hidden_dim // self.total_heads

            # Type-specific QK projections
            self.type_q_proj = nn.ModuleDict({
                etype: nn.Linear(hidden_dim, heads_per_type * self.head_dim)
                for etype in edge_types
            })
            self.type_k_proj = nn.ModuleDict({
                etype: nn.Linear(hidden_dim, heads_per_type * self.head_dim)
                for etype in edge_types
            })

            # Shared value projection
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)

            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.head_dim)

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_type: torch.Tensor,
            edge_type_names: List[str],
        ) -> torch.Tensor:
            """
            Compute edge-type-aware attention.

            Args:
                x: Node features [num_nodes, hidden_dim]
                edge_index: Edge indices [2, num_edges]
                edge_type: Edge type indices [num_edges]
                edge_type_names: List mapping indices to type names

            Returns:
                Updated node features [num_nodes, hidden_dim]
            """
            num_nodes = x.size(0)
            src, dst = edge_index

            # Project values (shared across types)
            V = self.v_proj(x)  # [num_nodes, hidden_dim]

            # Initialize output
            out = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
            attn_sum = torch.zeros(num_nodes, device=x.device)

            # Process each edge type
            for i, etype in enumerate(edge_type_names):
                if etype not in self.type_q_proj:
                    continue

                # Get edges of this type
                mask = edge_type == i
                if not mask.any():
                    continue

                type_src = src[mask]
                type_dst = dst[mask]

                # Type-specific Q and K
                Q = self.type_q_proj[etype](x[type_dst])  # [num_type_edges, heads_per_type * head_dim]
                K = self.type_k_proj[etype](x[type_src])  # [num_type_edges, heads_per_type * head_dim]

                # Reshape for multi-head
                Q = Q.view(-1, self.heads_per_type, self.head_dim)
                K = K.view(-1, self.heads_per_type, self.head_dim)

                # Compute attention scores per edge
                attn_scores = (Q * K).sum(dim=-1) / self.scale  # [num_type_edges, heads_per_type]
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = self.dropout(attn_weights)

                # Get values and weight them
                V_src = V[type_src]  # [num_type_edges, hidden_dim]
                weighted_V = V_src * attn_weights.mean(dim=-1, keepdim=True)

                # Aggregate to destination nodes
                out.scatter_add_(0, type_dst.unsqueeze(-1).expand_as(weighted_V), weighted_V)
                attn_sum.scatter_add_(0, type_dst, torch.ones_like(type_dst, dtype=torch.float))

            # Normalize by number of incoming edges
            attn_sum = attn_sum.clamp(min=1).unsqueeze(-1)
            out = out / attn_sum

            # Output projection
            out = self.out_proj(out)

            return out


    class GOSemanticAttention(nn.Module):
        """
        Attention based on Gene Ontology semantic similarity.

        Uses information content or graph-based similarity
        between GO terms to modulate attention.
        """

        def __init__(
            self,
            hidden_dim: int,
            num_heads: int = 4,
            similarity_method: str = "resnik",
            dropout: float = 0.1,
        ):
            """
            Initialize GO semantic attention.

            Args:
                hidden_dim: Hidden dimension
                num_heads: Number of attention heads
                similarity_method: GO similarity method (resnik, lin, wang)
                dropout: Dropout rate
            """
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads
            self.similarity_method = similarity_method

            # Standard attention components
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)

            # Project GO similarity to attention bias
            self.sim_proj = nn.Sequential(
                nn.Linear(1, num_heads),
                nn.Tanh(),
            )

            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.head_dim)

        def forward(
            self,
            x: torch.Tensor,
            go_similarity: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute GO-aware attention.

            Args:
                x: Node features [num_nodes, hidden_dim]
                go_similarity: Pairwise GO similarity matrix [num_nodes, num_nodes]
                attention_mask: Optional attention mask

            Returns:
                Tuple of (output, attention weights)
            """
            num_nodes = x.size(0)

            # Standard QKV
            Q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)
            K = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)
            V = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)

            # Transpose for attention: [num_heads, num_nodes, head_dim]
            Q = Q.transpose(0, 1)
            K = K.transpose(0, 1)
            V = V.transpose(0, 1)

            # Compute attention scores
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            # [num_heads, num_nodes, num_nodes]

            # Add GO similarity bias
            if go_similarity is not None:
                # go_similarity: [num_nodes, num_nodes]
                sim_flat = go_similarity.view(-1, 1)  # [num_nodes^2, 1]
                sim_bias = self.sim_proj(sim_flat)  # [num_nodes^2, num_heads]
                sim_bias = sim_bias.view(num_nodes, num_nodes, self.num_heads)
                sim_bias = sim_bias.permute(2, 0, 1)  # [num_heads, num_nodes, num_nodes]
                attn_scores = attn_scores + sim_bias

            # Apply mask
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

            # Softmax
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Apply to values
            out = torch.matmul(attn_weights, V)  # [num_heads, num_nodes, head_dim]

            # Reshape output
            out = out.transpose(0, 1).contiguous().view(num_nodes, self.hidden_dim)
            out = self.out_proj(out)

            return out, attn_weights


    class PathwayCoAttention(nn.Module):
        """
        Co-attention between genes and pathways.

        Allows genes to attend to pathways they belong to,
        and pathways to attend to their constituent genes.
        """

        def __init__(
            self,
            gene_dim: int,
            pathway_dim: int,
            num_heads: int = 4,
            dropout: float = 0.1,
        ):
            """
            Initialize pathway co-attention.

            Args:
                gene_dim: Gene feature dimension
                pathway_dim: Pathway feature dimension
                num_heads: Number of attention heads
                dropout: Dropout rate
            """
            super().__init__()
            self.gene_dim = gene_dim
            self.pathway_dim = pathway_dim
            self.num_heads = num_heads

            # Use larger dimension for attention
            self.hidden_dim = max(gene_dim, pathway_dim)
            self.head_dim = self.hidden_dim // num_heads

            # Gene -> Pathway attention
            self.gene_q = nn.Linear(gene_dim, self.hidden_dim)
            self.pathway_k = nn.Linear(pathway_dim, self.hidden_dim)
            self.pathway_v = nn.Linear(pathway_dim, self.hidden_dim)
            self.gene_out = nn.Linear(self.hidden_dim, gene_dim)

            # Pathway -> Gene attention
            self.pathway_q = nn.Linear(pathway_dim, self.hidden_dim)
            self.gene_k = nn.Linear(gene_dim, self.hidden_dim)
            self.gene_v = nn.Linear(gene_dim, self.hidden_dim)
            self.pathway_out = nn.Linear(self.hidden_dim, pathway_dim)

            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.head_dim)

        def forward(
            self,
            gene_features: torch.Tensor,
            pathway_features: torch.Tensor,
            membership_matrix: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute co-attention between genes and pathways.

            Args:
                gene_features: Gene features [num_genes, gene_dim]
                pathway_features: Pathway features [num_pathways, pathway_dim]
                membership_matrix: Binary membership [num_genes, num_pathways]

            Returns:
                Tuple of (updated gene features, updated pathway features)
            """
            num_genes = gene_features.size(0)
            num_pathways = pathway_features.size(0)

            # Gene -> Pathway attention (genes query pathways)
            Q_g = self.gene_q(gene_features).view(num_genes, self.num_heads, self.head_dim)
            K_p = self.pathway_k(pathway_features).view(num_pathways, self.num_heads, self.head_dim)
            V_p = self.pathway_v(pathway_features).view(num_pathways, self.num_heads, self.head_dim)

            # [num_heads, num_genes, num_pathways]
            attn_g2p = torch.einsum('gnh,pnh->ngp', Q_g, K_p) / self.scale

            # Mask by membership
            membership_mask = membership_matrix.unsqueeze(0).expand(self.num_heads, -1, -1)
            attn_g2p = attn_g2p.masked_fill(membership_mask == 0, float('-inf'))

            attn_g2p = F.softmax(attn_g2p, dim=-1)
            attn_g2p = self.dropout(attn_g2p)

            # Aggregate pathway info to genes
            gene_ctx = torch.einsum('ngp,pnh->gnh', attn_g2p, V_p)
            gene_ctx = gene_ctx.contiguous().view(num_genes, self.hidden_dim)
            gene_update = self.gene_out(gene_ctx)

            # Pathway -> Gene attention (pathways query genes)
            Q_p = self.pathway_q(pathway_features).view(num_pathways, self.num_heads, self.head_dim)
            K_g = self.gene_k(gene_features).view(num_genes, self.num_heads, self.head_dim)
            V_g = self.gene_v(gene_features).view(num_genes, self.num_heads, self.head_dim)

            # [num_heads, num_pathways, num_genes]
            attn_p2g = torch.einsum('pnh,gnh->npg', Q_p, K_g) / self.scale

            # Mask by membership (transposed)
            membership_mask_t = membership_matrix.t().unsqueeze(0).expand(self.num_heads, -1, -1)
            attn_p2g = attn_p2g.masked_fill(membership_mask_t == 0, float('-inf'))

            attn_p2g = F.softmax(attn_p2g, dim=-1)
            attn_p2g = self.dropout(attn_p2g)

            # Aggregate gene info to pathways
            pathway_ctx = torch.einsum('npg,gnh->pnh', attn_p2g, V_g)
            pathway_ctx = pathway_ctx.contiguous().view(num_pathways, self.hidden_dim)
            pathway_update = self.pathway_out(pathway_ctx)

            return gene_update, pathway_update


else:
    # Numpy fallback implementations

    class BiologicalAttention:
        """Numpy fallback for BiologicalAttention."""

        def __init__(self, hidden_dim: int, num_heads: int = 8, **kwargs):
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads

        def forward(self, query, key, value, **kwargs):
            logger.warning("Using numpy fallback - limited functionality")
            # Simple mean aggregation as fallback
            out = np.mean(np.stack([query, key, value]), axis=0)
            attn = np.ones((query.shape[0], query.shape[0])) / query.shape[0]
            return out, attn


    class EdgeTypeAttention:
        """Numpy fallback for EdgeTypeAttention."""

        def __init__(self, hidden_dim: int, edge_types: List[str], **kwargs):
            self.hidden_dim = hidden_dim

        def forward(self, x, edge_index, edge_type, edge_type_names):
            logger.warning("Using numpy fallback - limited functionality")
            return x


    class GOSemanticAttention:
        """Numpy fallback for GOSemanticAttention."""

        def __init__(self, hidden_dim: int, **kwargs):
            self.hidden_dim = hidden_dim

        def forward(self, x, go_similarity=None, attention_mask=None):
            logger.warning("Using numpy fallback - limited functionality")
            attn = np.ones((x.shape[0], x.shape[0])) / x.shape[0]
            return x, attn


    class PathwayCoAttention:
        """Numpy fallback for PathwayCoAttention."""

        def __init__(self, gene_dim: int, pathway_dim: int, **kwargs):
            self.gene_dim = gene_dim
            self.pathway_dim = pathway_dim

        def forward(self, gene_features, pathway_features, membership_matrix):
            logger.warning("Using numpy fallback - limited functionality")
            return gene_features, pathway_features
