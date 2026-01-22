"""
GNN Layers for Ontology-Aware Graph Neural Networks

Provides message passing layers that are aware of:
- Different edge types (PPI, pathway membership, GO annotations)
- Biological hierarchies (GO term hierarchy, pathway containment)
- Node type heterogeneity (genes, pathways, GO terms)
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

    class EdgeTypeTransform(nn.Module):
        """
        Separate linear transformation for each edge type.

        In heterogeneous biological graphs, different edge types
        (e.g., PPI vs pathway membership) should have different
        transformation matrices.

        Example:
            >>> transform = EdgeTypeTransform(
            ...     in_dim=128,
            ...     out_dim=64,
            ...     edge_types=["gene_interacts", "gene_in_pathway", "gene_has_go"]
            ... )
            >>> # Apply to features based on edge type
            >>> h_transformed = transform(h, edge_type="gene_interacts")
        """

        def __init__(
            self,
            in_dim: int,
            out_dim: int,
            edge_types: List[str],
            bias: bool = True,
        ):
            """
            Initialize edge-type-specific transformations.

            Args:
                in_dim: Input feature dimension
                out_dim: Output feature dimension
                edge_types: List of edge type names
                bias: Whether to include bias terms
            """
            super().__init__()
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.edge_types = edge_types

            # Create separate linear layers for each edge type
            self.transforms = nn.ModuleDict({
                etype: nn.Linear(in_dim, out_dim, bias=bias)
                for etype in edge_types
            })

            # Default transform for unknown edge types
            self.default_transform = nn.Linear(in_dim, out_dim, bias=bias)

        def forward(
            self,
            x: torch.Tensor,
            edge_type: str,
        ) -> torch.Tensor:
            """
            Apply edge-type-specific transformation.

            Args:
                x: Input features [batch_size, in_dim]
                edge_type: Edge type name

            Returns:
                Transformed features [batch_size, out_dim]
            """
            if edge_type in self.transforms:
                return self.transforms[edge_type](x)
            else:
                logger.warning(f"Unknown edge type: {edge_type}, using default")
                return self.default_transform(x)

        def forward_all(
            self,
            x: torch.Tensor,
            edge_types: torch.Tensor,
            edge_type_names: List[str],
        ) -> torch.Tensor:
            """
            Apply transformations for all edge types in a batch.

            Args:
                x: Input features [num_edges, in_dim]
                edge_types: Edge type indices [num_edges]
                edge_type_names: List mapping indices to names

            Returns:
                Transformed features [num_edges, out_dim]
            """
            out = torch.zeros(x.size(0), self.out_dim, device=x.device)

            for i, etype_name in enumerate(edge_type_names):
                mask = edge_types == i
                if mask.any():
                    out[mask] = self.forward(x[mask], etype_name)

            return out


    class MessagePassingLayer(nn.Module):
        """
        Generic message passing layer for heterogeneous graphs.

        Implements the message-aggregate-update paradigm:
        1. Message: Compute messages from neighbors
        2. Aggregate: Combine messages (mean, sum, attention)
        3. Update: Update node representations

        Supports multiple edge types and biological priors.
        """

        def __init__(
            self,
            in_dim: int,
            out_dim: int,
            edge_types: List[str],
            aggregation: str = "mean",
            activation: str = "relu",
            dropout: float = 0.1,
            residual: bool = True,
        ):
            """
            Initialize message passing layer.

            Args:
                in_dim: Input feature dimension
                out_dim: Output feature dimension
                edge_types: List of edge type names
                aggregation: Aggregation method ("mean", "sum", "max")
                activation: Activation function ("relu", "gelu", "none")
                dropout: Dropout rate
                residual: Whether to use residual connections
            """
            super().__init__()
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.edge_types = edge_types
            self.aggregation = aggregation
            self.residual = residual and (in_dim == out_dim)

            # Edge-type-specific message transforms
            self.message_transform = EdgeTypeTransform(
                in_dim, out_dim, edge_types
            )

            # Update MLP
            self.update_mlp = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.ReLU() if activation == "relu" else nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim),
            )

            # Layer normalization
            self.layer_norm = nn.LayerNorm(out_dim)

            # Activation
            if activation == "relu":
                self.activation = nn.ReLU()
            elif activation == "gelu":
                self.activation = nn.GELU()
            else:
                self.activation = nn.Identity()

            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_type: torch.Tensor,
            edge_type_names: List[str],
            edge_weight: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Forward pass of message passing.

            Args:
                x: Node features [num_nodes, in_dim]
                edge_index: Edge indices [2, num_edges]
                edge_type: Edge type indices [num_edges]
                edge_type_names: List mapping indices to type names
                edge_weight: Optional edge weights [num_edges]

            Returns:
                Updated node features [num_nodes, out_dim]
            """
            num_nodes = x.size(0)
            src, dst = edge_index

            # Compute messages from source nodes
            src_features = x[src]  # [num_edges, in_dim]

            # Apply edge-type-specific transformation
            messages = self.message_transform.forward_all(
                src_features, edge_type, edge_type_names
            )  # [num_edges, out_dim]

            # Apply edge weights if provided
            if edge_weight is not None:
                messages = messages * edge_weight.unsqueeze(-1)

            # Aggregate messages to destination nodes
            aggregated = self._aggregate(messages, dst, num_nodes)

            # Update
            out = self.update_mlp(aggregated)

            # Residual connection
            if self.residual:
                out = out + x

            # Layer norm and activation
            out = self.layer_norm(out)
            out = self.activation(out)
            out = self.dropout(out)

            return out

        def _aggregate(
            self,
            messages: torch.Tensor,
            dst: torch.Tensor,
            num_nodes: int,
        ) -> torch.Tensor:
            """Aggregate messages to destination nodes."""
            out = torch.zeros(num_nodes, self.out_dim, device=messages.device)

            if self.aggregation == "sum":
                out.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
            elif self.aggregation == "mean":
                # Sum and count for mean
                out.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
                count = torch.zeros(num_nodes, device=messages.device)
                count.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
                count = count.clamp(min=1).unsqueeze(-1)
                out = out / count
            elif self.aggregation == "max":
                # Use scatter_reduce for max (PyTorch 1.12+)
                out.scatter_reduce_(
                    0, dst.unsqueeze(-1).expand_as(messages),
                    messages, reduce="amax", include_self=False
                )

            return out


    class HierarchicalAggregator(nn.Module):
        """
        Aggregate node features following ontology hierarchy.

        For GO terms or pathway hierarchies, this layer propagates
        information from child nodes to parent nodes, respecting
        the biological hierarchy structure.

        Example:
            GO:0007268 (synaptic transmission)
            ├── GO:0007269 (neurotransmitter secretion)
            └── GO:0098916 (anterograde synaptic signaling)

        Child features are aggregated into parent representations.
        """

        def __init__(
            self,
            hidden_dim: int,
            aggregation: str = "attention",
            num_heads: int = 4,
        ):
            """
            Initialize hierarchical aggregator.

            Args:
                hidden_dim: Hidden dimension
                aggregation: Aggregation method ("attention", "mean", "max")
                num_heads: Number of attention heads (if using attention)
            """
            super().__init__()
            self.hidden_dim = hidden_dim
            self.aggregation = aggregation
            self.num_heads = num_heads

            if aggregation == "attention":
                self.attention = nn.MultiheadAttention(
                    hidden_dim, num_heads, batch_first=True
                )

            # Transform for combining self and children
            self.combine = nn.Linear(hidden_dim * 2, hidden_dim)
            self.layer_norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            x: torch.Tensor,
            hierarchy_edges: torch.Tensor,
            num_levels: int = 3,
        ) -> torch.Tensor:
            """
            Aggregate features following hierarchy.

            Args:
                x: Node features [num_nodes, hidden_dim]
                hierarchy_edges: Parent-child edges [2, num_edges]
                    where hierarchy_edges[0] = child, hierarchy_edges[1] = parent
                num_levels: Number of hierarchy levels to propagate

            Returns:
                Updated features with hierarchical information
            """
            out = x.clone()

            for _ in range(num_levels):
                out = self._propagate_level(out, hierarchy_edges)

            return out

        def _propagate_level(
            self,
            x: torch.Tensor,
            hierarchy_edges: torch.Tensor,
        ) -> torch.Tensor:
            """Propagate one level of hierarchy."""
            num_nodes = x.size(0)
            child_idx, parent_idx = hierarchy_edges

            # Get child features
            child_features = x[child_idx]  # [num_edges, hidden_dim]

            # Aggregate children for each parent
            parent_agg = torch.zeros(num_nodes, self.hidden_dim, device=x.device)

            if self.aggregation == "mean":
                parent_agg.scatter_add_(
                    0, parent_idx.unsqueeze(-1).expand_as(child_features),
                    child_features
                )
                count = torch.zeros(num_nodes, device=x.device)
                count.scatter_add_(0, parent_idx, torch.ones_like(parent_idx, dtype=torch.float))
                count = count.clamp(min=1).unsqueeze(-1)
                parent_agg = parent_agg / count

            elif self.aggregation == "max":
                parent_agg.scatter_reduce_(
                    0, parent_idx.unsqueeze(-1).expand_as(child_features),
                    child_features, reduce="amax", include_self=False
                )

            elif self.aggregation == "attention":
                # Group children by parent and apply attention
                # Simplified: use mean for now, full attention would need batching
                parent_agg.scatter_add_(
                    0, parent_idx.unsqueeze(-1).expand_as(child_features),
                    child_features
                )
                count = torch.zeros(num_nodes, device=x.device)
                count.scatter_add_(0, parent_idx, torch.ones_like(parent_idx, dtype=torch.float))
                count = count.clamp(min=1).unsqueeze(-1)
                parent_agg = parent_agg / count

            # Combine self features with aggregated children
            combined = torch.cat([x, parent_agg], dim=-1)
            out = self.combine(combined)
            out = self.layer_norm(out + x)  # Residual

            return out


    class BioPriorWeighting(nn.Module):
        """
        Weight messages using biological prior knowledge.

        Uses prior knowledge like:
        - Gene constraint scores (pLI, LOEUF)
        - Expression levels
        - SFARI confidence scores

        To modulate message importance during aggregation.
        """

        def __init__(
            self,
            hidden_dim: int,
            prior_types: List[str],
            combination: str = "multiplicative",
        ):
            """
            Initialize biological prior weighting.

            Args:
                hidden_dim: Hidden dimension
                prior_types: List of prior types (e.g., ["pli", "expression", "sfari"])
                combination: How to combine priors ("multiplicative", "additive", "learned")
            """
            super().__init__()
            self.hidden_dim = hidden_dim
            self.prior_types = prior_types
            self.combination = combination

            # Transform each prior to a weight
            self.prior_transforms = nn.ModuleDict({
                ptype: nn.Sequential(
                    nn.Linear(1, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, 1),
                    nn.Sigmoid(),
                )
                for ptype in prior_types
            })

            if combination == "learned":
                self.combine_weights = nn.Linear(len(prior_types), 1)

        def forward(
            self,
            bio_priors: Dict[str, torch.Tensor],
        ) -> torch.Tensor:
            """
            Compute weights from biological priors.

            Args:
                bio_priors: Dict mapping prior type to values [num_nodes]

            Returns:
                Node weights [num_nodes, 1]
            """
            weights = []

            for ptype in self.prior_types:
                if ptype in bio_priors:
                    prior_val = bio_priors[ptype].unsqueeze(-1)  # [num_nodes, 1]
                    w = self.prior_transforms[ptype](prior_val)
                    weights.append(w)
                else:
                    # Default weight of 0.5 if prior not available
                    device = next(iter(bio_priors.values())).device if bio_priors else "cpu"
                    num_nodes = next(iter(bio_priors.values())).size(0) if bio_priors else 1
                    weights.append(torch.full((num_nodes, 1), 0.5, device=device))

            if not weights:
                return None

            weights = torch.cat(weights, dim=-1)  # [num_nodes, num_priors]

            if self.combination == "multiplicative":
                return weights.prod(dim=-1, keepdim=True)
            elif self.combination == "additive":
                return weights.mean(dim=-1, keepdim=True)
            elif self.combination == "learned":
                return torch.sigmoid(self.combine_weights(weights))

            return weights.mean(dim=-1, keepdim=True)


else:
    # Numpy fallback implementations for environments without PyTorch

    class EdgeTypeTransform:
        """Numpy fallback for EdgeTypeTransform."""

        def __init__(self, in_dim: int, out_dim: int, edge_types: List[str], bias: bool = True):
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.edge_types = edge_types
            self.weights = {
                etype: np.random.randn(in_dim, out_dim) / np.sqrt(in_dim)
                for etype in edge_types
            }
            self.biases = {
                etype: np.zeros(out_dim) for etype in edge_types
            } if bias else None

        def forward(self, x: np.ndarray, edge_type: str) -> np.ndarray:
            W = self.weights.get(edge_type, np.random.randn(self.in_dim, self.out_dim))
            out = x @ W
            if self.biases and edge_type in self.biases:
                out = out + self.biases[edge_type]
            return out


    class MessagePassingLayer:
        """Numpy fallback for MessagePassingLayer."""

        def __init__(self, in_dim: int, out_dim: int, edge_types: List[str], **kwargs):
            self.transform = EdgeTypeTransform(in_dim, out_dim, edge_types)
            self.out_dim = out_dim

        def forward(self, x, edge_index, edge_type, edge_type_names, edge_weight=None):
            logger.warning("Using numpy fallback - limited functionality")
            return np.random.randn(x.shape[0], self.out_dim)


    class HierarchicalAggregator:
        """Numpy fallback for HierarchicalAggregator."""

        def __init__(self, hidden_dim: int, **kwargs):
            self.hidden_dim = hidden_dim

        def forward(self, x, hierarchy_edges, num_levels=3):
            return x  # No-op in fallback


    class BioPriorWeighting:
        """Numpy fallback for BioPriorWeighting."""

        def __init__(self, hidden_dim: int, prior_types: List[str], **kwargs):
            self.prior_types = prior_types

        def forward(self, bio_priors):
            if not bio_priors:
                return None
            first_prior = next(iter(bio_priors.values()))
            return np.ones((first_prior.shape[0], 1)) * 0.5
