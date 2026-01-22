"""
Ontology-Aware Graph Neural Network Model

Main GNN architecture that integrates:
- Heterogeneous message passing (different edge types)
- Hierarchical aggregation (GO term hierarchy)
- Biological attention (constraint scores, expression)
- Multi-task learning (gene classification, link prediction)

Designed for autism gene discovery and pathway analysis.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
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

# Import local modules conditionally
if TORCH_AVAILABLE:
    try:
        from .layers import (
            EdgeTypeTransform,
            MessagePassingLayer,
            HierarchicalAggregator,
            BioPriorWeighting,
        )
        from .attention import (
            BiologicalAttention,
            EdgeTypeAttention,
            GOSemanticAttention,
            PathwayCoAttention,
        )
    except ImportError:
        from layers import (
            EdgeTypeTransform,
            MessagePassingLayer,
            HierarchicalAggregator,
            BioPriorWeighting,
        )
        from attention import (
            BiologicalAttention,
            EdgeTypeAttention,
            GOSemanticAttention,
            PathwayCoAttention,
        )


@dataclass
class GNNConfig:
    """Configuration for OntologyAwareGNN."""

    # Architecture
    input_dim: int = 256
    hidden_dim: int = 256
    output_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8

    # Edge types
    edge_types: List[str] = field(default_factory=lambda: [
        "gene_interacts",      # PPI edges
        "gene_in_pathway",     # Gene-pathway membership
        "gene_has_go",         # Gene-GO term annotation
        "pathway_contains",    # Pathway hierarchy
        "go_is_a",             # GO term hierarchy
    ])

    # Node types
    node_types: List[str] = field(default_factory=lambda: [
        "gene",
        "pathway",
        "go_term",
    ])

    # Biological priors
    prior_types: List[str] = field(default_factory=lambda: [
        "pli",
        "loeuf",
        "expression",
        "sfari_score",
    ])

    # Training
    dropout: float = 0.1
    use_residual: bool = True
    use_layer_norm: bool = True
    aggregation: str = "attention"  # "mean", "sum", "attention"

    # Hierarchy
    num_hierarchy_levels: int = 3

    # Tasks
    task_heads: List[str] = field(default_factory=lambda: [
        "gene_classification",
        "link_prediction",
    ])


@dataclass
class GNNOutput:
    """Output from OntologyAwareGNN."""

    # Node embeddings by type
    node_embeddings: Dict[str, Any]  # {node_type: embeddings}

    # Task-specific outputs
    gene_logits: Optional[Any] = None
    link_scores: Optional[Any] = None

    # Attention weights for interpretability
    attention_weights: Optional[Dict[str, Any]] = None

    # Loss components
    loss: Optional[float] = None
    loss_components: Optional[Dict[str, float]] = None


if TORCH_AVAILABLE:

    class NodeTypeEmbedding(nn.Module):
        """Learnable embeddings for different node types."""

        def __init__(self, node_types: List[str], embedding_dim: int):
            super().__init__()
            self.node_types = node_types
            self.embeddings = nn.Embedding(len(node_types), embedding_dim)
            self.type_to_idx = {t: i for i, t in enumerate(node_types)}

        def forward(self, node_type: str) -> torch.Tensor:
            idx = self.type_to_idx.get(node_type, 0)
            return self.embeddings(torch.tensor([idx], device=self.embeddings.weight.device))


    class InputProjection(nn.Module):
        """Project heterogeneous input features to common dimension."""

        def __init__(
            self,
            input_dims: Dict[str, int],
            output_dim: int,
            node_types: List[str],
        ):
            super().__init__()
            self.output_dim = output_dim

            # Type-specific projections
            self.projections = nn.ModuleDict({
                ntype: nn.Linear(input_dims.get(ntype, output_dim), output_dim)
                for ntype in node_types
            })

            # Default projection for unknown types
            self.default_proj = nn.Linear(output_dim, output_dim)

        def forward(
            self,
            features: Dict[str, torch.Tensor],
        ) -> Dict[str, torch.Tensor]:
            """Project all node features."""
            projected = {}
            for ntype, feat in features.items():
                if ntype in self.projections:
                    projected[ntype] = self.projections[ntype](feat)
                else:
                    # Pad or truncate to output_dim
                    if feat.size(-1) < self.output_dim:
                        padding = torch.zeros(
                            *feat.shape[:-1], self.output_dim - feat.size(-1),
                            device=feat.device
                        )
                        feat = torch.cat([feat, padding], dim=-1)
                    elif feat.size(-1) > self.output_dim:
                        feat = feat[..., :self.output_dim]
                    projected[ntype] = self.default_proj(feat)
            return projected


    class OntologyAwareGNN(nn.Module):
        """
        Ontology-Aware Graph Neural Network for biological graph analysis.

        Architecture:
        1. Input projection (heterogeneous features -> common dimension)
        2. N message passing layers with edge-type awareness
        3. Hierarchical aggregation for ontology structures
        4. Biological attention with prior knowledge
        5. Task-specific heads (classification, link prediction)

        Example:
            >>> config = GNNConfig(hidden_dim=256, num_layers=3)
            >>> model = OntologyAwareGNN(config)
            >>> output = model(
            ...     node_features={"gene": gene_feat, "pathway": pathway_feat},
            ...     edge_index=edges,
            ...     edge_type=edge_types,
            ...     bio_priors={"pli": pli_scores},
            ... )
        """

        def __init__(
            self,
            config: GNNConfig,
            input_dims: Optional[Dict[str, int]] = None,
        ):
            """
            Initialize OntologyAwareGNN.

            Args:
                config: Model configuration
                input_dims: Dict mapping node types to their input dimensions
            """
            super().__init__()
            self.config = config

            # Default input dims
            if input_dims is None:
                input_dims = {ntype: config.input_dim for ntype in config.node_types}

            # Input projection
            self.input_proj = InputProjection(
                input_dims, config.hidden_dim, config.node_types
            )

            # Node type embeddings (added to projected features)
            self.node_type_emb = NodeTypeEmbedding(
                config.node_types, config.hidden_dim
            )

            # Message passing layers
            self.mp_layers = nn.ModuleList([
                MessagePassingLayer(
                    in_dim=config.hidden_dim,
                    out_dim=config.hidden_dim,
                    edge_types=config.edge_types,
                    aggregation="mean",  # Use mean for stability
                    dropout=config.dropout,
                    residual=config.use_residual,
                )
                for _ in range(config.num_layers)
            ])

            # Biological attention
            self.bio_attention = BiologicalAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                prior_types=config.prior_types,
                dropout=config.dropout,
            )

            # Hierarchical aggregator for GO terms
            self.hierarchy_agg = HierarchicalAggregator(
                hidden_dim=config.hidden_dim,
                aggregation="attention",
                num_heads=config.num_heads // 2,
            )

            # Biological prior weighting
            self.bio_weighting = BioPriorWeighting(
                hidden_dim=config.hidden_dim,
                prior_types=config.prior_types,
                combination="learned",
            )

            # Output projection
            self.output_proj = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.output_dim),
            )

            # Task heads
            self.task_heads = nn.ModuleDict()

            if "gene_classification" in config.task_heads:
                self.task_heads["gene_classification"] = nn.Sequential(
                    nn.Linear(config.output_dim, config.output_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.output_dim // 2, 2),  # Binary: ASD gene or not
                )

            if "link_prediction" in config.task_heads:
                self.task_heads["link_prediction"] = nn.Bilinear(
                    config.output_dim, config.output_dim, 1
                )

            # Layer norm
            if config.use_layer_norm:
                self.final_norm = nn.LayerNorm(config.output_dim)
            else:
                self.final_norm = nn.Identity()

        def forward(
            self,
            node_features: Dict[str, torch.Tensor],
            edge_index: torch.Tensor,
            edge_type: torch.Tensor,
            edge_type_names: Optional[List[str]] = None,
            node_type_indices: Optional[Dict[str, torch.Tensor]] = None,
            bio_priors: Optional[Dict[str, torch.Tensor]] = None,
            hierarchy_edges: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            link_labels: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ) -> GNNOutput:
            """
            Forward pass of OntologyAwareGNN.

            Args:
                node_features: Dict of {node_type: features [num_nodes_of_type, input_dim]}
                edge_index: Edge indices [2, num_edges]
                edge_type: Edge type indices [num_edges]
                edge_type_names: List mapping type indices to names
                node_type_indices: Dict mapping node type to node indices
                bio_priors: Dict of biological prior values
                hierarchy_edges: Hierarchy edges for GO aggregation [2, num_hier_edges]
                labels: Node labels for classification
                link_labels: (edge_index, labels) for link prediction

            Returns:
                GNNOutput with embeddings and task outputs
            """
            edge_type_names = edge_type_names or self.config.edge_types

            # 1. Project input features to common dimension
            projected = self.input_proj(node_features)

            # Combine all node features (assumes global node indexing)
            # In practice, you'd handle this based on your graph structure
            all_features = self._combine_features(projected, node_type_indices)

            # 2. Add node type embeddings
            if node_type_indices:
                for ntype, indices in node_type_indices.items():
                    type_emb = self.node_type_emb(ntype).expand(len(indices), -1)
                    all_features[indices] = all_features[indices] + type_emb

            # 3. Message passing layers
            h = all_features
            for mp_layer in self.mp_layers:
                h = mp_layer(h, edge_index, edge_type, edge_type_names)

            # 4. Hierarchical aggregation (if hierarchy provided)
            if hierarchy_edges is not None and hierarchy_edges.numel() > 0:
                h = self.hierarchy_agg(h, hierarchy_edges, self.config.num_hierarchy_levels)

            # 5. Biological attention (global refinement)
            h, attn_weights = self.bio_attention(
                h, h, h,
                bio_priors=bio_priors,
            )

            # 6. Apply biological prior weighting
            if bio_priors:
                weights = self.bio_weighting(bio_priors)
                if weights is not None:
                    h = h * weights

            # 7. Output projection
            out = self.output_proj(h)
            out = self.final_norm(out)

            # 8. Split embeddings by node type
            node_embeddings = self._split_embeddings(out, node_type_indices)

            # 9. Task-specific heads
            gene_logits = None
            link_scores = None
            loss = None
            loss_components = {}

            if "gene_classification" in self.task_heads:
                gene_idx = node_type_indices.get("gene") if node_type_indices else None
                if gene_idx is not None:
                    gene_out = out[gene_idx]
                    gene_logits = self.task_heads["gene_classification"](gene_out)

                    if labels is not None:
                        cls_loss = F.cross_entropy(gene_logits, labels)
                        loss_components["classification"] = cls_loss.item()
                        loss = cls_loss

            if "link_prediction" in self.task_heads and link_labels is not None:
                link_edges, link_y = link_labels
                src_emb = out[link_edges[0]]
                dst_emb = out[link_edges[1]]
                link_scores = self.task_heads["link_prediction"](src_emb, dst_emb).squeeze()

                link_loss = F.binary_cross_entropy_with_logits(link_scores, link_y.float())
                loss_components["link_prediction"] = link_loss.item()
                if loss is None:
                    loss = link_loss
                else:
                    loss = loss + link_loss

            return GNNOutput(
                node_embeddings=node_embeddings,
                gene_logits=gene_logits,
                link_scores=link_scores,
                attention_weights={"biological": attn_weights},
                loss=loss.item() if loss is not None else None,
                loss_components=loss_components,
            )

        def _combine_features(
            self,
            projected: Dict[str, torch.Tensor],
            node_type_indices: Optional[Dict[str, torch.Tensor]],
        ) -> torch.Tensor:
            """Combine projected features into single tensor."""
            if not node_type_indices:
                # Assume single node type, just concatenate
                return torch.cat(list(projected.values()), dim=0)

            # Calculate total nodes
            max_idx = max(idx.max().item() for idx in node_type_indices.values()) + 1
            device = next(iter(projected.values())).device

            # Initialize combined tensor
            combined = torch.zeros(max_idx, self.config.hidden_dim, device=device)

            # Place features at correct indices
            for ntype, indices in node_type_indices.items():
                if ntype in projected:
                    combined[indices] = projected[ntype]

            return combined

        def _split_embeddings(
            self,
            embeddings: torch.Tensor,
            node_type_indices: Optional[Dict[str, torch.Tensor]],
        ) -> Dict[str, torch.Tensor]:
            """Split combined embeddings by node type."""
            if not node_type_indices:
                return {"all": embeddings}

            result = {}
            for ntype, indices in node_type_indices.items():
                result[ntype] = embeddings[indices]
            return result

        def get_gene_embeddings(self, output: GNNOutput) -> torch.Tensor:
            """Extract gene embeddings from output."""
            return output.node_embeddings.get("gene", output.node_embeddings.get("all"))

        def encode(
            self,
            node_features: Dict[str, torch.Tensor],
            edge_index: torch.Tensor,
            edge_type: torch.Tensor,
            **kwargs,
        ) -> torch.Tensor:
            """Convenience method to get embeddings without task heads."""
            output = self.forward(node_features, edge_index, edge_type, **kwargs)
            return self._combine_features(output.node_embeddings, None)


    class GNNTrainer:
        """Trainer for OntologyAwareGNN."""

        def __init__(
            self,
            model: OntologyAwareGNN,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5,
            device: str = "cpu",
        ):
            """
            Initialize trainer.

            Args:
                model: The GNN model
                learning_rate: Learning rate
                weight_decay: L2 regularization
                device: Device to train on
            """
            self.model = model.to(device)
            self.device = device
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10
            )

        def train_step(
            self,
            node_features: Dict[str, torch.Tensor],
            edge_index: torch.Tensor,
            edge_type: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> Dict[str, float]:
            """Single training step."""
            self.model.train()
            self.optimizer.zero_grad()

            # Move to device
            node_features = {k: v.to(self.device) for k, v in node_features.items()}
            edge_index = edge_index.to(self.device)
            edge_type = edge_type.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)

            # Forward pass
            output = self.model(
                node_features, edge_index, edge_type,
                labels=labels, **kwargs
            )

            # Backward pass
            if output.loss is not None:
                loss_tensor = torch.tensor(output.loss, requires_grad=True)
                # Actually need to recompute for gradients
                output = self.model(
                    node_features, edge_index, edge_type,
                    labels=labels, **kwargs
                )
                # Get actual loss tensor for backward
                gene_idx = kwargs.get("node_type_indices", {}).get("gene")
                if gene_idx is not None and output.gene_logits is not None:
                    loss = F.cross_entropy(output.gene_logits, labels)
                    loss.backward()
                    self.optimizer.step()

            return output.loss_components or {}

        def evaluate(
            self,
            node_features: Dict[str, torch.Tensor],
            edge_index: torch.Tensor,
            edge_type: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> Dict[str, float]:
            """Evaluate model."""
            self.model.eval()

            with torch.no_grad():
                node_features = {k: v.to(self.device) for k, v in node_features.items()}
                edge_index = edge_index.to(self.device)
                edge_type = edge_type.to(self.device)
                if labels is not None:
                    labels = labels.to(self.device)

                output = self.model(
                    node_features, edge_index, edge_type,
                    labels=labels, **kwargs
                )

                metrics = {"loss": output.loss or 0.0}

                if output.gene_logits is not None and labels is not None:
                    preds = output.gene_logits.argmax(dim=-1)
                    acc = (preds == labels).float().mean().item()
                    metrics["accuracy"] = acc

                return metrics


else:
    # Numpy fallback implementations

    @dataclass
    class GNNConfig:
        """Configuration for OntologyAwareGNN (fallback)."""
        input_dim: int = 256
        hidden_dim: int = 256
        output_dim: int = 128
        num_layers: int = 3
        num_heads: int = 8
        edge_types: List[str] = field(default_factory=lambda: [
            "gene_interacts", "gene_in_pathway", "gene_has_go",
            "pathway_contains", "go_is_a",
        ])
        node_types: List[str] = field(default_factory=lambda: [
            "gene", "pathway", "go_term",
        ])
        prior_types: List[str] = field(default_factory=lambda: [
            "pli", "loeuf", "expression", "sfari_score",
        ])
        dropout: float = 0.1
        use_residual: bool = True
        use_layer_norm: bool = True
        aggregation: str = "attention"
        num_hierarchy_levels: int = 3
        task_heads: List[str] = field(default_factory=lambda: [
            "gene_classification", "link_prediction",
        ])


    class OntologyAwareGNN:
        """Numpy fallback for OntologyAwareGNN."""

        def __init__(self, config: GNNConfig, **kwargs):
            self.config = config
            logger.warning("Using numpy fallback - limited functionality")

        def forward(self, node_features, edge_index, edge_type, **kwargs):
            logger.warning("Using numpy fallback - returning random embeddings")
            # Return random embeddings
            if isinstance(node_features, dict):
                embeddings = {
                    k: np.random.randn(v.shape[0], self.config.output_dim)
                    for k, v in node_features.items()
                }
            else:
                embeddings = {"all": np.random.randn(node_features.shape[0], self.config.output_dim)}

            return GNNOutput(
                node_embeddings=embeddings,
                gene_logits=None,
                link_scores=None,
            )

        def encode(self, node_features, edge_index, edge_type, **kwargs):
            output = self.forward(node_features, edge_index, edge_type, **kwargs)
            all_emb = list(output.node_embeddings.values())
            return np.concatenate(all_emb, axis=0)


    class GNNTrainer:
        """Numpy fallback for GNNTrainer."""

        def __init__(self, model, **kwargs):
            self.model = model
            logger.warning("Using numpy fallback - training not available")

        def train_step(self, *args, **kwargs):
            return {"loss": 0.0}

        def evaluate(self, *args, **kwargs):
            return {"loss": 0.0, "accuracy": 0.5}
