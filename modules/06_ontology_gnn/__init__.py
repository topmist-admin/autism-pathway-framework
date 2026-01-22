"""
Module 06: Ontology-Aware Graph Neural Network

Graph neural network architecture designed for biological knowledge graphs
with support for:
- Heterogeneous node types (genes, pathways, GO terms)
- Multiple edge types (PPI, pathway membership, GO annotations)
- Biological attention mechanisms (constraint scores, expression)
- Ontology hierarchy aggregation (GO term hierarchy)
- Multi-task learning (classification, link prediction)

Example Usage:
    >>> from modules.06_ontology_gnn import (
    ...     OntologyAwareGNN,
    ...     GNNConfig,
    ...     OntologyGNNConfig,
    ...     prepare_graph_data,
    ... )
    >>>
    >>> # Create configuration
    >>> config = GNNConfig(hidden_dim=256, num_layers=3)
    >>> model = OntologyAwareGNN(config)
    >>>
    >>> # Prepare data from knowledge graph
    >>> graph_data = prepare_graph_data(
    ...     knowledge_graph,
    ...     node_embeddings=embeddings,
    ...     bio_priors={"pli": pli_scores},
    ... )
    >>>
    >>> # Forward pass
    >>> output = model(
    ...     node_features=graph_data.node_features,
    ...     edge_index=graph_data.edge_index,
    ...     edge_type=graph_data.edge_type,
    ...     edge_type_names=graph_data.edge_type_names,
    ... )
    >>>
    >>> # Get gene embeddings
    >>> gene_embeddings = output.node_embeddings["gene"]

Note:
    Requires PyTorch for full functionality.
    Falls back to numpy implementations for testing without GPU.
"""

try:
    from .layers import (
        # GNN Layers
        EdgeTypeTransform,
        MessagePassingLayer,
        HierarchicalAggregator,
        BioPriorWeighting,
        TORCH_AVAILABLE,
    )
    from .attention import (
        # Attention Mechanisms
        BiologicalAttention,
        EdgeTypeAttention,
        GOSemanticAttention,
        PathwayCoAttention,
    )
    from .model import (
        # Main Model
        OntologyAwareGNN,
        GNNConfig,
        GNNOutput,
        GNNTrainer,
    )
    from .config import (
        # Configuration
        OntologyGNNConfig,
        ModelConfig,
        GraphConfig,
        EdgeTypeConfig,
        BioPriorConfig,
        HierarchyConfig,
        TrainingConfig,
        TaskConfig,
        # Enums
        AggregationType,
        ActivationType,
        PriorCombination,
        # Config factories
        create_default_config,
        create_autism_config,
        create_lightweight_config,
    )
    from .utils import (
        # Utilities
        GraphData,
        prepare_graph_data,
        normalize_priors,
        compute_metrics,
        compute_link_prediction_metrics,
        create_negative_samples,
        split_edges,
        get_subgraph,
    )
except ImportError:
    from layers import (
        EdgeTypeTransform,
        MessagePassingLayer,
        HierarchicalAggregator,
        BioPriorWeighting,
        TORCH_AVAILABLE,
    )
    from attention import (
        BiologicalAttention,
        EdgeTypeAttention,
        GOSemanticAttention,
        PathwayCoAttention,
    )
    from model import (
        OntologyAwareGNN,
        GNNConfig,
        GNNOutput,
        GNNTrainer,
    )
    from config import (
        OntologyGNNConfig,
        ModelConfig,
        GraphConfig,
        EdgeTypeConfig,
        BioPriorConfig,
        HierarchyConfig,
        TrainingConfig,
        TaskConfig,
        AggregationType,
        ActivationType,
        PriorCombination,
        create_default_config,
        create_autism_config,
        create_lightweight_config,
    )
    from utils import (
        GraphData,
        prepare_graph_data,
        normalize_priors,
        compute_metrics,
        compute_link_prediction_metrics,
        create_negative_samples,
        split_edges,
        get_subgraph,
    )


__all__ = [
    # GNN Layers
    "EdgeTypeTransform",
    "MessagePassingLayer",
    "HierarchicalAggregator",
    "BioPriorWeighting",
    # Attention Mechanisms
    "BiologicalAttention",
    "EdgeTypeAttention",
    "GOSemanticAttention",
    "PathwayCoAttention",
    # Main Model
    "OntologyAwareGNN",
    "GNNConfig",
    "GNNOutput",
    "GNNTrainer",
    # Configuration
    "OntologyGNNConfig",
    "ModelConfig",
    "GraphConfig",
    "EdgeTypeConfig",
    "BioPriorConfig",
    "HierarchyConfig",
    "TrainingConfig",
    "TaskConfig",
    # Enums
    "AggregationType",
    "ActivationType",
    "PriorCombination",
    # Config factories
    "create_default_config",
    "create_autism_config",
    "create_lightweight_config",
    # Utilities
    "GraphData",
    "prepare_graph_data",
    "normalize_priors",
    "compute_metrics",
    "compute_link_prediction_metrics",
    "create_negative_samples",
    "split_edges",
    "get_subgraph",
    # Flags
    "TORCH_AVAILABLE",
]
