"""
Module 04: Graph Embeddings

Knowledge graph embedding models for learning vector representations of
biological entities and their relationships.

This module provides:
- TransE: Translation-based embeddings (simple, fast, effective)
- RotatE: Rotation-based embeddings (handles complex relation patterns)
- EmbeddingTrainer: High-level training and evaluation utilities
- NodeEmbeddings: Container for learned embeddings with similarity search

Example Usage:
    >>> from modules.03_knowledge_graph import KnowledgeGraphBuilder, NodeType, EdgeType
    >>> from modules.04_graph_embeddings import TransEModel, train_embeddings
    >>>
    >>> # Build a knowledge graph
    >>> kg = (
    ...     KnowledgeGraphBuilder()
    ...     .add_genes(["SHANK3", "CHD8", "SCN2A", "NRXN1"])
    ...     .build()
    ... )
    >>>
    >>> # Train TransE embeddings
    >>> model = TransEModel(embedding_dim=64)
    >>> history = model.train(kg, epochs=50)
    >>>
    >>> # Get embeddings
    >>> embeddings = model.get_node_embeddings()
    >>> shank3_emb = embeddings.get("SHANK3")
    >>>
    >>> # Find similar genes
    >>> similar = embeddings.most_similar("SHANK3", k=5)

For more details, see the README.md in this module directory.
"""

try:
    from .base import (
        NodeEmbeddings,
        RelationEmbeddings,
        TrainingHistory,
        EvaluationMetrics,
        BaseEmbeddingModel,
    )
    from .transe import TransEModel, create_transe_model
    from .rotate import RotatEModel, create_rotate_model
    from .trainer import (
        TrainingConfig,
        EmbeddingTrainer,
        EmbeddingPipeline,
        train_embeddings,
        compare_models,
    )
except ImportError:
    from base import (
        NodeEmbeddings,
        RelationEmbeddings,
        TrainingHistory,
        EvaluationMetrics,
        BaseEmbeddingModel,
    )
    from transe import TransEModel, create_transe_model
    from rotate import RotatEModel, create_rotate_model
    from trainer import (
        TrainingConfig,
        EmbeddingTrainer,
        EmbeddingPipeline,
        train_embeddings,
        compare_models,
    )

__all__ = [
    # Data structures
    "NodeEmbeddings",
    "RelationEmbeddings",
    "TrainingHistory",
    "EvaluationMetrics",
    # Base class
    "BaseEmbeddingModel",
    # Models
    "TransEModel",
    "RotatEModel",
    # Factory functions
    "create_transe_model",
    "create_rotate_model",
    # Training utilities
    "TrainingConfig",
    "EmbeddingTrainer",
    "EmbeddingPipeline",
    "train_embeddings",
    "compare_models",
]
