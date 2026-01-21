"""
Module 03: Knowledge Graph

Provides tools for building and exporting heterogeneous biological knowledge graphs
from pathway databases, GO terms, and protein-protein interaction networks.
"""

try:
    from .schema import (
        NodeType,
        EdgeType,
        Node,
        Edge,
        NodeFeatures,
        GraphSchema,
        EDGE_TYPE_METADATA,
    )
    from .builder import (
        KnowledgeGraph,
        KnowledgeGraphBuilder,
        KnowledgeGraphStats,
        PPINetwork,
        load_ppi_from_file,
    )
    from .exporters import (
        NodeMapping,
        create_node_mapping,
        to_dgl,
        to_pyg,
        to_neo4j_cypher,
        to_csv,
        to_adjacency_matrix,
        to_edge_list,
    )
except ImportError:
    from schema import (
        NodeType,
        EdgeType,
        Node,
        Edge,
        NodeFeatures,
        GraphSchema,
        EDGE_TYPE_METADATA,
    )
    from builder import (
        KnowledgeGraph,
        KnowledgeGraphBuilder,
        KnowledgeGraphStats,
        PPINetwork,
        load_ppi_from_file,
    )
    from exporters import (
        NodeMapping,
        create_node_mapping,
        to_dgl,
        to_pyg,
        to_neo4j_cypher,
        to_csv,
        to_adjacency_matrix,
        to_edge_list,
    )

__all__ = [
    # Schema
    "NodeType",
    "EdgeType",
    "Node",
    "Edge",
    "NodeFeatures",
    "GraphSchema",
    "EDGE_TYPE_METADATA",
    # Builder
    "KnowledgeGraph",
    "KnowledgeGraphBuilder",
    "KnowledgeGraphStats",
    "PPINetwork",
    "load_ppi_from_file",
    # Exporters
    "NodeMapping",
    "create_node_mapping",
    "to_dgl",
    "to_pyg",
    "to_neo4j_cypher",
    "to_csv",
    "to_adjacency_matrix",
    "to_edge_list",
]
