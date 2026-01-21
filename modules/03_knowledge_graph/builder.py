"""
Knowledge Graph Builder

Constructs a heterogeneous biological knowledge graph from various data sources
including pathways, GO terms, protein-protein interactions, and gene annotations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union
import json
import logging
import pickle

import networkx as nx

try:
    from .schema import EdgeType, GraphSchema, Node, NodeType, Edge
except ImportError:
    from schema import EdgeType, GraphSchema, Node, NodeType, Edge

logger = logging.getLogger(__name__)


@dataclass
class PPINetwork:
    """Protein-protein interaction network data."""

    interactions: List[Tuple[str, str, float]]  # (protein1, protein2, score)
    source: str = "STRING"
    score_threshold: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.interactions)

    def filter_by_score(self, min_score: float) -> "PPINetwork":
        """Filter interactions by minimum combined score."""
        filtered = [
            (p1, p2, score)
            for p1, p2, score in self.interactions
            if score >= min_score
        ]
        return PPINetwork(
            interactions=filtered,
            source=self.source,
            score_threshold=min_score,
            metadata={**self.metadata, "filtered_from": len(self.interactions)},
        )


@dataclass
class KnowledgeGraphStats:
    """Statistics about the knowledge graph."""

    n_nodes: int
    n_edges: int
    node_type_counts: Dict[str, int]
    edge_type_counts: Dict[str, int]
    avg_degree: float
    density: float
    n_connected_components: int


class KnowledgeGraph:
    """
    Heterogeneous knowledge graph for biological data.

    Uses NetworkX internally for graph operations with support for
    multiple node types and edge types.
    """

    def __init__(self, schema: Optional[GraphSchema] = None):
        """
        Initialize knowledge graph.

        Args:
            schema: Optional schema defining valid node/edge types
        """
        self._graph = nx.MultiDiGraph()
        self._schema = schema or GraphSchema.default_schema()
        self._node_types: Dict[str, NodeType] = {}
        self._edge_types: Dict[Tuple[str, str, int], EdgeType] = {}

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Access underlying NetworkX graph."""
        return self._graph

    @property
    def schema(self) -> GraphSchema:
        """Get graph schema."""
        return self._schema

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a node to the graph.

        Args:
            node_id: Unique node identifier
            node_type: Type of the node
            attributes: Optional node attributes
        """
        if node_type not in self._schema.node_types:
            raise ValueError(f"Invalid node type: {node_type}")

        attrs = attributes or {}
        self._graph.add_node(node_id, node_type=node_type.value, **attrs)
        self._node_types[node_id] = node_type

    def add_nodes(
        self,
        node_ids: List[str],
        node_type: NodeType,
        attributes: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Add multiple nodes of the same type.

        Args:
            node_ids: List of node identifiers
            node_type: Type of all nodes
            attributes: Optional dict mapping node_id to attributes
        """
        attrs = attributes or {}
        for node_id in node_ids:
            node_attrs = attrs.get(node_id, {})
            self.add_node(node_id, node_type, node_attrs)

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an edge to the graph.

        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of the edge
            weight: Edge weight
            attributes: Optional edge attributes
        """
        if source not in self._node_types:
            raise ValueError(f"Source node not found: {source}")
        if target not in self._node_types:
            raise ValueError(f"Target node not found: {target}")

        source_type = self._node_types[source]
        target_type = self._node_types[target]

        if not self._schema.is_valid_edge(source_type, target_type, edge_type):
            raise ValueError(
                f"Invalid edge type {edge_type} for {source_type} -> {target_type}"
            )

        attrs = attributes or {}
        key = self._graph.add_edge(
            source, target, edge_type=edge_type.value, weight=weight, **attrs
        )
        self._edge_types[(source, target, key)] = edge_type

    def add_edges(
        self,
        edges: List[Tuple[str, str]],
        edge_type: EdgeType,
        weights: Optional[List[float]] = None,
        skip_missing_nodes: bool = True,
    ) -> int:
        """
        Add multiple edges of the same type.

        Args:
            edges: List of (source, target) tuples
            edge_type: Type of all edges
            weights: Optional weights for each edge
            skip_missing_nodes: If True, skip edges with missing nodes

        Returns:
            Number of edges added
        """
        if weights is None:
            weights = [1.0] * len(edges)

        added = 0
        for (source, target), weight in zip(edges, weights):
            try:
                self.add_edge(source, target, edge_type, weight)
                added += 1
            except ValueError as e:
                if not skip_missing_nodes:
                    raise
                logger.debug(f"Skipping edge {source}->{target}: {e}")

        return added

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        if node_id not in self._graph:
            return None
        attrs = dict(self._graph.nodes[node_id])
        node_type = NodeType(attrs.pop("node_type"))
        return Node(id=node_id, node_type=node_type, attributes=attrs)

    def get_node_type(self, node_id: str) -> Optional[NodeType]:
        """Get the type of a node."""
        return self._node_types.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> List[str]:
        """Get all nodes of a specific type."""
        return [
            node_id
            for node_id, nt in self._node_types.items()
            if nt == node_type
        ]

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "out",
    ) -> List[str]:
        """
        Get neighbors of a node.

        Args:
            node_id: Node to get neighbors for
            edge_type: Optional edge type filter
            direction: "out" for successors, "in" for predecessors, "both" for both

        Returns:
            List of neighbor node IDs
        """
        if node_id not in self._graph:
            return []

        neighbors = set()

        if direction in ("out", "both"):
            for _, target, data in self._graph.out_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type.value:
                    neighbors.add(target)

        if direction in ("in", "both"):
            for source, _, data in self._graph.in_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type.value:
                    neighbors.add(source)

        return list(neighbors)

    def get_edges_by_type(self, edge_type: EdgeType) -> List[Tuple[str, str, float]]:
        """Get all edges of a specific type."""
        edges = []
        for source, target, data in self._graph.edges(data=True):
            if data.get("edge_type") == edge_type.value:
                edges.append((source, target, data.get("weight", 1.0)))
        return edges

    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return node_id in self._graph

    def has_edge(
        self,
        source: str,
        target: str,
        edge_type: Optional[EdgeType] = None,
    ) -> bool:
        """Check if edge exists."""
        if not self._graph.has_edge(source, target):
            return False
        if edge_type is None:
            return True
        for _, _, data in self._graph.edges(source, data=True):
            if data.get("edge_type") == edge_type.value:
                return True
        return False

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        """Number of edges in the graph."""
        return self._graph.number_of_edges()

    def get_stats(self) -> KnowledgeGraphStats:
        """Get statistics about the graph."""
        node_type_counts = {}
        for node_type in NodeType:
            count = len(self.get_nodes_by_type(node_type))
            if count > 0:
                node_type_counts[node_type.value] = count

        edge_type_counts = {}
        for source, target, data in self._graph.edges(data=True):
            et = data.get("edge_type", "unknown")
            edge_type_counts[et] = edge_type_counts.get(et, 0) + 1

        # Calculate average degree
        if self.n_nodes > 0:
            avg_degree = self.n_edges / self.n_nodes
        else:
            avg_degree = 0.0

        # Calculate density
        if self.n_nodes > 1:
            max_edges = self.n_nodes * (self.n_nodes - 1)
            density = self.n_edges / max_edges
        else:
            density = 0.0

        # Count connected components (treating as undirected)
        undirected = self._graph.to_undirected()
        n_components = nx.number_connected_components(undirected)

        return KnowledgeGraphStats(
            n_nodes=self.n_nodes,
            n_edges=self.n_edges,
            node_type_counts=node_type_counts,
            edge_type_counts=edge_type_counts,
            avg_degree=avg_degree,
            density=density,
            n_connected_components=n_components,
        )

    def subgraph(
        self,
        node_ids: List[str],
        include_edges: bool = True,
    ) -> "KnowledgeGraph":
        """
        Create a subgraph with specified nodes.

        Args:
            node_ids: Nodes to include
            include_edges: Whether to include edges between nodes

        Returns:
            New KnowledgeGraph with subset of nodes
        """
        subgraph = KnowledgeGraph(schema=self._schema)

        for node_id in node_ids:
            if node_id in self._node_types:
                node = self.get_node(node_id)
                if node:
                    subgraph.add_node(node_id, node.node_type, node.attributes)

        if include_edges:
            node_set = set(node_ids)
            for source, target, data in self._graph.edges(data=True):
                if source in node_set and target in node_set:
                    edge_type = EdgeType(data.get("edge_type"))
                    weight = data.get("weight", 1.0)
                    attrs = {k: v for k, v in data.items() if k not in ("edge_type", "weight")}
                    subgraph.add_edge(source, target, edge_type, weight, attrs)

        return subgraph

    def save(self, path: str) -> None:
        """
        Save graph to file.

        Args:
            path: Output path (supports .gpickle, .json)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".gpickle":
            with open(path, "wb") as f:
                pickle.dump(
                    {
                        "graph": self._graph,
                        "node_types": self._node_types,
                        "edge_types": self._edge_types,
                    },
                    f,
                )
        elif path.suffix == ".json":
            data = nx.node_link_data(self._graph)
            data["node_types"] = {k: v.value for k, v in self._node_types.items()}
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        logger.info(f"Saved graph to {path}")

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        """
        Load graph from file.

        Args:
            path: Input path

        Returns:
            KnowledgeGraph
        """
        path = Path(path)

        kg = cls()

        if path.suffix == ".gpickle":
            with open(path, "rb") as f:
                data = pickle.load(f)
            kg._graph = data["graph"]
            kg._node_types = data["node_types"]
            kg._edge_types = data.get("edge_types", {})
        elif path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
            node_types_data = data.pop("node_types", {})
            kg._graph = nx.node_link_graph(data)
            kg._node_types = {k: NodeType(v) for k, v in node_types_data.items()}
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        logger.info(f"Loaded graph from {path}: {kg.n_nodes} nodes, {kg.n_edges} edges")
        return kg


class KnowledgeGraphBuilder:
    """
    Builder for constructing knowledge graphs from various data sources.

    Supports:
    - Pathway databases (GO, Reactome)
    - Protein-protein interaction networks (STRING)
    - Gene annotations
    - BigQuery data sources (for cloud deployment)
    """

    def __init__(self, schema: Optional[GraphSchema] = None):
        """
        Initialize builder.

        Args:
            schema: Optional schema for validation
        """
        self._schema = schema or GraphSchema.default_schema()
        self._genes: Set[str] = set()
        self._pathways: Dict[str, Set[str]] = {}
        self._pathway_names: Dict[str, str] = {}
        self._go_terms: Dict[str, Dict[str, str]] = {}  # id -> {name, namespace}
        self._gene_go_annotations: Dict[str, Set[str]] = {}  # gene -> go_ids
        self._ppi_edges: List[Tuple[str, str, float]] = []
        self._go_hierarchy: List[Tuple[str, str, str]] = []  # (child, parent, rel_type)

    def add_genes(self, gene_list: List[str]) -> "KnowledgeGraphBuilder":
        """
        Add genes to the graph.

        Args:
            gene_list: List of gene symbols/IDs

        Returns:
            Self for chaining
        """
        self._genes.update(gene_list)
        logger.info(f"Added {len(gene_list)} genes (total: {len(self._genes)})")
        return self

    def add_pathways(self, pathway_db: Any) -> "KnowledgeGraphBuilder":
        """
        Add pathways from a PathwayDatabase.

        Args:
            pathway_db: PathwayDatabase instance from Module 01

        Returns:
            Self for chaining
        """
        for pathway_id, genes in pathway_db.pathways.items():
            self._pathways[pathway_id] = genes.copy()
            self._pathway_names[pathway_id] = pathway_db.pathway_names.get(
                pathway_id, pathway_id
            )
            # Add genes from pathways
            self._genes.update(genes)

        logger.info(
            f"Added {len(pathway_db.pathways)} pathways "
            f"(source: {pathway_db.source})"
        )
        return self

    def add_go_terms(
        self,
        go_terms: Dict[str, Dict[str, str]],
        annotations: Dict[str, Set[str]],
    ) -> "KnowledgeGraphBuilder":
        """
        Add GO terms and gene annotations.

        Args:
            go_terms: Dict mapping GO ID to term info (name, namespace)
            annotations: Dict mapping gene symbol to set of GO IDs

        Returns:
            Self for chaining
        """
        self._go_terms.update(go_terms)
        for gene, go_ids in annotations.items():
            if gene not in self._gene_go_annotations:
                self._gene_go_annotations[gene] = set()
            self._gene_go_annotations[gene].update(go_ids)
            self._genes.add(gene)

        logger.info(
            f"Added {len(go_terms)} GO terms with "
            f"{len(annotations)} gene annotations"
        )
        return self

    def add_go_hierarchy(
        self,
        hierarchy: List[Tuple[str, str, str]],
    ) -> "KnowledgeGraphBuilder":
        """
        Add GO term hierarchy relationships.

        Args:
            hierarchy: List of (child_id, parent_id, relationship_type) tuples
                      where relationship_type is 'is_a' or 'part_of'

        Returns:
            Self for chaining
        """
        self._go_hierarchy.extend(hierarchy)
        logger.info(f"Added {len(hierarchy)} GO hierarchy relationships")
        return self

    def add_ppi(
        self,
        ppi_network: PPINetwork,
        gene_to_protein: Optional[Dict[str, str]] = None,
    ) -> "KnowledgeGraphBuilder":
        """
        Add protein-protein interactions.

        Args:
            ppi_network: PPINetwork with interaction data
            gene_to_protein: Optional mapping from gene symbols to protein IDs

        Returns:
            Self for chaining
        """
        for protein1, protein2, score in ppi_network.interactions:
            # Convert protein IDs to gene symbols if mapping provided
            if gene_to_protein:
                # Reverse mapping
                protein_to_gene = {v: k for k, v in gene_to_protein.items()}
                gene1 = protein_to_gene.get(protein1, protein1)
                gene2 = protein_to_gene.get(protein2, protein2)
            else:
                # Assume protein IDs are gene symbols or can be mapped directly
                # STRING uses ENSP format, we'll extract gene symbols
                gene1 = self._protein_id_to_gene(protein1)
                gene2 = self._protein_id_to_gene(protein2)

            self._ppi_edges.append((gene1, gene2, score))
            self._genes.add(gene1)
            self._genes.add(gene2)

        logger.info(
            f"Added {len(ppi_network.interactions)} PPI edges "
            f"(threshold: {ppi_network.score_threshold})"
        )
        return self

    def add_ppi_from_dataframe(
        self,
        df: Any,
        protein1_col: str = "protein1",
        protein2_col: str = "protein2",
        score_col: str = "combined_score",
        min_score: float = 0.0,
    ) -> "KnowledgeGraphBuilder":
        """
        Add PPI from a pandas DataFrame.

        Args:
            df: DataFrame with PPI data
            protein1_col: Column name for first protein
            protein2_col: Column name for second protein
            score_col: Column name for interaction score
            min_score: Minimum score threshold

        Returns:
            Self for chaining
        """
        count = 0
        for _, row in df.iterrows():
            score = row[score_col]
            if score < min_score:
                continue

            protein1 = str(row[protein1_col])
            protein2 = str(row[protein2_col])

            gene1 = self._protein_id_to_gene(protein1)
            gene2 = self._protein_id_to_gene(protein2)

            self._ppi_edges.append((gene1, gene2, score))
            self._genes.add(gene1)
            self._genes.add(gene2)
            count += 1

        logger.info(f"Added {count} PPI edges from DataFrame")
        return self

    def _protein_id_to_gene(self, protein_id: str) -> str:
        """
        Convert protein ID to gene symbol.

        STRING uses format like '9606.ENSP00000269305'.
        """
        if "." in protein_id:
            # Remove species prefix
            protein_id = protein_id.split(".")[-1]

        # For ENSP IDs, we'd need a mapping table
        # For now, return as-is (user should provide mapping)
        return protein_id

    def build(self) -> KnowledgeGraph:
        """
        Build the knowledge graph from added data.

        Returns:
            Constructed KnowledgeGraph
        """
        kg = KnowledgeGraph(schema=self._schema)

        # Add gene nodes
        logger.info(f"Adding {len(self._genes)} gene nodes...")
        for gene in self._genes:
            kg.add_node(gene, NodeType.GENE)

        # Add pathway nodes and gene-pathway edges
        logger.info(f"Adding {len(self._pathways)} pathway nodes...")
        for pathway_id, genes in self._pathways.items():
            kg.add_node(
                pathway_id,
                NodeType.PATHWAY,
                {"name": self._pathway_names.get(pathway_id, pathway_id)},
            )
            for gene in genes:
                if kg.has_node(gene):
                    kg.add_edge(gene, pathway_id, EdgeType.GENE_IN_PATHWAY)

        # Add GO term nodes and gene-GO edges
        logger.info(f"Adding {len(self._go_terms)} GO term nodes...")
        for go_id, term_info in self._go_terms.items():
            kg.add_node(
                go_id,
                NodeType.GO_TERM,
                {
                    "name": term_info.get("name", go_id),
                    "namespace": term_info.get("namespace", "unknown"),
                },
            )

        for gene, go_ids in self._gene_go_annotations.items():
            if not kg.has_node(gene):
                continue
            for go_id in go_ids:
                if kg.has_node(go_id):
                    kg.add_edge(gene, go_id, EdgeType.GENE_HAS_GO)

        # Add GO hierarchy edges
        logger.info(f"Adding {len(self._go_hierarchy)} GO hierarchy edges...")
        for child_id, parent_id, rel_type in self._go_hierarchy:
            if kg.has_node(child_id) and kg.has_node(parent_id):
                if rel_type == "is_a":
                    kg.add_edge(child_id, parent_id, EdgeType.GO_IS_A)
                elif rel_type == "part_of":
                    kg.add_edge(child_id, parent_id, EdgeType.GO_PART_OF)

        # Add PPI edges
        logger.info(f"Adding {len(self._ppi_edges)} PPI edges...")
        ppi_added = 0
        for gene1, gene2, score in self._ppi_edges:
            if kg.has_node(gene1) and kg.has_node(gene2) and gene1 != gene2:
                # Normalize score to 0-1 (STRING scores are 0-1000)
                normalized_score = score / 1000.0 if score > 1 else score
                kg.add_edge(gene1, gene2, EdgeType.GENE_INTERACTS, normalized_score)
                ppi_added += 1

        stats = kg.get_stats()
        logger.info(
            f"Built knowledge graph: {stats.n_nodes} nodes, {stats.n_edges} edges, "
            f"{stats.n_connected_components} components"
        )

        return kg

    def clear(self) -> "KnowledgeGraphBuilder":
        """Clear all added data."""
        self._genes.clear()
        self._pathways.clear()
        self._pathway_names.clear()
        self._go_terms.clear()
        self._gene_go_annotations.clear()
        self._ppi_edges.clear()
        self._go_hierarchy.clear()
        return self


def load_ppi_from_file(
    file_path: str,
    min_score: float = 400.0,
    max_edges: Optional[int] = None,
) -> PPINetwork:
    """
    Load PPI network from a STRING-format file.

    Args:
        file_path: Path to STRING PPI file
        min_score: Minimum combined score (STRING uses 0-1000)
        max_edges: Optional maximum number of edges to load

    Returns:
        PPINetwork
    """
    interactions = []
    path = Path(file_path)

    with open(path, "r") as f:
        header = f.readline()  # Skip header

        for i, line in enumerate(f):
            if max_edges and i >= max_edges:
                break

            parts = line.strip().split()
            if len(parts) >= 3:
                protein1 = parts[0]
                protein2 = parts[1]
                score = float(parts[-1])  # Last column is combined_score

                if score >= min_score:
                    interactions.append((protein1, protein2, score))

    logger.info(
        f"Loaded {len(interactions)} PPI interactions from {file_path} "
        f"(min_score={min_score})"
    )

    return PPINetwork(
        interactions=interactions,
        source="STRING",
        score_threshold=min_score,
        metadata={"file": str(path)},
    )
