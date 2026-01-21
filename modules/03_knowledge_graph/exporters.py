"""
Knowledge Graph Exporters

Export knowledge graphs to various formats for GNN training and analysis:
- DGL (Deep Graph Library)
- PyTorch Geometric (PyG)
- Neo4j (graph database)
- CSV/TSV (tabular)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging

import numpy as np

try:
    from .schema import EdgeType, NodeType
    from .builder import KnowledgeGraph
except ImportError:
    from schema import EdgeType, NodeType
    from builder import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class NodeMapping:
    """Mapping between node IDs and integer indices."""

    node_to_idx: Dict[str, int]
    idx_to_node: Dict[int, str]
    node_types: Dict[str, NodeType]

    def __len__(self) -> int:
        return len(self.node_to_idx)

    def get_idx(self, node_id: str) -> int:
        return self.node_to_idx[node_id]

    def get_node(self, idx: int) -> str:
        return self.idx_to_node[idx]

    def get_type(self, node_id: str) -> NodeType:
        return self.node_types[node_id]

    def save(self, path: str) -> None:
        """Save mapping to JSON."""
        data = {
            "node_to_idx": self.node_to_idx,
            "node_types": {k: v.value for k, v in self.node_types.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "NodeMapping":
        """Load mapping from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        node_to_idx = data["node_to_idx"]
        idx_to_node = {int(v): k for k, v in node_to_idx.items()}
        node_types = {k: NodeType(v) for k, v in data["node_types"].items()}
        return cls(node_to_idx=node_to_idx, idx_to_node=idx_to_node, node_types=node_types)


def create_node_mapping(kg: KnowledgeGraph) -> NodeMapping:
    """
    Create a mapping from node IDs to integer indices.

    Args:
        kg: Knowledge graph

    Returns:
        NodeMapping with bidirectional mappings
    """
    node_to_idx = {}
    idx_to_node = {}
    node_types = {}

    for idx, node_id in enumerate(kg.graph.nodes()):
        node_to_idx[node_id] = idx
        idx_to_node[idx] = node_id
        node_types[node_id] = kg.get_node_type(node_id)

    return NodeMapping(
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        node_types=node_types,
    )


def to_dgl(
    kg: KnowledgeGraph,
    node_features: Optional[Dict[str, np.ndarray]] = None,
    edge_features: Optional[Dict[EdgeType, np.ndarray]] = None,
) -> Tuple[Any, NodeMapping]:
    """
    Export knowledge graph to DGL format.

    Creates a heterogeneous graph with different node and edge types.

    Args:
        kg: Knowledge graph to export
        node_features: Optional dict mapping node_type to feature array
        edge_features: Optional dict mapping edge_type to feature array

    Returns:
        Tuple of (DGLGraph, NodeMapping)

    Raises:
        ImportError: If DGL is not installed
    """
    try:
        import dgl
        import torch
    except ImportError:
        raise ImportError(
            "DGL and PyTorch are required for DGL export. "
            "Install with: pip install dgl torch"
        )

    # Create node mapping
    node_mapping = create_node_mapping(kg)

    # Group edges by (source_type, edge_type, target_type) for heterogeneous graph
    edge_dict: Dict[Tuple[str, str, str], Tuple[List[int], List[int]]] = {}

    for source, target, data in kg.graph.edges(data=True):
        edge_type_str = data.get("edge_type", "unknown")
        source_type = kg.get_node_type(source)
        target_type = kg.get_node_type(target)

        if source_type is None or target_type is None:
            continue

        # DGL uses (src_type, edge_type, dst_type) as key
        key = (source_type.value, edge_type_str, target_type.value)

        if key not in edge_dict:
            edge_dict[key] = ([], [])

        edge_dict[key][0].append(node_mapping.get_idx(source))
        edge_dict[key][1].append(node_mapping.get_idx(target))

    # Convert to tensors
    edge_dict_tensors = {
        key: (torch.tensor(src), torch.tensor(dst))
        for key, (src, dst) in edge_dict.items()
    }

    # Create heterogeneous graph
    g = dgl.heterograph(edge_dict_tensors)

    # Add node features if provided
    if node_features:
        for node_type, features in node_features.items():
            if node_type in g.ntypes:
                g.nodes[node_type].data["feat"] = torch.tensor(features, dtype=torch.float32)

    # Add edge features if provided
    if edge_features:
        for edge_type, features in edge_features.items():
            etype_str = edge_type.value
            for etype in g.etypes:
                if etype_str in etype:
                    g.edges[etype].data["feat"] = torch.tensor(features, dtype=torch.float32)

    logger.info(
        f"Exported to DGL: {g.num_nodes()} nodes, {g.num_edges()} edges, "
        f"{len(g.ntypes)} node types, {len(g.etypes)} edge types"
    )

    return g, node_mapping


def to_pyg(
    kg: KnowledgeGraph,
    node_features: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Any, NodeMapping]:
    """
    Export knowledge graph to PyTorch Geometric (PyG) format.

    Creates a HeteroData object with different node and edge types.

    Args:
        kg: Knowledge graph to export
        node_features: Optional dict mapping node_type to feature array

    Returns:
        Tuple of (HeteroData, NodeMapping)

    Raises:
        ImportError: If PyG is not installed
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError:
        raise ImportError(
            "PyTorch Geometric is required for PyG export. "
            "Install with: pip install torch-geometric"
        )

    # Create node mapping
    node_mapping = create_node_mapping(kg)

    # Create HeteroData
    data = HeteroData()

    # Add nodes by type
    node_type_counts: Dict[str, int] = {}
    node_type_to_local_idx: Dict[str, Dict[str, int]] = {}

    for node_id in kg.graph.nodes():
        node_type = kg.get_node_type(node_id)
        if node_type is None:
            continue

        type_str = node_type.value
        if type_str not in node_type_counts:
            node_type_counts[type_str] = 0
            node_type_to_local_idx[type_str] = {}

        node_type_to_local_idx[type_str][node_id] = node_type_counts[type_str]
        node_type_counts[type_str] += 1

    # Set number of nodes per type
    for node_type, count in node_type_counts.items():
        data[node_type].num_nodes = count

    # Add node features if provided
    if node_features:
        for node_type, features in node_features.items():
            if node_type in node_type_counts:
                data[node_type].x = torch.tensor(features, dtype=torch.float32)

    # Group edges by (source_type, edge_type, target_type)
    edge_dict: Dict[Tuple[str, str, str], Tuple[List[int], List[int]]] = {}

    for source, target, edge_data in kg.graph.edges(data=True):
        edge_type_str = edge_data.get("edge_type", "unknown")
        source_type = kg.get_node_type(source)
        target_type = kg.get_node_type(target)

        if source_type is None or target_type is None:
            continue

        src_type_str = source_type.value
        tgt_type_str = target_type.value

        # PyG uses (src_type, edge_type, dst_type) as key
        key = (src_type_str, edge_type_str, tgt_type_str)

        if key not in edge_dict:
            edge_dict[key] = ([], [])

        # Use local indices within each node type
        src_idx = node_type_to_local_idx[src_type_str][source]
        tgt_idx = node_type_to_local_idx[tgt_type_str][target]

        edge_dict[key][0].append(src_idx)
        edge_dict[key][1].append(tgt_idx)

    # Add edges to HeteroData
    for (src_type, edge_type, dst_type), (src_list, dst_list) in edge_dict.items():
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        data[src_type, edge_type, dst_type].edge_index = edge_index

    logger.info(
        f"Exported to PyG: {sum(node_type_counts.values())} nodes, "
        f"{sum(len(e[0]) for e in edge_dict.values())} edges"
    )

    return data, node_mapping


def to_neo4j_cypher(
    kg: KnowledgeGraph,
    output_path: str,
    batch_size: int = 1000,
) -> None:
    """
    Export knowledge graph to Neo4j Cypher statements.

    Creates a .cypher file with CREATE statements for nodes and relationships.

    Args:
        kg: Knowledge graph to export
        output_path: Path to output .cypher file
        batch_size: Number of statements per batch
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        # Write node creation statements
        f.write("// Create nodes\n")
        for i, (node_id, data) in enumerate(kg.graph.nodes(data=True)):
            node_type = data.get("node_type", "Node")
            # Escape quotes in properties
            props = {k: v for k, v in data.items() if k != "node_type"}
            props["id"] = node_id

            props_str = ", ".join(
                f'{k}: "{str(v).replace(chr(34), chr(92) + chr(34))}"'
                for k, v in props.items()
            )

            f.write(f"CREATE (:{node_type} {{{props_str}}})\n")

            if (i + 1) % batch_size == 0:
                f.write(";\n")

        f.write(";\n\n")

        # Create indexes
        f.write("// Create indexes\n")
        for node_type in NodeType:
            f.write(f"CREATE INDEX ON :{node_type.value}(id);\n")
        f.write("\n")

        # Write relationship creation statements
        f.write("// Create relationships\n")
        for i, (source, target, data) in enumerate(kg.graph.edges(data=True)):
            edge_type = data.get("edge_type", "RELATES_TO")
            weight = data.get("weight", 1.0)

            f.write(
                f'MATCH (a {{id: "{source}"}}), (b {{id: "{target}"}}) '
                f"CREATE (a)-[:{edge_type} {{weight: {weight}}}]->(b)\n"
            )

            if (i + 1) % batch_size == 0:
                f.write(";\n")

        f.write(";\n")

    logger.info(f"Exported Neo4j Cypher to {path}")


def to_csv(
    kg: KnowledgeGraph,
    output_dir: str,
    include_attributes: bool = True,
) -> Dict[str, str]:
    """
    Export knowledge graph to CSV files.

    Creates separate files for nodes and edges, compatible with Neo4j import.

    Args:
        kg: Knowledge graph to export
        output_dir: Directory for output files
        include_attributes: Whether to include node/edge attributes

    Returns:
        Dict mapping file type to path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = {}

    # Export nodes
    nodes_path = output_path / "nodes.csv"
    with open(nodes_path, "w") as f:
        f.write("id:ID,type:LABEL")
        if include_attributes:
            f.write(",name")
        f.write("\n")

        for node_id, data in kg.graph.nodes(data=True):
            node_type = data.get("node_type", "Node")
            line = f'"{node_id}","{node_type}"'
            if include_attributes:
                name = data.get("name", node_id)
                line += f',"{str(name).replace('"', '""')}"'
            f.write(line + "\n")

    files["nodes"] = str(nodes_path)

    # Export edges
    edges_path = output_path / "edges.csv"
    with open(edges_path, "w") as f:
        f.write(":START_ID,:END_ID,:TYPE,weight:float\n")

        for source, target, data in kg.graph.edges(data=True):
            edge_type = data.get("edge_type", "RELATES_TO")
            weight = data.get("weight", 1.0)
            f.write(f'"{source}","{target}","{edge_type}",{weight}\n')

    files["edges"] = str(edges_path)

    logger.info(f"Exported CSV to {output_dir}")
    return files


def to_adjacency_matrix(
    kg: KnowledgeGraph,
    edge_types: Optional[List[EdgeType]] = None,
    sparse: bool = True,
) -> Tuple[Any, NodeMapping]:
    """
    Export knowledge graph to adjacency matrix.

    Args:
        kg: Knowledge graph
        edge_types: Optional list of edge types to include
        sparse: Whether to return sparse matrix

    Returns:
        Tuple of (adjacency matrix, NodeMapping)
    """
    try:
        import scipy.sparse as sp
    except ImportError:
        raise ImportError("scipy is required for adjacency matrix export")

    node_mapping = create_node_mapping(kg)
    n = len(node_mapping)

    rows = []
    cols = []
    weights = []

    for source, target, data in kg.graph.edges(data=True):
        if edge_types is not None:
            edge_type_str = data.get("edge_type")
            if not any(et.value == edge_type_str for et in edge_types):
                continue

        i = node_mapping.get_idx(source)
        j = node_mapping.get_idx(target)
        w = data.get("weight", 1.0)

        rows.append(i)
        cols.append(j)
        weights.append(w)

    if sparse:
        adj = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
    else:
        adj = np.zeros((n, n))
        for i, j, w in zip(rows, cols, weights):
            adj[i, j] = w

    logger.info(f"Created adjacency matrix: {n}x{n}, {len(rows)} edges")
    return adj, node_mapping


def to_edge_list(
    kg: KnowledgeGraph,
    output_path: Optional[str] = None,
    include_weights: bool = True,
    include_types: bool = True,
) -> List[Tuple]:
    """
    Export knowledge graph to edge list format.

    Args:
        kg: Knowledge graph
        output_path: Optional path to save edge list
        include_weights: Include edge weights
        include_types: Include edge types

    Returns:
        List of edge tuples
    """
    edges = []

    for source, target, data in kg.graph.edges(data=True):
        edge = [source, target]
        if include_weights:
            edge.append(data.get("weight", 1.0))
        if include_types:
            edge.append(data.get("edge_type", "unknown"))
        edges.append(tuple(edge))

    if output_path:
        with open(output_path, "w") as f:
            for edge in edges:
                f.write("\t".join(str(e) for e in edge) + "\n")
        logger.info(f"Saved edge list to {output_path}")

    return edges
