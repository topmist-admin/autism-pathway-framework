"""
Tests for Knowledge Graph Exporters

Tests the export functionality to various formats (CSV, adjacency matrix, etc.).
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from schema import NodeType, EdgeType
from builder import KnowledgeGraph
from exporters import (
    NodeMapping,
    create_node_mapping,
    to_csv,
    to_adjacency_matrix,
    to_edge_list,
    to_neo4j_cypher,
)


@pytest.fixture
def sample_graph():
    """Create a sample knowledge graph for testing."""
    kg = KnowledgeGraph()
    kg.add_nodes(["SHANK3", "CHD8", "SCN2A"], NodeType.GENE)
    kg.add_node("synaptic", NodeType.PATHWAY)
    kg.add_edge("SHANK3", "CHD8", EdgeType.GENE_INTERACTS, weight=0.9)
    kg.add_edge("CHD8", "SCN2A", EdgeType.GENE_INTERACTS, weight=0.7)
    kg.add_edge("SHANK3", "synaptic", EdgeType.GENE_IN_PATHWAY)
    return kg


class TestNodeMapping:
    """Tests for NodeMapping."""

    def test_create_node_mapping(self, sample_graph):
        mapping = create_node_mapping(sample_graph)

        assert len(mapping) == 4
        assert mapping.get_node(mapping.get_idx("SHANK3")) == "SHANK3"
        assert mapping.get_type("SHANK3") == NodeType.GENE
        assert mapping.get_type("synaptic") == NodeType.PATHWAY

    def test_save_and_load_mapping(self, sample_graph):
        mapping = create_node_mapping(sample_graph)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mapping.json"
            mapping.save(str(path))

            loaded = NodeMapping.load(str(path))
            assert len(loaded) == len(mapping)
            assert loaded.get_idx("SHANK3") == mapping.get_idx("SHANK3")


class TestCSVExport:
    """Tests for CSV export."""

    def test_export_to_csv(self, sample_graph):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = to_csv(sample_graph, tmpdir)

            assert "nodes" in files
            assert "edges" in files

            # Check nodes file
            with open(files["nodes"], "r") as f:
                lines = f.readlines()
                assert len(lines) == 5  # header + 4 nodes

            # Check edges file
            with open(files["edges"], "r") as f:
                lines = f.readlines()
                assert len(lines) == 4  # header + 3 edges


class TestAdjacencyMatrix:
    """Tests for adjacency matrix export."""

    def test_sparse_adjacency_matrix(self, sample_graph):
        adj, mapping = to_adjacency_matrix(sample_graph, sparse=True)

        assert adj.shape == (4, 4)
        assert adj.nnz == 3  # 3 edges

        # Check specific edge
        shank3_idx = mapping.get_idx("SHANK3")
        chd8_idx = mapping.get_idx("CHD8")
        assert adj[shank3_idx, chd8_idx] == pytest.approx(0.9)

    def test_dense_adjacency_matrix(self, sample_graph):
        adj, mapping = to_adjacency_matrix(sample_graph, sparse=False)

        assert isinstance(adj, np.ndarray)
        assert adj.shape == (4, 4)

    def test_filter_by_edge_type(self, sample_graph):
        # Only PPI edges
        adj, mapping = to_adjacency_matrix(
            sample_graph,
            edge_types=[EdgeType.GENE_INTERACTS],
        )

        assert adj.nnz == 2  # Only 2 PPI edges


class TestEdgeList:
    """Tests for edge list export."""

    def test_edge_list(self, sample_graph):
        edges = to_edge_list(sample_graph)

        assert len(edges) == 3
        assert all(len(e) == 4 for e in edges)  # (src, dst, weight, type)

    def test_edge_list_no_weights(self, sample_graph):
        edges = to_edge_list(sample_graph, include_weights=False)

        assert len(edges) == 3
        assert all(len(e) == 3 for e in edges)  # (src, dst, type)

    def test_edge_list_to_file(self, sample_graph):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "edges.tsv"
            edges = to_edge_list(sample_graph, output_path=str(path))

            assert path.exists()
            with open(path, "r") as f:
                lines = f.readlines()
                assert len(lines) == 3


class TestNeo4jExport:
    """Tests for Neo4j Cypher export."""

    def test_neo4j_cypher_export(self, sample_graph):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.cypher"
            to_neo4j_cypher(sample_graph, str(path))

            assert path.exists()
            with open(path, "r") as f:
                content = f.read()
                assert "CREATE" in content
                assert "SHANK3" in content
                assert "CREATE INDEX" in content


class TestDGLExport:
    """Tests for DGL export (requires DGL)."""

    @pytest.mark.skipif(
        True,  # Skip by default since DGL is optional
        reason="DGL not installed",
    )
    def test_dgl_export(self, sample_graph):
        from exporters import to_dgl

        g, mapping = to_dgl(sample_graph)

        assert g.num_nodes() == 4
        assert g.num_edges() == 3


class TestPyGExport:
    """Tests for PyG export (requires PyG)."""

    @pytest.mark.skipif(
        True,  # Skip by default since PyG is optional
        reason="PyG not installed",
    )
    def test_pyg_export(self, sample_graph):
        from exporters import to_pyg

        data, mapping = to_pyg(sample_graph)

        # Check node counts
        total_nodes = sum(data[nt].num_nodes for nt in data.node_types)
        assert total_nodes == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
