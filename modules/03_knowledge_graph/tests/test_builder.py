"""
Tests for Knowledge Graph Builder

Tests the construction and manipulation of biological knowledge graphs.
"""

import pytest
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Set

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from schema import NodeType, EdgeType, GraphSchema, Node, Edge
from builder import (
    KnowledgeGraph,
    KnowledgeGraphBuilder,
    PPINetwork,
    load_ppi_from_file,
)


# Mock PathwayDatabase for testing (mimics Module 01 interface)
@dataclass
class MockPathwayDatabase:
    """Mock PathwayDatabase for testing."""

    pathways: Dict[str, Set[str]]
    pathway_names: Dict[str, str]
    pathway_descriptions: Dict[str, str] = field(default_factory=dict)
    source: str = "Test"


class TestNodeType:
    """Tests for NodeType enum."""

    def test_node_types_exist(self):
        assert NodeType.GENE is not None
        assert NodeType.PATHWAY is not None
        assert NodeType.GO_TERM is not None

    def test_node_type_values(self):
        assert NodeType.GENE.value == "gene"
        assert NodeType.PATHWAY.value == "pathway"
        assert NodeType.GO_TERM.value == "go_term"


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_edge_types_exist(self):
        assert EdgeType.GENE_INTERACTS is not None
        assert EdgeType.GENE_IN_PATHWAY is not None
        assert EdgeType.GENE_HAS_GO is not None

    def test_edge_type_values(self):
        assert EdgeType.GENE_INTERACTS.value == "gene_interacts_gene"
        assert EdgeType.GENE_IN_PATHWAY.value == "gene_in_pathway"


class TestGraphSchema:
    """Tests for GraphSchema."""

    def test_default_schema(self):
        schema = GraphSchema.default_schema()
        assert NodeType.GENE in schema.node_types
        assert NodeType.PATHWAY in schema.node_types
        assert EdgeType.GENE_IN_PATHWAY in schema.edge_types

    def test_valid_edge_check(self):
        schema = GraphSchema.default_schema()

        # Valid: GENE -> PATHWAY with GENE_IN_PATHWAY
        assert schema.is_valid_edge(
            NodeType.GENE, NodeType.PATHWAY, EdgeType.GENE_IN_PATHWAY
        )

        # Invalid: PATHWAY -> GENE with GENE_IN_PATHWAY
        assert not schema.is_valid_edge(
            NodeType.PATHWAY, NodeType.GENE, EdgeType.GENE_IN_PATHWAY
        )


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph."""

    def test_create_empty_graph(self):
        kg = KnowledgeGraph()
        assert kg.n_nodes == 0
        assert kg.n_edges == 0

    def test_add_node(self):
        kg = KnowledgeGraph()
        kg.add_node("SHANK3", NodeType.GENE)
        assert kg.n_nodes == 1
        assert kg.has_node("SHANK3")
        assert kg.get_node_type("SHANK3") == NodeType.GENE

    def test_add_node_with_attributes(self):
        kg = KnowledgeGraph()
        kg.add_node("SHANK3", NodeType.GENE, {"pli": 0.99, "loeuf": 0.2})
        node = kg.get_node("SHANK3")
        assert node is not None
        assert node.attributes["pli"] == 0.99
        assert node.attributes["loeuf"] == 0.2

    def test_add_multiple_nodes(self):
        kg = KnowledgeGraph()
        genes = ["SHANK3", "CHD8", "SCN2A"]
        kg.add_nodes(genes, NodeType.GENE)
        assert kg.n_nodes == 3
        for gene in genes:
            assert kg.has_node(gene)

    def test_add_edge(self):
        kg = KnowledgeGraph()
        kg.add_node("SHANK3", NodeType.GENE)
        kg.add_node("synaptic_pathway", NodeType.PATHWAY)
        kg.add_edge("SHANK3", "synaptic_pathway", EdgeType.GENE_IN_PATHWAY)
        assert kg.n_edges == 1
        assert kg.has_edge("SHANK3", "synaptic_pathway")

    def test_add_edge_with_weight(self):
        kg = KnowledgeGraph()
        kg.add_node("SHANK3", NodeType.GENE)
        kg.add_node("CHD8", NodeType.GENE)
        kg.add_edge("SHANK3", "CHD8", EdgeType.GENE_INTERACTS, weight=0.95)
        edges = kg.get_edges_by_type(EdgeType.GENE_INTERACTS)
        assert len(edges) == 1
        assert edges[0][2] == 0.95

    def test_add_edge_missing_node(self):
        kg = KnowledgeGraph()
        kg.add_node("SHANK3", NodeType.GENE)
        with pytest.raises(ValueError):
            kg.add_edge("SHANK3", "missing_pathway", EdgeType.GENE_IN_PATHWAY)

    def test_add_invalid_edge_type(self):
        kg = KnowledgeGraph()
        kg.add_node("SHANK3", NodeType.GENE)
        kg.add_node("CHD8", NodeType.GENE)
        # GENE_IN_PATHWAY is not valid for GENE -> GENE
        with pytest.raises(ValueError):
            kg.add_edge("SHANK3", "CHD8", EdgeType.GENE_IN_PATHWAY)

    def test_get_neighbors(self):
        kg = KnowledgeGraph()
        kg.add_node("SHANK3", NodeType.GENE)
        kg.add_node("CHD8", NodeType.GENE)
        kg.add_node("SCN2A", NodeType.GENE)
        kg.add_edge("SHANK3", "CHD8", EdgeType.GENE_INTERACTS)
        kg.add_edge("SHANK3", "SCN2A", EdgeType.GENE_INTERACTS)

        neighbors = kg.get_neighbors("SHANK3")
        assert len(neighbors) == 2
        assert "CHD8" in neighbors
        assert "SCN2A" in neighbors

    def test_get_neighbors_by_edge_type(self):
        kg = KnowledgeGraph()
        kg.add_node("SHANK3", NodeType.GENE)
        kg.add_node("CHD8", NodeType.GENE)
        kg.add_node("synaptic", NodeType.PATHWAY)
        kg.add_edge("SHANK3", "CHD8", EdgeType.GENE_INTERACTS)
        kg.add_edge("SHANK3", "synaptic", EdgeType.GENE_IN_PATHWAY)

        # Filter by edge type
        ppi_neighbors = kg.get_neighbors("SHANK3", edge_type=EdgeType.GENE_INTERACTS)
        pathway_neighbors = kg.get_neighbors("SHANK3", edge_type=EdgeType.GENE_IN_PATHWAY)

        assert len(ppi_neighbors) == 1
        assert "CHD8" in ppi_neighbors
        assert len(pathway_neighbors) == 1
        assert "synaptic" in pathway_neighbors

    def test_get_nodes_by_type(self):
        kg = KnowledgeGraph()
        kg.add_nodes(["SHANK3", "CHD8"], NodeType.GENE)
        kg.add_node("synaptic", NodeType.PATHWAY)

        genes = kg.get_nodes_by_type(NodeType.GENE)
        pathways = kg.get_nodes_by_type(NodeType.PATHWAY)

        assert len(genes) == 2
        assert len(pathways) == 1

    def test_get_stats(self):
        kg = KnowledgeGraph()
        kg.add_nodes(["SHANK3", "CHD8", "SCN2A"], NodeType.GENE)
        kg.add_node("synaptic", NodeType.PATHWAY)
        kg.add_edge("SHANK3", "CHD8", EdgeType.GENE_INTERACTS)
        kg.add_edge("SHANK3", "synaptic", EdgeType.GENE_IN_PATHWAY)

        stats = kg.get_stats()
        assert stats.n_nodes == 4
        assert stats.n_edges == 2
        assert stats.node_type_counts["gene"] == 3
        assert stats.node_type_counts["pathway"] == 1

    def test_subgraph(self):
        kg = KnowledgeGraph()
        kg.add_nodes(["SHANK3", "CHD8", "SCN2A"], NodeType.GENE)
        kg.add_edge("SHANK3", "CHD8", EdgeType.GENE_INTERACTS)
        kg.add_edge("CHD8", "SCN2A", EdgeType.GENE_INTERACTS)

        sub = kg.subgraph(["SHANK3", "CHD8"])
        assert sub.n_nodes == 2
        assert sub.n_edges == 1  # Only SHANK3->CHD8

    def test_save_and_load_gpickle(self):
        kg = KnowledgeGraph()
        kg.add_nodes(["SHANK3", "CHD8"], NodeType.GENE)
        kg.add_edge("SHANK3", "CHD8", EdgeType.GENE_INTERACTS)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gpickle"
            kg.save(str(path))

            loaded = KnowledgeGraph.load(str(path))
            assert loaded.n_nodes == 2
            assert loaded.n_edges == 1
            assert loaded.has_node("SHANK3")

    def test_save_and_load_json(self):
        kg = KnowledgeGraph()
        kg.add_nodes(["SHANK3", "CHD8"], NodeType.GENE)
        kg.add_edge("SHANK3", "CHD8", EdgeType.GENE_INTERACTS)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            kg.save(str(path))

            loaded = KnowledgeGraph.load(str(path))
            assert loaded.n_nodes == 2
            assert loaded.n_edges == 1


class TestKnowledgeGraphBuilder:
    """Tests for KnowledgeGraphBuilder."""

    def test_add_genes(self):
        builder = KnowledgeGraphBuilder()
        builder.add_genes(["SHANK3", "CHD8", "SCN2A"])
        kg = builder.build()
        assert kg.n_nodes == 3
        assert all(kg.get_node_type(g) == NodeType.GENE for g in ["SHANK3", "CHD8", "SCN2A"])

    def test_add_pathways(self):
        builder = KnowledgeGraphBuilder()

        pathway_db = MockPathwayDatabase(
            pathways={
                "R-HSA-1": {"SHANK3", "NRXN1"},
                "R-HSA-2": {"CHD8", "ADNP"},
            },
            pathway_names={
                "R-HSA-1": "Synaptic transmission",
                "R-HSA-2": "Chromatin regulation",
            },
        )

        builder.add_pathways(pathway_db)
        kg = builder.build()

        # Check pathways exist
        assert kg.has_node("R-HSA-1")
        assert kg.has_node("R-HSA-2")
        assert kg.get_node_type("R-HSA-1") == NodeType.PATHWAY

        # Check genes exist
        assert kg.has_node("SHANK3")
        assert kg.has_node("CHD8")

        # Check gene-pathway edges
        assert kg.has_edge("SHANK3", "R-HSA-1")
        assert kg.has_edge("CHD8", "R-HSA-2")

    def test_add_go_terms(self):
        builder = KnowledgeGraphBuilder()

        go_terms = {
            "GO:0007268": {"name": "synaptic transmission", "namespace": "biological_process"},
            "GO:0006355": {"name": "regulation of transcription", "namespace": "biological_process"},
        }

        annotations = {
            "SHANK3": {"GO:0007268"},
            "CHD8": {"GO:0006355"},
        }

        builder.add_go_terms(go_terms, annotations)
        kg = builder.build()

        # Check GO terms exist
        assert kg.has_node("GO:0007268")
        assert kg.get_node_type("GO:0007268") == NodeType.GO_TERM

        # Check gene-GO edges
        assert kg.has_edge("SHANK3", "GO:0007268", EdgeType.GENE_HAS_GO)

    def test_add_ppi(self):
        builder = KnowledgeGraphBuilder()

        ppi = PPINetwork(
            interactions=[
                ("SHANK3", "NRXN1", 950),
                ("SHANK3", "NLGN1", 800),
                ("CHD8", "ADNP", 700),
            ],
            source="STRING",
        )

        builder.add_genes(["SHANK3", "NRXN1", "NLGN1", "CHD8", "ADNP"])
        builder.add_ppi(ppi)
        kg = builder.build()

        # Check PPI edges (normalized weights)
        assert kg.has_edge("SHANK3", "NRXN1", EdgeType.GENE_INTERACTS)
        assert kg.has_edge("SHANK3", "NLGN1", EdgeType.GENE_INTERACTS)
        assert kg.has_edge("CHD8", "ADNP", EdgeType.GENE_INTERACTS)

    def test_add_go_hierarchy(self):
        builder = KnowledgeGraphBuilder()

        go_terms = {
            "GO:0007268": {"name": "synaptic transmission", "namespace": "biological_process"},
            "GO:0050877": {"name": "nervous system process", "namespace": "biological_process"},
        }

        # GO:0007268 is_a GO:0050877
        hierarchy = [("GO:0007268", "GO:0050877", "is_a")]

        builder.add_go_terms(go_terms, {})
        builder.add_go_hierarchy(hierarchy)
        kg = builder.build()

        assert kg.has_edge("GO:0007268", "GO:0050877", EdgeType.GO_IS_A)

    def test_builder_chaining(self):
        pathway_db = MockPathwayDatabase(
            pathways={"R-HSA-1": {"SHANK3", "CHD8"}},
            pathway_names={"R-HSA-1": "Test pathway"},
        )

        ppi = PPINetwork(
            interactions=[("SHANK3", "CHD8", 900)],
            source="STRING",
        )

        kg = (
            KnowledgeGraphBuilder()
            .add_genes(["SHANK3", "CHD8"])
            .add_pathways(pathway_db)
            .add_ppi(ppi)
            .build()
        )

        assert kg.n_nodes > 0
        assert kg.has_edge("SHANK3", "CHD8", EdgeType.GENE_INTERACTS)

    def test_builder_clear(self):
        builder = KnowledgeGraphBuilder()
        builder.add_genes(["SHANK3", "CHD8"])

        builder.clear()
        kg = builder.build()

        assert kg.n_nodes == 0


class TestPPINetwork:
    """Tests for PPINetwork."""

    def test_create_ppi_network(self):
        ppi = PPINetwork(
            interactions=[
                ("SHANK3", "NRXN1", 950),
                ("SHANK3", "NLGN1", 800),
            ],
            source="STRING",
        )
        assert len(ppi) == 2

    def test_filter_by_score(self):
        ppi = PPINetwork(
            interactions=[
                ("SHANK3", "NRXN1", 950),
                ("SHANK3", "NLGN1", 600),
                ("CHD8", "ADNP", 400),
            ],
            source="STRING",
        )

        filtered = ppi.filter_by_score(700)
        assert len(filtered) == 1
        assert filtered.score_threshold == 700


class TestLoadPPIFromFile:
    """Tests for load_ppi_from_file function."""

    def test_load_ppi_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock STRING file
            path = Path(tmpdir) / "string.txt"
            with open(path, "w") as f:
                f.write("protein1\tprotein2\tcombined_score\n")
                f.write("9606.SHANK3\t9606.NRXN1\t950\n")
                f.write("9606.CHD8\t9606.ADNP\t700\n")
                f.write("9606.GENE1\t9606.GENE2\t300\n")

            ppi = load_ppi_from_file(str(path), min_score=400)

            # Should filter out the 300 score edge
            assert len(ppi) == 2
            assert ppi.score_threshold == 400


class TestKnowledgeGraphIntegration:
    """Integration tests for knowledge graph construction."""

    def test_build_complete_graph(self):
        """Test building a complete knowledge graph with all data types."""
        # Setup data
        pathway_db = MockPathwayDatabase(
            pathways={
                "R-HSA-SYNAPTIC": {"SHANK3", "NRXN1", "NLGN1"},
                "R-HSA-CHROMATIN": {"CHD8", "ADNP"},
            },
            pathway_names={
                "R-HSA-SYNAPTIC": "Synaptic signaling",
                "R-HSA-CHROMATIN": "Chromatin remodeling",
            },
        )

        go_terms = {
            "GO:0007268": {"name": "synaptic transmission", "namespace": "BP"},
            "GO:0006355": {"name": "transcription regulation", "namespace": "BP"},
        }

        annotations = {
            "SHANK3": {"GO:0007268"},
            "CHD8": {"GO:0006355"},
        }

        ppi = PPINetwork(
            interactions=[
                ("SHANK3", "NRXN1", 950),
                ("NRXN1", "NLGN1", 850),
                ("CHD8", "ADNP", 700),
            ],
            source="STRING",
        )

        # Build graph
        kg = (
            KnowledgeGraphBuilder()
            .add_pathways(pathway_db)
            .add_go_terms(go_terms, annotations)
            .add_ppi(ppi)
            .build()
        )

        # Verify structure
        stats = kg.get_stats()

        # 5 genes + 2 pathways + 2 GO terms = 9 nodes
        assert stats.n_nodes == 9

        # Gene-pathway edges + Gene-GO edges + PPI edges
        # 3 genes in synaptic + 2 in chromatin = 5 gene-pathway edges
        # 2 gene-GO edges (SHANK3, CHD8)
        # 3 PPI edges
        assert stats.n_edges == 10

        # Check node types
        assert stats.node_type_counts["gene"] == 5
        assert stats.node_type_counts["pathway"] == 2
        assert stats.node_type_counts["go_term"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
