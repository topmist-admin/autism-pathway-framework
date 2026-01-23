"""Tests for Structural Causal Model."""

import sys
from pathlib import Path

# Add module to path
_module_dir = Path(__file__).parent.parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

import pytest
from causal_graph import (
    CausalNodeType,
    CausalEdgeType,
    CausalNode,
    CausalEdge,
    CausalQuery,
    CausalQueryBuilder,
    StructuralCausalModel,
    create_sample_asd_scm,
)


class TestCausalNodeType:
    """Tests for CausalNodeType enum."""

    def test_all_types_defined(self):
        """Test all node types are defined."""
        types = [
            CausalNodeType.VARIANT,
            CausalNodeType.GENE_FUNCTION,
            CausalNodeType.PATHWAY,
            CausalNodeType.CIRCUIT,
            CausalNodeType.PHENOTYPE,
            CausalNodeType.CONFOUNDER,
        ]
        assert len(types) == 6

    def test_type_values(self):
        """Test type string values."""
        assert CausalNodeType.VARIANT.value == "variant"
        assert CausalNodeType.PHENOTYPE.value == "phenotype"


class TestCausalEdgeType:
    """Tests for CausalEdgeType enum."""

    def test_all_types_defined(self):
        """Test all edge types are defined."""
        types = [
            CausalEdgeType.CAUSES,
            CausalEdgeType.MEDIATES,
            CausalEdgeType.CONFOUNDS,
            CausalEdgeType.MODIFIES,
        ]
        assert len(types) == 4

    def test_type_values(self):
        """Test type string values."""
        assert CausalEdgeType.CAUSES.value == "causes"
        assert CausalEdgeType.CONFOUNDS.value == "confounds"


class TestCausalNode:
    """Tests for CausalNode dataclass."""

    def test_creation(self):
        """Test basic node creation."""
        node = CausalNode(
            id="test_node",
            node_type=CausalNodeType.GENE_FUNCTION,
            observed=True,
            value=0.5,
        )
        assert node.id == "test_node"
        assert node.node_type == CausalNodeType.GENE_FUNCTION
        assert node.observed is True
        assert node.value == 0.5

    def test_empty_id_raises(self):
        """Test that empty id raises ValueError."""
        with pytest.raises(ValueError, match="Node id cannot be empty"):
            CausalNode(id="", node_type=CausalNodeType.VARIANT, observed=True)

    def test_default_metadata(self):
        """Test default metadata is empty dict."""
        node = CausalNode(
            id="test",
            node_type=CausalNodeType.VARIANT,
            observed=True,
        )
        assert node.metadata == {}

    def test_to_dict(self):
        """Test serialization to dictionary."""
        node = CausalNode(
            id="test",
            node_type=CausalNodeType.PATHWAY,
            observed=False,
            value=0.3,
            metadata={"key": "value"},
        )
        d = node.to_dict()
        assert d["id"] == "test"
        assert d["node_type"] == "pathway"
        assert d["observed"] is False
        assert d["value"] == 0.3
        assert d["metadata"]["key"] == "value"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "test",
            "node_type": "phenotype",
            "observed": True,
            "value": 0.8,
        }
        node = CausalNode.from_dict(data)
        assert node.id == "test"
        assert node.node_type == CausalNodeType.PHENOTYPE
        assert node.value == 0.8


class TestCausalEdge:
    """Tests for CausalEdge dataclass."""

    def test_creation(self):
        """Test basic edge creation."""
        edge = CausalEdge(
            source="A",
            target="B",
            edge_type=CausalEdgeType.CAUSES,
            strength=0.8,
            mechanism="direct causation",
        )
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.strength == 0.8

    def test_empty_source_raises(self):
        """Test that empty source raises ValueError."""
        with pytest.raises(ValueError, match="source and target cannot be empty"):
            CausalEdge(
                source="",
                target="B",
                edge_type=CausalEdgeType.CAUSES,
                strength=0.5,
                mechanism="test",
            )

    def test_invalid_strength_raises(self):
        """Test that invalid strength raises ValueError."""
        with pytest.raises(ValueError, match="strength must be between 0 and 1"):
            CausalEdge(
                source="A",
                target="B",
                edge_type=CausalEdgeType.CAUSES,
                strength=1.5,
                mechanism="test",
            )

    def test_to_dict(self):
        """Test serialization to dictionary."""
        edge = CausalEdge(
            source="A",
            target="B",
            edge_type=CausalEdgeType.MEDIATES,
            strength=0.6,
            mechanism="mediation",
        )
        d = edge.to_dict()
        assert d["source"] == "A"
        assert d["edge_type"] == "mediates"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "source": "A",
            "target": "B",
            "edge_type": "confounds",
            "strength": 0.3,
            "mechanism": "confounding",
        }
        edge = CausalEdge.from_dict(data)
        assert edge.edge_type == CausalEdgeType.CONFOUNDS


class TestCausalQuery:
    """Tests for CausalQuery dataclass."""

    def test_creation(self):
        """Test query creation."""
        query = CausalQuery(
            query_type="intervention",
            treatment="T",
            outcome="Y",
            intervention_value=1.0,
        )
        assert query.query_type == "intervention"
        assert query.treatment == "T"

    def test_to_dict(self):
        """Test serialization."""
        query = CausalQuery(
            query_type="effect",
            treatment="T",
            outcome="Y",
            mediator="M",
        )
        d = query.to_dict()
        assert d["mediator"] == "M"


class TestCausalQueryBuilder:
    """Tests for CausalQueryBuilder."""

    def test_build_basic_query(self):
        """Test building basic query."""
        query = (CausalQueryBuilder()
            .treatment("T")
            .outcome("Y")
            .build())

        assert query.treatment == "T"
        assert query.outcome == "Y"
        assert query.query_type == "intervention"

    def test_build_with_do(self):
        """Test building query with do-intervention."""
        query = (CausalQueryBuilder()
            .treatment("T")
            .outcome("Y")
            .do({"T": 1.0})
            .build())

        assert query.intervention_value == 1.0

    def test_build_with_mediation(self):
        """Test building mediation query."""
        query = (CausalQueryBuilder()
            .treatment("T")
            .outcome("Y")
            .mediated_by("M")
            .build())

        assert query.mediator == "M"
        assert query.query_type == "effect"

    def test_build_counterfactual(self):
        """Test building counterfactual query."""
        query = (CausalQueryBuilder()
            .treatment("T")
            .outcome("Y")
            .given({"Z": 0.5})
            .counterfactual()
            .build())

        assert query.query_type == "counterfactual"
        assert query.conditioning == {"Z": 0.5}

    def test_missing_treatment_raises(self):
        """Test that missing treatment raises error."""
        with pytest.raises(ValueError, match="Treatment variable must be specified"):
            CausalQueryBuilder().outcome("Y").build()

    def test_missing_outcome_raises(self):
        """Test that missing outcome raises error."""
        with pytest.raises(ValueError, match="Outcome variable must be specified"):
            CausalQueryBuilder().treatment("T").build()


class TestStructuralCausalModel:
    """Tests for StructuralCausalModel."""

    @pytest.fixture
    def simple_scm(self):
        """Create simple SCM: A -> B -> C."""
        scm = StructuralCausalModel()
        scm.add_node(CausalNode("A", CausalNodeType.VARIANT, observed=True))
        scm.add_node(CausalNode("B", CausalNodeType.GENE_FUNCTION, observed=True))
        scm.add_node(CausalNode("C", CausalNodeType.PHENOTYPE, observed=True))
        scm.add_edge(CausalEdge("A", "B", CausalEdgeType.CAUSES, 0.8, "A causes B"))
        scm.add_edge(CausalEdge("B", "C", CausalEdgeType.CAUSES, 0.7, "B causes C"))
        return scm

    @pytest.fixture
    def fork_scm(self):
        """Create fork SCM: A <- B -> C (B is confounder)."""
        scm = StructuralCausalModel()
        scm.add_node(CausalNode("A", CausalNodeType.GENE_FUNCTION, observed=True))
        scm.add_node(CausalNode("B", CausalNodeType.CONFOUNDER, observed=True))
        scm.add_node(CausalNode("C", CausalNodeType.PHENOTYPE, observed=True))
        scm.add_edge(CausalEdge("B", "A", CausalEdgeType.CONFOUNDS, 0.5, "B confounds A"))
        scm.add_edge(CausalEdge("B", "C", CausalEdgeType.CONFOUNDS, 0.5, "B confounds C"))
        return scm

    @pytest.fixture
    def collider_scm(self):
        """Create collider SCM: A -> B <- C."""
        scm = StructuralCausalModel()
        scm.add_node(CausalNode("A", CausalNodeType.VARIANT, observed=True))
        scm.add_node(CausalNode("B", CausalNodeType.GENE_FUNCTION, observed=True))
        scm.add_node(CausalNode("C", CausalNodeType.VARIANT, observed=True))
        scm.add_edge(CausalEdge("A", "B", CausalEdgeType.CAUSES, 0.6, "A causes B"))
        scm.add_edge(CausalEdge("C", "B", CausalEdgeType.CAUSES, 0.6, "C causes B"))
        return scm

    def test_add_node(self):
        """Test adding nodes."""
        scm = StructuralCausalModel()
        node = CausalNode("test", CausalNodeType.VARIANT, observed=True)
        scm.add_node(node)
        assert "test" in scm.nodes

    def test_add_edge(self, simple_scm):
        """Test adding edges."""
        assert len(simple_scm.edges) == 2

    def test_add_edge_missing_node_raises(self):
        """Test that adding edge with missing node raises error."""
        scm = StructuralCausalModel()
        scm.add_node(CausalNode("A", CausalNodeType.VARIANT, observed=True))

        with pytest.raises(ValueError, match="Target node B not found"):
            scm.add_edge(CausalEdge("A", "B", CausalEdgeType.CAUSES, 0.5, "test"))

    def test_get_parents(self, simple_scm):
        """Test getting parent nodes."""
        parents = simple_scm.get_parents("B")
        assert parents == ["A"]

        parents_c = simple_scm.get_parents("C")
        assert parents_c == ["B"]

    def test_get_children(self, simple_scm):
        """Test getting child nodes."""
        children = simple_scm.get_children("A")
        assert children == ["B"]

    def test_get_ancestors(self, simple_scm):
        """Test getting all ancestors."""
        ancestors = simple_scm.get_ancestors("C")
        assert ancestors == {"A", "B"}

    def test_get_descendants(self, simple_scm):
        """Test getting all descendants."""
        descendants = simple_scm.get_descendants("A")
        assert descendants == {"B", "C"}

    def test_get_edge(self, simple_scm):
        """Test getting specific edge."""
        edge = simple_scm.get_edge("A", "B")
        assert edge is not None
        assert edge.strength == 0.8

        missing = simple_scm.get_edge("A", "C")
        assert missing is None

    def test_remove_edge(self, simple_scm):
        """Test removing edge."""
        result = simple_scm.remove_edge("A", "B")
        assert result is True
        assert simple_scm.get_edge("A", "B") is None

    def test_d_separation_chain(self, simple_scm):
        """Test d-separation in chain A -> B -> C."""
        # A and C are NOT d-separated when nothing is conditioned
        assert not simple_scm.is_d_separated("A", "C", set())

        # A and C ARE d-separated when B is conditioned
        assert simple_scm.is_d_separated("A", "C", {"B"})

    def test_d_separation_fork(self, fork_scm):
        """Test d-separation in fork A <- B -> C."""
        # A and C are NOT d-separated (common cause)
        assert not fork_scm.is_d_separated("A", "C", set())

        # A and C ARE d-separated when B is conditioned
        assert fork_scm.is_d_separated("A", "C", {"B"})

    def test_d_separation_collider(self, collider_scm):
        """Test d-separation in collider A -> B <- C."""
        # A and C ARE d-separated (no common cause or effect)
        assert collider_scm.is_d_separated("A", "C", set())

        # A and C are NOT d-separated when B is conditioned
        assert not collider_scm.is_d_separated("A", "C", {"B"})

    def test_get_backdoor_paths(self):
        """Test finding backdoor paths."""
        # Create SCM with backdoor path: A -> B -> C with A <- D -> C
        scm = StructuralCausalModel()
        scm.add_node(CausalNode("A", CausalNodeType.VARIANT, observed=True))
        scm.add_node(CausalNode("B", CausalNodeType.PATHWAY, observed=True))
        scm.add_node(CausalNode("C", CausalNodeType.PHENOTYPE, observed=True))
        scm.add_node(CausalNode("D", CausalNodeType.CONFOUNDER, observed=True))

        scm.add_edge(CausalEdge("A", "B", CausalEdgeType.CAUSES, 0.7, "causal"))
        scm.add_edge(CausalEdge("B", "C", CausalEdgeType.CAUSES, 0.7, "causal"))
        scm.add_edge(CausalEdge("D", "A", CausalEdgeType.CONFOUNDS, 0.5, "confound"))
        scm.add_edge(CausalEdge("D", "C", CausalEdgeType.CONFOUNDS, 0.5, "confound"))

        backdoors = scm.get_backdoor_paths("A", "C")
        # Should find path through D
        assert len(backdoors) >= 1

    def test_get_valid_adjustment_sets(self):
        """Test finding valid adjustment sets."""
        # Simple case: A -> B -> C with confounder D -> A, D -> C
        scm = StructuralCausalModel()
        scm.add_node(CausalNode("A", CausalNodeType.VARIANT, observed=True))
        scm.add_node(CausalNode("B", CausalNodeType.PATHWAY, observed=True))
        scm.add_node(CausalNode("C", CausalNodeType.PHENOTYPE, observed=True))
        scm.add_node(CausalNode("D", CausalNodeType.CONFOUNDER, observed=True))

        scm.add_edge(CausalEdge("A", "B", CausalEdgeType.CAUSES, 0.7, "causal"))
        scm.add_edge(CausalEdge("B", "C", CausalEdgeType.CAUSES, 0.7, "causal"))
        scm.add_edge(CausalEdge("D", "A", CausalEdgeType.CONFOUNDS, 0.5, "confound"))
        scm.add_edge(CausalEdge("D", "C", CausalEdgeType.CONFOUNDS, 0.5, "confound"))

        adj_sets = scm.get_valid_adjustment_sets("A", "C")
        # D should be in a valid adjustment set
        assert len(adj_sets) >= 1
        # At least one set should contain D
        has_d = any("D" in s for s in adj_sets)
        assert has_d

    def test_copy(self, simple_scm):
        """Test copying the SCM."""
        copy = simple_scm.copy()

        assert len(copy.nodes) == len(simple_scm.nodes)
        assert len(copy.edges) == len(simple_scm.edges)

        # Modifying copy shouldn't affect original
        copy.nodes["A"].value = 0.9
        assert simple_scm.nodes["A"].value is None

    def test_to_dict_and_from_dict(self, simple_scm):
        """Test serialization round-trip."""
        d = simple_scm.to_dict()
        restored = StructuralCausalModel.from_dict(d)

        assert len(restored.nodes) == len(simple_scm.nodes)
        assert len(restored.edges) == len(simple_scm.edges)

    def test_get_statistics(self, simple_scm):
        """Test getting model statistics."""
        stats = simple_scm.get_statistics()

        assert stats["total_nodes"] == 3
        assert stats["total_edges"] == 2

    def test_set_structural_equation(self, simple_scm):
        """Test setting structural equations."""
        def b_equation(A=0.5):
            return A * 0.8

        simple_scm.set_structural_equation("B", b_equation)
        assert "B" in simple_scm.structural_equations


class TestCreateSampleAsdScm:
    """Tests for sample ASD SCM."""

    def test_create_sample(self):
        """Test creating sample SCM."""
        scm = create_sample_asd_scm()

        assert len(scm.nodes) > 0
        assert len(scm.edges) > 0

    def test_sample_has_expected_nodes(self):
        """Test sample has expected node types."""
        scm = create_sample_asd_scm()

        # Should have variants
        variants = [n for n in scm.nodes.values()
                   if n.node_type == CausalNodeType.VARIANT]
        assert len(variants) >= 3

        # Should have phenotype
        phenotypes = [n for n in scm.nodes.values()
                     if n.node_type == CausalNodeType.PHENOTYPE]
        assert len(phenotypes) >= 1

    def test_sample_has_causal_chain(self):
        """Test sample has proper causal chain."""
        scm = create_sample_asd_scm()

        # SHANK3_variant should have path to asd_phenotype
        descendants = scm.get_descendants("SHANK3_variant")
        assert "asd_phenotype" in descendants

    def test_sample_has_confounders(self):
        """Test sample has confounders."""
        scm = create_sample_asd_scm()

        confounders = [n for n in scm.nodes.values()
                      if n.node_type == CausalNodeType.CONFOUNDER]
        assert len(confounders) >= 1
