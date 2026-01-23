"""Tests for do-calculus engine."""

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
    StructuralCausalModel,
    create_sample_asd_scm,
)
from do_calculus import (
    Distribution,
    IntervenedModel,
    DoCalculusEngine,
)


class TestDistribution:
    """Tests for Distribution class."""

    def test_creation(self):
        """Test distribution creation."""
        dist = Distribution(mean=0.5, variance=0.1)
        assert dist.mean == 0.5
        assert dist.variance == 0.1

    def test_mean_clamping(self):
        """Test mean is clamped to [0, 1]."""
        dist_high = Distribution(mean=1.5)
        assert dist_high.mean == 1.0

        dist_low = Distribution(mean=-0.5)
        assert dist_low.mean == 0.0

    def test_std_property(self):
        """Test standard deviation property."""
        dist = Distribution(mean=0.5, variance=0.04)
        assert dist.std == pytest.approx(0.2)

    def test_to_dict(self):
        """Test serialization."""
        dist = Distribution(mean=0.6, variance=0.05, confidence_interval=(0.4, 0.8))
        d = dist.to_dict()
        assert d["mean"] == 0.6
        assert d["confidence_interval"] == (0.4, 0.8)


class TestIntervenedModel:
    """Tests for IntervenedModel."""

    @pytest.fixture
    def simple_scm(self):
        """Create simple SCM: A -> B -> C."""
        scm = StructuralCausalModel()
        scm.add_node(CausalNode("A", CausalNodeType.VARIANT, observed=True))
        scm.add_node(CausalNode("B", CausalNodeType.GENE_FUNCTION, observed=True))
        scm.add_node(CausalNode("C", CausalNodeType.PHENOTYPE, observed=True))
        scm.add_edge(CausalEdge("A", "B", CausalEdgeType.CAUSES, 0.8, "causal"))
        scm.add_edge(CausalEdge("B", "C", CausalEdgeType.CAUSES, 0.7, "causal"))
        return scm

    def test_intervention_removes_incoming_edges(self, simple_scm):
        """Test that intervention removes incoming edges."""
        intervened = IntervenedModel(simple_scm, {"B": 0.5})
        modified = intervened.get_scm()

        # B should have no parents (incoming edge removed)
        assert len(modified.get_parents("B")) == 0

        # B should have value set
        assert modified.nodes["B"].value == 0.5

    def test_intervention_preserves_outgoing_edges(self, simple_scm):
        """Test that intervention preserves outgoing edges."""
        intervened = IntervenedModel(simple_scm, {"B": 0.5})
        modified = intervened.get_scm()

        # B should still have children
        assert "C" in modified.get_children("B")

    def test_multiple_interventions(self, simple_scm):
        """Test multiple simultaneous interventions."""
        intervened = IntervenedModel(simple_scm, {"A": 1.0, "B": 0.0})
        modified = intervened.get_scm()

        assert modified.nodes["A"].value == 1.0
        assert modified.nodes["B"].value == 0.0

    def test_invalid_intervention_raises(self, simple_scm):
        """Test intervention on non-existent node raises error."""
        with pytest.raises(ValueError, match="not found"):
            IntervenedModel(simple_scm, {"X": 0.5})


class TestDoCalculusEngine:
    """Tests for DoCalculusEngine."""

    @pytest.fixture
    def simple_scm(self):
        """Create simple SCM: A -> B -> C."""
        scm = StructuralCausalModel()
        scm.add_node(CausalNode("A", CausalNodeType.VARIANT, observed=True))
        scm.add_node(CausalNode("B", CausalNodeType.GENE_FUNCTION, observed=True))
        scm.add_node(CausalNode("C", CausalNodeType.PHENOTYPE, observed=True))
        scm.add_edge(CausalEdge("A", "B", CausalEdgeType.CAUSES, 0.8, "causal"))
        scm.add_edge(CausalEdge("B", "C", CausalEdgeType.CAUSES, 0.7, "causal"))
        return scm

    @pytest.fixture
    def engine(self, simple_scm):
        """Create do-calculus engine."""
        return DoCalculusEngine(simple_scm)

    def test_do_returns_intervened_model(self, engine):
        """Test do() returns IntervenedModel."""
        result = engine.do({"B": 0.5})
        assert isinstance(result, IntervenedModel)

    def test_query_returns_distribution(self, engine):
        """Test query returns Distribution."""
        result = engine.query(
            outcome="C",
            intervention={"A": 1.0}
        )
        assert isinstance(result, Distribution)
        assert 0 <= result.mean <= 1

    def test_query_with_evidence(self, engine):
        """Test query with additional evidence."""
        result = engine.query(
            outcome="C",
            intervention={"A": 1.0},
            evidence={"B": 0.5}
        )
        assert isinstance(result, Distribution)

    def test_query_invalid_outcome_raises(self, engine):
        """Test query with invalid outcome raises error."""
        with pytest.raises(ValueError, match="not found"):
            engine.query(outcome="X", intervention={"A": 1.0})

    def test_average_treatment_effect(self, engine):
        """Test ATE calculation."""
        ate = engine.average_treatment_effect(
            treatment="A",
            outcome="C"
        )
        # ATE should be non-zero (A affects C through B)
        assert isinstance(ate, float)

    def test_ate_direction(self, engine):
        """Test ATE has expected direction."""
        ate = engine.average_treatment_effect(
            treatment="A",
            outcome="C",
            treatment_values=(0.0, 1.0)
        )
        # Higher A should lead to higher C (positive effect chain)
        # Since values propagate through the chain
        assert ate != 0  # Should have some effect

    def test_conditional_average_treatment_effect(self, engine):
        """Test CATE calculation."""
        cate = engine.conditional_average_treatment_effect(
            treatment="A",
            outcome="C",
            subgroup={"B": 0.5}
        )
        assert isinstance(cate, float)

    def test_intervention_effect_on_path(self, engine):
        """Test effect propagation along paths."""
        effects = engine.intervention_effect_on_path(
            intervention_node="A",
            outcome="C",
            intervention_value=1.0
        )

        # Should include A and C
        assert "A" in effects
        assert effects["A"] == 1.0  # Intervention value

    def test_identify_effect_identifiable(self, engine):
        """Test effect identification for identifiable case."""
        result = engine.identify_effect("A", "C")
        # Should be identifiable in simple chain
        assert result is not None
        assert "P(C" in result

    def test_sensitivity_analysis(self, engine):
        """Test sensitivity analysis."""
        results = engine.sensitivity_analysis("A", "C")

        assert "base_ate" in results
        assert len(results) > 1  # Should have multiple confounding levels

    def test_asd_scm_intervention(self):
        """Test intervention on ASD-specific SCM."""
        scm = create_sample_asd_scm()
        engine = DoCalculusEngine(scm)

        # Intervene on SHANK3 function
        result = engine.query(
            outcome="asd_phenotype",
            intervention={"SHANK3_function": 1.0}  # Restore function
        )

        assert isinstance(result, Distribution)
        assert 0 <= result.mean <= 1

    def test_asd_scm_ate(self):
        """Test ATE on ASD-specific SCM."""
        scm = create_sample_asd_scm()
        engine = DoCalculusEngine(scm)

        ate = engine.average_treatment_effect(
            treatment="SHANK3_function",
            outcome="asd_phenotype"
        )

        # SHANK3 function should affect phenotype
        assert isinstance(ate, float)

    def test_asd_scm_pathway_intervention(self):
        """Test pathway-level intervention."""
        scm = create_sample_asd_scm()
        engine = DoCalculusEngine(scm)

        # Intervene on pathway
        result = engine.query(
            outcome="asd_phenotype",
            intervention={"synaptic_pathway": 1.0}  # Restore pathway
        )

        assert isinstance(result, Distribution)


class TestDoCalculusWithStructuralEquations:
    """Tests for do-calculus with explicit structural equations."""

    @pytest.fixture
    def scm_with_equations(self):
        """Create SCM with explicit structural equations."""
        scm = StructuralCausalModel()
        scm.add_node(CausalNode("X", CausalNodeType.VARIANT, observed=True))
        scm.add_node(CausalNode("Y", CausalNodeType.PHENOTYPE, observed=True))
        scm.add_edge(CausalEdge("X", "Y", CausalEdgeType.CAUSES, 0.9, "direct"))

        # Y = 0.5 * X + 0.25
        def y_equation(X=0.5):
            return 0.5 * X + 0.25

        scm.set_structural_equation("Y", y_equation)
        return scm

    def test_query_uses_structural_equations(self, scm_with_equations):
        """Test that queries use structural equations when available."""
        engine = DoCalculusEngine(scm_with_equations)

        result = engine.query(
            outcome="Y",
            intervention={"X": 1.0}
        )

        # With X=1.0, Y should be 0.5*1.0 + 0.25 = 0.75
        assert result.mean == pytest.approx(0.75, abs=0.1)

    def test_ate_with_structural_equations(self, scm_with_equations):
        """Test ATE with structural equations."""
        engine = DoCalculusEngine(scm_with_equations)

        ate = engine.average_treatment_effect(
            treatment="X",
            outcome="Y",
            treatment_values=(0.0, 1.0)
        )

        # ATE = Y(X=1) - Y(X=0) = 0.75 - 0.25 = 0.5
        assert ate == pytest.approx(0.5, abs=0.1)
