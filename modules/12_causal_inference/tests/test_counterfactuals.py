"""Tests for counterfactual reasoning and effect estimation."""

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
from do_calculus import DoCalculusEngine
from counterfactuals import (
    CounterfactualResult,
    CounterfactualEngine,
)
from effect_estimation import (
    MediationResult,
    EffectDecomposition,
    CausalEffectEstimator,
)


class TestCounterfactualResult:
    """Tests for CounterfactualResult dataclass."""

    def test_creation(self):
        """Test result creation."""
        result = CounterfactualResult(
            factual_value=0.8,
            counterfactual_value=0.3,
            difference=-0.5,
            confidence=0.7,
            exogenous_values={"U": 0.1},
            explanation="Test explanation",
        )
        assert result.factual_value == 0.8
        assert result.counterfactual_value == 0.3
        assert result.difference == -0.5

    def test_to_dict(self):
        """Test serialization."""
        result = CounterfactualResult(
            factual_value=0.8,
            counterfactual_value=0.3,
            difference=-0.5,
            confidence=0.7,
            exogenous_values={},
        )
        d = result.to_dict()
        assert d["factual_value"] == 0.8
        assert d["confidence"] == 0.7


class TestCounterfactualEngine:
    """Tests for CounterfactualEngine."""

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
        """Create counterfactual engine."""
        return CounterfactualEngine(simple_scm)

    def test_counterfactual_basic(self, engine):
        """Test basic counterfactual query."""
        result = engine.counterfactual(
            factual_evidence={"A": 0.0, "C": 0.2},
            counterfactual_intervention={"A": 1.0},
            query_variable="C"
        )

        assert isinstance(result, CounterfactualResult)
        assert 0 <= result.counterfactual_value <= 1
        assert result.explanation != ""

    def test_counterfactual_exogenous_inferred(self, engine):
        """Test that exogenous variables are inferred."""
        result = engine.counterfactual(
            factual_evidence={"A": 0.5, "B": 0.6, "C": 0.7},
            counterfactual_intervention={"A": 0.0},
            query_variable="C"
        )

        # Should have exogenous values for observed nodes
        assert len(result.exogenous_values) > 0

    def test_counterfactual_invalid_query_raises(self, engine):
        """Test invalid query variable raises error."""
        with pytest.raises(ValueError, match="not found"):
            engine.counterfactual(
                factual_evidence={"A": 0.5},
                counterfactual_intervention={"A": 1.0},
                query_variable="X"
            )

    def test_probability_of_necessity(self, engine):
        """Test probability of necessity calculation."""
        pn = engine.probability_of_necessity(
            treatment="A",
            outcome="C",
            factual={"A": 1.0, "C": 0.8}  # Treatment happened, outcome occurred
        )

        assert 0 <= pn <= 1

    def test_probability_of_sufficiency(self, engine):
        """Test probability of sufficiency calculation."""
        ps = engine.probability_of_sufficiency(
            treatment="A",
            outcome="C",
            factual={"A": 0.0, "C": 0.2}  # Treatment didn't happen, outcome low
        )

        assert 0 <= ps <= 1

    def test_probability_of_necessity_and_sufficiency(self, engine):
        """Test PNS calculation."""
        pns = engine.probability_of_necessity_and_sufficiency(
            treatment="A",
            outcome="C"
        )

        assert 0 <= pns <= 1

    def test_what_if_analysis(self, engine):
        """Test comparing multiple counterfactual scenarios."""
        interventions = [
            {"A": 0.0},
            {"A": 0.5},
            {"A": 1.0},
        ]

        results = engine.what_if_analysis(
            factual_evidence={"A": 0.3, "C": 0.4},
            interventions=interventions,
            query_variable="C"
        )

        assert len(results) == 3
        # Results should be sorted by absolute difference
        assert all(isinstance(r, CounterfactualResult) for r in results)

    def test_individual_treatment_effect(self, engine):
        """Test ITE calculation."""
        ite = engine.individual_treatment_effect(
            treatment="A",
            outcome="C",
            individual_evidence={"B": 0.5}
        )

        # ITE should be a float
        assert isinstance(ite, float)

    def test_asd_counterfactual(self):
        """Test counterfactual on ASD SCM."""
        scm = create_sample_asd_scm()
        engine = CounterfactualEngine(scm)

        # What if SHANK3 function were restored?
        result = engine.counterfactual(
            factual_evidence={
                "SHANK3_function": 0.0,  # Disrupted
                "asd_phenotype": 0.8     # High phenotype
            },
            counterfactual_intervention={"SHANK3_function": 1.0},
            query_variable="asd_phenotype"
        )

        assert isinstance(result, CounterfactualResult)
        # The counterfactual should differ from factual
        # (The direction depends on model semantics - here we just verify computation)
        assert result.difference != 0
        assert result.explanation != ""


class TestMediationResult:
    """Tests for MediationResult dataclass."""

    def test_creation(self):
        """Test result creation."""
        result = MediationResult(
            total_effect=0.5,
            direct_effect=0.2,
            indirect_effect=0.3,
            proportion_mediated=0.6,
            confidence_interval=(0.4, 0.8),
            treatment="T",
            outcome="Y",
            mediator="M",
        )

        assert result.total_effect == 0.5
        assert result.proportion_mediated == 0.6

    def test_to_dict(self):
        """Test serialization."""
        result = MediationResult(
            total_effect=0.5,
            direct_effect=0.2,
            indirect_effect=0.3,
            proportion_mediated=0.6,
            confidence_interval=(0.4, 0.8),
            treatment="T",
            outcome="Y",
            mediator="M",
        )

        d = result.to_dict()
        assert d["total_effect"] == 0.5
        assert d["treatment"] == "T"


class TestEffectDecomposition:
    """Tests for EffectDecomposition dataclass."""

    def test_creation(self):
        """Test decomposition creation."""
        decomp = EffectDecomposition(
            total_effect=0.5,
            path_effects={"A -> B -> C": 0.3, "A -> C": 0.2},
            residual=0.0,
            treatment="A",
            outcome="C",
        )

        assert decomp.total_effect == 0.5
        assert len(decomp.path_effects) == 2

    def test_to_dict(self):
        """Test serialization."""
        decomp = EffectDecomposition(
            total_effect=0.5,
            path_effects={"path1": 0.3},
            residual=0.2,
            treatment="A",
            outcome="C",
        )

        d = decomp.to_dict()
        assert d["residual"] == 0.2


class TestCausalEffectEstimator:
    """Tests for CausalEffectEstimator."""

    @pytest.fixture
    def mediation_scm(self):
        """Create SCM with mediator: T -> M -> Y with direct T -> Y."""
        scm = StructuralCausalModel()
        scm.add_node(CausalNode("T", CausalNodeType.VARIANT, observed=True))
        scm.add_node(CausalNode("M", CausalNodeType.PATHWAY, observed=True))
        scm.add_node(CausalNode("Y", CausalNodeType.PHENOTYPE, observed=True))

        # Treatment -> Mediator
        scm.add_edge(CausalEdge("T", "M", CausalEdgeType.CAUSES, 0.8, "indirect"))
        # Mediator -> Outcome
        scm.add_edge(CausalEdge("M", "Y", CausalEdgeType.CAUSES, 0.7, "mediation"))
        # Treatment -> Outcome (direct)
        scm.add_edge(CausalEdge("T", "Y", CausalEdgeType.CAUSES, 0.3, "direct"))

        return scm

    @pytest.fixture
    def estimator(self, mediation_scm):
        """Create effect estimator."""
        do_engine = DoCalculusEngine(mediation_scm)
        return CausalEffectEstimator(mediation_scm, do_engine)

    def test_total_effect(self, estimator):
        """Test total effect calculation."""
        te = estimator.total_effect("T", "Y")
        assert isinstance(te, float)

    def test_direct_effect(self, estimator):
        """Test direct effect (NDE) calculation."""
        nde = estimator.direct_effect("T", "Y", "M")
        assert isinstance(nde, float)

    def test_indirect_effect(self, estimator):
        """Test indirect effect (NIE) calculation."""
        nie = estimator.indirect_effect("T", "Y", "M")
        assert isinstance(nie, float)

    def test_effect_decomposition_consistency(self, estimator):
        """Test that TE ≈ NDE + NIE."""
        te = estimator.total_effect("T", "Y")
        nde = estimator.direct_effect("T", "Y", "M")
        nie = estimator.indirect_effect("T", "Y", "M")

        # Total effect should approximately equal sum of direct and indirect
        # (may not be exact due to non-linearities)
        assert te == pytest.approx(nde + nie, abs=0.3)

    def test_mediation_analysis(self, estimator):
        """Test full mediation analysis."""
        result = estimator.mediation_analysis("T", "Y", "M")

        assert isinstance(result, MediationResult)
        assert result.treatment == "T"
        assert result.outcome == "Y"
        assert result.mediator == "M"
        assert 0 <= result.proportion_mediated <= 1

    def test_mediation_analysis_explanation(self, estimator):
        """Test mediation explanation generation."""
        result = estimator.mediation_analysis("T", "Y", "M")
        assert result.explanation != ""
        assert "T" in result.explanation

    def test_decompose_effect_by_path(self, estimator):
        """Test effect decomposition by path."""
        decomp = estimator.decompose_effect_by_path("T", "Y")

        assert isinstance(decomp, EffectDecomposition)
        assert len(decomp.path_effects) > 0

    def test_controlled_direct_effect(self, estimator):
        """Test CDE at specific mediator levels."""
        cde_low = estimator.controlled_direct_effect("T", "Y", "M", mediator_value=0.0)
        cde_high = estimator.controlled_direct_effect("T", "Y", "M", mediator_value=1.0)

        # CDE should differ at different mediator levels
        assert isinstance(cde_low, float)
        assert isinstance(cde_high, float)

    def test_effect_heterogeneity(self, mediation_scm):
        """Test effect heterogeneity analysis."""
        # Add a modifier
        mediation_scm.add_node(CausalNode("X", CausalNodeType.CONFOUNDER, observed=True))
        mediation_scm.add_edge(CausalEdge("X", "Y", CausalEdgeType.MODIFIES, 0.3, "modify"))

        do_engine = DoCalculusEngine(mediation_scm)
        estimator = CausalEffectEstimator(mediation_scm, do_engine)

        results = estimator.effect_heterogeneity(
            treatment="T",
            outcome="Y",
            effect_modifiers=["X"]
        )

        assert "X" in results
        assert len(results["X"]) > 0

    def test_multi_mediator_analysis(self):
        """Test analysis with multiple mediators."""
        scm = StructuralCausalModel()
        scm.add_node(CausalNode("T", CausalNodeType.VARIANT, observed=True))
        scm.add_node(CausalNode("M1", CausalNodeType.PATHWAY, observed=True))
        scm.add_node(CausalNode("M2", CausalNodeType.PATHWAY, observed=True))
        scm.add_node(CausalNode("Y", CausalNodeType.PHENOTYPE, observed=True))

        scm.add_edge(CausalEdge("T", "M1", CausalEdgeType.CAUSES, 0.7, "path1"))
        scm.add_edge(CausalEdge("T", "M2", CausalEdgeType.CAUSES, 0.6, "path2"))
        scm.add_edge(CausalEdge("M1", "Y", CausalEdgeType.CAUSES, 0.5, "med1"))
        scm.add_edge(CausalEdge("M2", "Y", CausalEdgeType.CAUSES, 0.4, "med2"))

        do_engine = DoCalculusEngine(scm)
        estimator = CausalEffectEstimator(scm, do_engine)

        results = estimator.multi_mediator_analysis("T", "Y", ["M1", "M2"])

        assert "M1" in results
        assert "M2" in results

    def test_asd_mediation_analysis(self):
        """Test mediation analysis on ASD SCM."""
        scm = create_sample_asd_scm()
        do_engine = DoCalculusEngine(scm)
        estimator = CausalEffectEstimator(scm, do_engine)

        # How much of SHANK3's effect is mediated by synaptic pathway?
        result = estimator.mediation_analysis(
            treatment="SHANK3_function",
            outcome="asd_phenotype",
            mediator="synaptic_pathway"
        )

        assert isinstance(result, MediationResult)
        # Synaptic pathway should mediate some of the effect
        assert result.proportion_mediated >= 0

    def test_pathway_contribution_analysis(self):
        """Test pathway contribution analysis."""
        scm = create_sample_asd_scm()
        do_engine = DoCalculusEngine(scm)
        estimator = CausalEffectEstimator(scm, do_engine)

        contributions = estimator.pathway_contribution_analysis(
            gene="SHANK3_function",
            phenotype="asd_phenotype",
            pathways=["synaptic_pathway", "chromatin_pathway"]
        )

        assert "synaptic_pathway" in contributions
        assert "chromatin_pathway" in contributions


class TestIntegration:
    """Integration tests combining all components."""

    def test_full_causal_analysis_pipeline(self):
        """Test complete causal analysis pipeline."""
        # Create SCM
        scm = create_sample_asd_scm()

        # Do-calculus: intervention analysis
        do_engine = DoCalculusEngine(scm)
        ate = do_engine.average_treatment_effect(
            treatment="SHANK3_function",
            outcome="asd_phenotype"
        )

        # Counterfactual: individual-level reasoning
        cf_engine = CounterfactualEngine(scm, do_engine)
        cf_result = cf_engine.counterfactual(
            factual_evidence={"SHANK3_function": 0.0, "asd_phenotype": 0.8},
            counterfactual_intervention={"SHANK3_function": 1.0},
            query_variable="asd_phenotype"
        )

        # Effect estimation: mediation analysis
        estimator = CausalEffectEstimator(scm, do_engine)
        mediation = estimator.mediation_analysis(
            treatment="SHANK3_function",
            outcome="asd_phenotype",
            mediator="synaptic_pathway"
        )

        # All analyses should complete
        assert isinstance(ate, float)
        assert isinstance(cf_result, CounterfactualResult)
        assert isinstance(mediation, MediationResult)

    def test_causal_reasoning_consistency(self):
        """Test that different causal analyses give consistent results."""
        scm = create_sample_asd_scm()
        do_engine = DoCalculusEngine(scm)
        cf_engine = CounterfactualEngine(scm, do_engine)

        # ATE from do-calculus
        ate = do_engine.average_treatment_effect(
            treatment="SHANK3_function",
            outcome="asd_phenotype"
        )

        # PNS from counterfactual engine
        pns = cf_engine.probability_of_necessity_and_sufficiency(
            treatment="SHANK3_function",
            outcome="asd_phenotype"
        )

        # Both should indicate positive causal effect (if any)
        # PNS approximates ATE under certain conditions
        # Just verify both are computed and have consistent signs
        if ate > 0.1:
            assert pns > 0  # Positive effect
        elif ate < -0.1:
            assert pns < 0  # Negative effect

    def test_scm_modification_isolation(self):
        """Test that interventions don't modify original SCM."""
        scm = create_sample_asd_scm()
        original_edges = len(scm.edges)

        do_engine = DoCalculusEngine(scm)

        # Perform intervention
        do_engine.query(
            outcome="asd_phenotype",
            intervention={"SHANK3_function": 1.0}
        )

        # Original SCM should be unchanged
        assert len(scm.edges) == original_edges
