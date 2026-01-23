"""Tests for condition data structures and evaluators."""

import pytest
import sys
from pathlib import Path

# Add module to path
module_dir = Path(__file__).parent.parent
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

from conditions import (
    Condition,
    ConditionType,
    ConditionResult,
    ConditionEvaluator,
)
from rule_engine import (
    BiologicalContext,
    IndividualData,
)


class MockVariant:
    """Mock variant for testing."""

    def __init__(
        self,
        gene_id: str,
        is_lof: bool = False,
        is_damaging: bool = False,
        is_missense: bool = False,
        cadd_phred: float = None,
    ):
        self.gene_id = gene_id
        self.is_lof = is_lof
        self.is_damaging = is_damaging or is_lof
        self.is_missense = is_missense
        self.cadd_phred = cadd_phred

    def __str__(self):
        return f"Variant({self.gene_id})"


class TestCondition:
    """Tests for Condition dataclass."""

    def test_condition_creation(self):
        """Test basic condition creation."""
        cond = Condition(
            predicate="has_variant",
            arguments={"gene": "SHANK3"},
        )
        assert cond.predicate == "has_variant"
        assert cond.arguments["gene"] == "SHANK3"
        assert cond.negated is False

    def test_condition_negated(self):
        """Test negated condition."""
        cond = Condition(
            predicate="has_variant",
            arguments={"gene": "SHANK3"},
            negated=True,
        )
        assert cond.negated is True
        assert "NOT" in str(cond)

    def test_condition_to_dict(self):
        """Test condition serialization."""
        cond = Condition(
            predicate="is_constrained",
            arguments={"pli_threshold": 0.9},
        )
        d = cond.to_dict()
        assert d["predicate"] == "is_constrained"
        assert d["arguments"]["pli_threshold"] == 0.9

    def test_condition_from_dict(self):
        """Test condition deserialization."""
        d = {
            "predicate": "is_sfari_gene",
            "arguments": {"max_score": 2},
            "negated": False,
        }
        cond = Condition.from_dict(d)
        assert cond.predicate == "is_sfari_gene"
        assert cond.arguments["max_score"] == 2

    def test_condition_empty_predicate_raises(self):
        """Test that empty predicate raises error."""
        with pytest.raises(ValueError):
            Condition(predicate="", arguments={})

    def test_condition_str_representation(self):
        """Test string representation."""
        cond = Condition(
            predicate="expressed_in",
            arguments={"tissue": "cortex", "stage": "prenatal"},
        )
        s = str(cond)
        assert "expressed_in" in s
        assert "cortex" in s


class TestConditionResult:
    """Tests for ConditionResult dataclass."""

    def test_result_satisfied(self):
        """Test satisfied result."""
        result = ConditionResult(
            satisfied=True,
            evidence={"gene": "SHANK3", "pLI": 0.95},
            explanation="SHANK3 is constrained",
        )
        assert result.satisfied is True
        assert result.evidence["pLI"] == 0.95

    def test_result_not_satisfied(self):
        """Test not satisfied result."""
        result = ConditionResult(
            satisfied=False,
            explanation="Gene not found in database",
        )
        assert result.satisfied is False

    def test_result_bound_variables(self):
        """Test result with bound variables."""
        result = ConditionResult(
            satisfied=True,
            bound_variables={"G": "SHANK3", "P": "PSD"},
        )
        assert result.bound_variables["G"] == "SHANK3"


class TestConditionEvaluator:
    """Tests for ConditionEvaluator."""

    @pytest.fixture
    def context(self):
        """Create test biological context."""
        ctx = BiologicalContext()
        # Set up mock data
        ctx.chd8_targets = {"ARID1B", "SMARCC2", "KMT2A"}
        ctx.syngo_genes = {"SHANK3", "NLGN1", "NRXN1", "GRIN2A"}
        ctx.paralog_map = {
            "SHANK3": ["SHANK1", "SHANK2"],
            "NLGN1": ["NLGN2", "NLGN3"],
        }
        return ctx

    @pytest.fixture
    def evaluator(self, context):
        """Create condition evaluator."""
        return ConditionEvaluator(context)

    @pytest.fixture
    def individual_with_variants(self):
        """Create individual with test variants."""
        variants = [
            MockVariant("SHANK3", is_lof=True),
            MockVariant("CHD8", is_damaging=True),
            MockVariant("GRIN2A", is_missense=True, cadd_phred=25.0),
        ]
        return IndividualData(
            sample_id="TEST_001",
            variants=variants,
            gene_burdens={"SHANK3": 1.0, "CHD8": 0.8, "GRIN2A": 0.5},
            pathway_scores={"synaptic_transmission": 2.5, "chromatin_remodeling": 1.8},
        )

    def test_eval_has_lof_variant(self, evaluator, individual_with_variants):
        """Test LoF variant condition evaluation."""
        cond = Condition(predicate="has_lof_variant", arguments={})
        result = evaluator.evaluate(cond, individual_with_variants)

        assert result.satisfied is True
        assert "lof_count" in result.evidence
        assert result.evidence["lof_count"] >= 1

    def test_eval_has_lof_variant_specific_gene(self, evaluator, individual_with_variants):
        """Test LoF variant in specific gene."""
        # Gene with LoF
        cond = Condition(predicate="has_lof_variant", arguments={"gene": "SHANK3"})
        result = evaluator.evaluate(cond, individual_with_variants)
        # Note: This checks if there are LoF variants in SHANK3 specifically

    def test_eval_has_damaging_variant(self, evaluator, individual_with_variants):
        """Test damaging variant condition."""
        cond = Condition(predicate="has_damaging_variant", arguments={})
        result = evaluator.evaluate(cond, individual_with_variants)

        assert result.satisfied is True
        assert "damaging_count" in result.evidence

    def test_eval_is_chd8_target(self, evaluator, individual_with_variants):
        """Test CHD8 target condition."""
        # ARID1B is a CHD8 target
        cond = Condition(predicate="is_chd8_target", arguments={"gene": "ARID1B"})
        result = evaluator.evaluate(cond, individual_with_variants)
        assert result.satisfied is True

        # SHANK3 is not a CHD8 target
        cond = Condition(predicate="is_chd8_target", arguments={"gene": "SHANK3"})
        result = evaluator.evaluate(cond, individual_with_variants)
        assert result.satisfied is False

    def test_eval_is_synaptic_gene(self, evaluator, individual_with_variants):
        """Test synaptic gene condition."""
        # SHANK3 is synaptic
        cond = Condition(predicate="is_synaptic_gene", arguments={"gene": "SHANK3"})
        result = evaluator.evaluate(cond, individual_with_variants)
        assert result.satisfied is True

        # CHD8 is not synaptic
        cond = Condition(predicate="is_synaptic_gene", arguments={"gene": "CHD8"})
        result = evaluator.evaluate(cond, individual_with_variants)
        assert result.satisfied is False

    def test_eval_has_paralog(self, evaluator, individual_with_variants):
        """Test paralog condition."""
        # SHANK3 has paralogs
        cond = Condition(predicate="has_paralog", arguments={"gene": "SHANK3"})
        result = evaluator.evaluate(cond, individual_with_variants)
        assert result.satisfied is True
        assert "paralogs" in result.evidence

    def test_eval_pathway_disrupted(self, evaluator, individual_with_variants):
        """Test pathway disruption condition."""
        # Above threshold
        cond = Condition(
            predicate="pathway_disrupted",
            arguments={"pathway": "synaptic_transmission", "score_threshold": 2.0},
        )
        result = evaluator.evaluate(cond, individual_with_variants)
        assert result.satisfied is True

        # Below threshold
        cond = Condition(
            predicate="pathway_disrupted",
            arguments={"pathway": "synaptic_transmission", "score_threshold": 3.0},
        )
        result = evaluator.evaluate(cond, individual_with_variants)
        assert result.satisfied is False

    def test_eval_negated_condition(self, evaluator, individual_with_variants):
        """Test negated condition evaluation."""
        # SHANK3 IS synaptic, so negated should be False
        cond = Condition(
            predicate="is_synaptic_gene",
            arguments={"gene": "SHANK3"},
            negated=True,
        )
        result = evaluator.evaluate(cond, individual_with_variants)
        assert result.satisfied is False

        # CHD8 is NOT synaptic, so negated should be True
        cond = Condition(
            predicate="is_synaptic_gene",
            arguments={"gene": "CHD8"},
            negated=True,
        )
        result = evaluator.evaluate(cond, individual_with_variants)
        assert result.satisfied is True

    def test_unknown_predicate(self, evaluator, individual_with_variants):
        """Test unknown predicate handling."""
        cond = Condition(predicate="unknown_predicate", arguments={})
        result = evaluator.evaluate(cond, individual_with_variants)
        assert result.satisfied is False
        assert "Unknown predicate" in result.explanation


class TestConditionType:
    """Tests for ConditionType enum."""

    def test_variant_conditions(self):
        """Test variant-level condition types."""
        assert ConditionType.HAS_VARIANT.value == "has_variant"
        assert ConditionType.HAS_LOF_VARIANT.value == "has_lof_variant"
        assert ConditionType.HAS_DAMAGING_VARIANT.value == "has_damaging_variant"

    def test_gene_conditions(self):
        """Test gene-level condition types."""
        assert ConditionType.IS_CONSTRAINED.value == "is_constrained"
        assert ConditionType.IS_SFARI_GENE.value == "is_sfari_gene"

    def test_expression_conditions(self):
        """Test expression condition types."""
        assert ConditionType.EXPRESSED_IN.value == "expressed_in"
        assert ConditionType.PRENATALLY_EXPRESSED.value == "prenatally_expressed"
