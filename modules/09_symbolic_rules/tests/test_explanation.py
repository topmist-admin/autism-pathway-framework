"""Tests for explanation and reasoning chain generation."""

import pytest
import sys
from pathlib import Path

# Add module to path
module_dir = Path(__file__).parent.parent
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

from explanation import (
    ReasoningChain,
    ReasoningStep,
    ExplanationGenerator,
    format_fired_rule,
    format_rule_summary,
)
from rule_engine import FiredRule
from biological_rules import (
    Rule,
    Conclusion,
    ConclusionType,
    BiologicalRules,
)
from conditions import Condition


def create_test_rule(
    rule_id: str,
    name: str,
    conclusion_type: str,
    confidence: float = 0.8,
    attributes: dict = None,
) -> Rule:
    """Helper to create test rules."""
    return Rule(
        id=rule_id,
        name=name,
        description=f"Test rule {rule_id}",
        conditions=[Condition(predicate="test", arguments={})],
        conclusion=Conclusion(
            type=conclusion_type,
            attributes=attributes or {},
        ),
        base_confidence=confidence,
        evidence_sources=["Test source"],
    )


def create_fired_rule(
    rule_id: str,
    name: str,
    conclusion_type: str,
    gene: str = None,
    confidence: float = 0.8,
    evidence: dict = None,
) -> FiredRule:
    """Helper to create fired rules for testing."""
    rule = create_test_rule(rule_id, name, conclusion_type, confidence)
    bindings = {"gene": gene, "G": gene} if gene else {}
    return FiredRule(
        rule=rule,
        bindings=bindings,
        confidence=confidence,
        explanation=f"Test explanation for {rule_id}",
        evidence=evidence or {},
    )


class TestReasoningStep:
    """Tests for ReasoningStep."""

    def test_step_creation(self):
        """Test basic step creation."""
        step = ReasoningStep(
            step_number=1,
            description="Identified variant in SHANK3",
            evidence={"gene": "SHANK3", "type": "LoF"},
        )
        assert step.step_number == 1
        assert "SHANK3" in step.description
        assert step.evidence["type"] == "LoF"

    def test_step_with_rule(self):
        """Test step with associated rule."""
        step = ReasoningStep(
            step_number=2,
            description="Applied R1 rule",
            rule_id="R1",
            confidence=0.9,
        )
        assert step.rule_id == "R1"
        assert step.confidence == 0.9

    def test_step_to_dict(self):
        """Test step serialization."""
        step = ReasoningStep(
            step_number=1,
            description="Test step",
            evidence={"key": "value"},
            rule_id="R1",
            confidence=0.85,
        )
        d = step.to_dict()
        assert d["step"] == 1
        assert d["description"] == "Test step"
        assert d["rule_id"] == "R1"
        assert d["confidence"] == 0.85


class TestReasoningChain:
    """Tests for ReasoningChain."""

    @pytest.fixture
    def sample_fired_rules(self):
        """Create sample fired rules."""
        return [
            create_fired_rule(
                "R1", "Constrained LoF", "pathway_disruption",
                gene="SHANK3", confidence=0.9,
                evidence={"pLI": 0.99},
            ),
            create_fired_rule(
                "R4", "Synaptic Disruption", "subtype_indicator",
                gene="SHANK3", confidence=0.82,
                evidence={"cell_type": "excitatory"},
            ),
            create_fired_rule(
                "R7", "SFARI High Confidence", "pathway_disruption",
                gene="SHANK3", confidence=0.88,
                evidence={"sfari_score": 1},
            ),
        ]

    def test_chain_creation(self):
        """Test basic chain creation."""
        chain = ReasoningChain(individual_id="TEST_001")
        assert chain.individual_id == "TEST_001"
        assert len(chain.fired_rules) == 0

    def test_chain_with_rules(self, sample_fired_rules):
        """Test chain with fired rules."""
        chain = ReasoningChain(
            individual_id="TEST_001",
            fired_rules=sample_fired_rules,
        )
        assert len(chain.fired_rules) == 3
        assert chain.n_rules_fired == 3

    def test_average_confidence(self, sample_fired_rules):
        """Test average confidence calculation."""
        chain = ReasoningChain(
            individual_id="TEST_001",
            fired_rules=sample_fired_rules,
        )
        # (0.9 + 0.82 + 0.88) / 3 ≈ 0.867
        assert chain.average_confidence == pytest.approx(0.867, rel=0.01)

    def test_genes_affected(self, sample_fired_rules):
        """Test genes affected property."""
        chain = ReasoningChain(
            individual_id="TEST_001",
            fired_rules=sample_fired_rules,
        )
        assert "SHANK3" in chain.genes_affected

    def test_get_rules_by_type(self, sample_fired_rules):
        """Test filtering rules by type."""
        chain = ReasoningChain(
            individual_id="TEST_001",
            fired_rules=sample_fired_rules,
        )
        pathway_rules = chain.get_rules_by_type("pathway_disruption")
        assert len(pathway_rules) == 2  # R1 and R7

        subtype_rules = chain.get_rules_by_type("subtype_indicator")
        assert len(subtype_rules) == 1  # R4

    def test_chain_to_dict(self, sample_fired_rules):
        """Test chain serialization."""
        chain = ReasoningChain(
            individual_id="TEST_001",
            fired_rules=sample_fired_rules,
            pathway_conclusions={"test_pathway": 0.85},
            subtype_indicators=["synaptic_dysfunction"],
        )
        d = chain.to_dict()
        assert d["individual_id"] == "TEST_001"
        assert len(d["fired_rules"]) == 3
        assert "synaptic_dysfunction" in d["subtype_indicators"]

    def test_chain_to_json(self, sample_fired_rules):
        """Test JSON conversion."""
        chain = ReasoningChain(
            individual_id="TEST_001",
            fired_rules=sample_fired_rules,
        )
        json_str = chain.to_json()
        assert "TEST_001" in json_str
        assert "R1" in json_str


class TestExplanationGenerator:
    """Tests for ExplanationGenerator."""

    @pytest.fixture
    def generator(self):
        """Create explanation generator."""
        return ExplanationGenerator(
            include_evidence=True,
            include_disclaimers=True,
            verbose=True,
        )

    @pytest.fixture
    def sample_fired_rules(self):
        """Create sample fired rules."""
        return [
            create_fired_rule(
                "R1", "Constrained LoF in Developing Cortex", "pathway_disruption",
                gene="SHANK3", confidence=0.9,
                evidence={"pLI": 0.99, "prenatal": True},
            ),
            create_fired_rule(
                "R4", "Synaptic Excitatory Disruption", "subtype_indicator",
                gene="SHANK3", confidence=0.82,
                evidence={"cell_type": "excitatory"},
            ),
        ]

    def test_generator_creation(self):
        """Test generator creation."""
        gen = ExplanationGenerator()
        assert gen.include_evidence is True
        assert gen.include_disclaimers is True

    def test_generate_reasoning_chain(self, generator, sample_fired_rules):
        """Test full reasoning chain generation."""
        chain = generator.generate_reasoning_chain("TEST_001", sample_fired_rules)

        assert chain.individual_id == "TEST_001"
        assert len(chain.fired_rules) == 2
        assert len(chain.steps) > 0
        assert len(chain.explanation_text) > 0

    def test_extract_pathway_conclusions(self, generator, sample_fired_rules):
        """Test pathway conclusion extraction."""
        chain = generator.generate_reasoning_chain("TEST_001", sample_fired_rules)

        # R1 produces pathway_disruption
        assert len(chain.pathway_conclusions) > 0

    def test_extract_subtype_indicators(self, generator, sample_fired_rules):
        """Test subtype indicator extraction."""
        # Add subtype attribute to R4
        sample_fired_rules[1].rule.conclusion.attributes["subtype"] = "synaptic_dysfunction"

        chain = generator.generate_reasoning_chain("TEST_001", sample_fired_rules)
        assert "synaptic_dysfunction" in chain.subtype_indicators

    def test_generate_steps(self, generator, sample_fired_rules):
        """Test step generation."""
        chain = generator.generate_reasoning_chain("TEST_001", sample_fired_rules)

        assert len(chain.steps) >= 2  # At least variant identification + rule steps
        assert chain.steps[0].step_number == 1

    def test_explanation_text_contains_key_info(self, generator, sample_fired_rules):
        """Test explanation text content."""
        chain = generator.generate_reasoning_chain("TEST_001", sample_fired_rules)

        text = chain.explanation_text
        assert "TEST_001" in text
        assert "Rules fired" in text
        assert "SHANK3" in text

    def test_explanation_includes_disclaimer(self, generator, sample_fired_rules):
        """Test disclaimer inclusion."""
        chain = generator.generate_reasoning_chain("TEST_001", sample_fired_rules)
        assert "research purposes" in chain.explanation_text.lower()

    def test_generate_clinical_summary(self, generator, sample_fired_rules):
        """Test clinical summary generation."""
        chain = generator.generate_reasoning_chain("TEST_001", sample_fired_rules)
        summary = generator.generate_clinical_summary(chain)

        assert "TEST_001" in summary
        assert "KEY FINDINGS" in summary

    def test_compare_individuals(self, generator):
        """Test individual comparison."""
        chains = [
            ReasoningChain(
                individual_id="IND_001",
                pathway_conclusions={"synaptic": 0.9, "chromatin": 0.7},
                subtype_indicators=["synaptic_dysfunction"],
            ),
            ReasoningChain(
                individual_id="IND_002",
                pathway_conclusions={"synaptic": 0.85},
                subtype_indicators=["synaptic_dysfunction"],
            ),
            ReasoningChain(
                individual_id="IND_003",
                pathway_conclusions={"chromatin": 0.8},
                subtype_indicators=["chromatin_remodeling"],
            ),
        ]
        chains[0]._genes_affected = {"SHANK3", "GRIN2A"}
        chains[1]._genes_affected = {"SHANK3"}
        chains[2]._genes_affected = {"CHD8"}

        comparison = generator.compare_individuals(chains)
        assert "3 Individuals" in comparison
        assert "synaptic" in comparison.lower()


class TestFormatFunctions:
    """Tests for formatting functions."""

    def test_format_fired_rule(self):
        """Test fired rule formatting."""
        fired = create_fired_rule(
            "R1", "Test Rule", "pathway_disruption",
            gene="SHANK3", confidence=0.9,
        )
        formatted = format_fired_rule(fired)

        assert "[R1]" in formatted
        assert "Test Rule" in formatted
        assert "SHANK3" in formatted
        assert "90" in formatted  # 90% confidence

    def test_format_fired_rule_verbose(self):
        """Test verbose fired rule formatting."""
        fired = create_fired_rule(
            "R1", "Test Rule", "pathway_disruption",
            gene="SHANK3", confidence=0.9,
            evidence={"pLI": 0.99},
        )
        formatted = format_fired_rule(fired, verbose=True)

        assert "Evidence" in formatted
        assert "pLI" in formatted

    def test_format_rule_summary_empty(self):
        """Test summary with no rules."""
        summary = format_rule_summary([])
        assert "No rules fired" in summary

    def test_format_rule_summary(self):
        """Test rule summary formatting."""
        fired_rules = [
            create_fired_rule("R1", "Rule 1", "pathway_disruption", gene="A", confidence=0.9),
            create_fired_rule("R2", "Rule 2", "pathway_disruption", gene="B", confidence=0.85),
            create_fired_rule("R4", "Rule 4", "subtype_indicator", gene="A", confidence=0.8),
        ]
        summary = format_rule_summary(fired_rules)

        assert "Rules fired: 3" in summary
        assert "pathway_disruption" in summary
        assert "subtype_indicator" in summary


class TestExplanationGeneratorEdgeCases:
    """Edge case tests for ExplanationGenerator."""

    def test_empty_fired_rules(self):
        """Test with no fired rules."""
        gen = ExplanationGenerator()
        chain = gen.generate_reasoning_chain("TEST", [])

        assert chain.n_rules_fired == 0
        assert chain.average_confidence == 0.0
        assert len(chain.explanation_text) > 0

    def test_generator_without_disclaimers(self):
        """Test generator without disclaimers."""
        gen = ExplanationGenerator(include_disclaimers=False)
        fired = [create_fired_rule("R1", "Test", "pathway_disruption")]
        chain = gen.generate_reasoning_chain("TEST", fired)

        assert "research purposes" not in chain.explanation_text.lower()

    def test_generator_non_verbose(self):
        """Test non-verbose generator."""
        gen = ExplanationGenerator(verbose=False)
        fired = [create_fired_rule("R1", "Test", "pathway_disruption")]
        chain = gen.generate_reasoning_chain("TEST", fired)

        # Non-verbose should still have summary but fewer details
        assert len(chain.explanation_text) > 0

    def test_therapeutic_hypothesis_extraction(self):
        """Test therapeutic hypothesis extraction."""
        gen = ExplanationGenerator()

        # Create R6-like therapeutic hypothesis rule
        rule = Rule(
            id="R6",
            name="Therapeutic Target",
            description="Test therapeutic",
            conditions=[Condition(predicate="test", arguments={})],
            conclusion=Conclusion(
                type="therapeutic_hypothesis",
                attributes={
                    "requires_validation": True,
                    "drug": "TestDrug",
                },
            ),
            base_confidence=0.6,
        )
        fired = FiredRule(
            rule=rule,
            confidence=0.6,
            evidence={"pathway": "test_pathway"},
        )

        chain = gen.generate_reasoning_chain("TEST", [fired])
        assert len(chain.therapeutic_hypotheses) == 1
        assert chain.therapeutic_hypotheses[0]["rule_id"] == "R6"
