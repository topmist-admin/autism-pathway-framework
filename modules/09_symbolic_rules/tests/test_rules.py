"""Tests for biological rules (R1-R7)."""

import pytest
import sys
from pathlib import Path

# Add module to path
module_dir = Path(__file__).parent.parent
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

from biological_rules import (
    Rule,
    Conclusion,
    ConclusionType,
    BiologicalRules,
)
from conditions import Condition


class TestConclusion:
    """Tests for Conclusion dataclass."""

    def test_conclusion_creation(self):
        """Test basic conclusion creation."""
        conc = Conclusion(
            type=ConclusionType.PATHWAY_DISRUPTION.value,
            attributes={"mechanism": "haploinsufficiency"},
        )
        assert conc.type == "pathway_disruption"
        assert conc.attributes["mechanism"] == "haploinsufficiency"
        assert conc.confidence_modifier == 1.0

    def test_conclusion_with_modifier(self):
        """Test conclusion with confidence modifier."""
        conc = Conclusion(
            type=ConclusionType.EFFECT_MODIFIER.value,
            attributes={"modifier_type": "compensation"},
            confidence_modifier=0.7,
        )
        assert conc.confidence_modifier == 0.7

    def test_conclusion_to_dict(self):
        """Test conclusion serialization."""
        conc = Conclusion(
            type="therapeutic_hypothesis",
            attributes={"drug": "example_drug"},
        )
        d = conc.to_dict()
        assert d["type"] == "therapeutic_hypothesis"
        assert d["attributes"]["drug"] == "example_drug"

    def test_conclusion_from_dict(self):
        """Test conclusion deserialization."""
        d = {
            "type": "subtype_indicator",
            "attributes": {"subtype": "synaptic"},
            "confidence_modifier": 0.9,
        }
        conc = Conclusion.from_dict(d)
        assert conc.type == "subtype_indicator"
        assert conc.confidence_modifier == 0.9


class TestRule:
    """Tests for Rule dataclass."""

    def test_rule_creation(self):
        """Test basic rule creation."""
        rule = Rule(
            id="TEST_1",
            name="Test Rule",
            description="A test rule",
            conditions=[
                Condition(predicate="has_variant", arguments={}),
            ],
            conclusion=Conclusion(type="test", attributes={}),
            base_confidence=0.8,
        )
        assert rule.id == "TEST_1"
        assert rule.name == "Test Rule"
        assert len(rule.conditions) == 1
        assert rule.base_confidence == 0.8

    def test_rule_with_evidence_sources(self):
        """Test rule with evidence sources."""
        rule = Rule(
            id="TEST_2",
            name="Test Rule with Evidence",
            description="A rule with evidence",
            conditions=[Condition(predicate="test", arguments={})],
            conclusion=Conclusion(type="test", attributes={}),
            evidence_sources=["Paper 1", "Paper 2"],
        )
        assert len(rule.evidence_sources) == 2
        assert "Paper 1" in rule.evidence_sources

    def test_rule_validation_empty_id(self):
        """Test that empty ID raises error."""
        with pytest.raises(ValueError):
            Rule(
                id="",
                name="Test",
                description="Test",
                conditions=[Condition(predicate="test", arguments={})],
                conclusion=Conclusion(type="test", attributes={}),
            )

    def test_rule_validation_no_conditions(self):
        """Test that no conditions raises error."""
        with pytest.raises(ValueError):
            Rule(
                id="TEST",
                name="Test",
                description="Test",
                conditions=[],
                conclusion=Conclusion(type="test", attributes={}),
            )

    def test_rule_validation_invalid_confidence(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError):
            Rule(
                id="TEST",
                name="Test",
                description="Test",
                conditions=[Condition(predicate="test", arguments={})],
                conclusion=Conclusion(type="test", attributes={}),
                base_confidence=1.5,  # Invalid: > 1
            )

    def test_rule_validation_invalid_logic(self):
        """Test that invalid logic raises error."""
        with pytest.raises(ValueError):
            Rule(
                id="TEST",
                name="Test",
                description="Test",
                conditions=[Condition(predicate="test", arguments={})],
                conclusion=Conclusion(type="test", attributes={}),
                logic="XOR",  # Invalid
            )

    def test_rule_to_dict(self):
        """Test rule serialization."""
        rule = Rule(
            id="TEST",
            name="Test",
            description="Test rule",
            conditions=[Condition(predicate="has_variant", arguments={})],
            conclusion=Conclusion(type="test", attributes={"key": "value"}),
        )
        d = rule.to_dict()
        assert d["id"] == "TEST"
        assert len(d["conditions"]) == 1
        assert d["conclusion"]["type"] == "test"


class TestBiologicalRules:
    """Tests for BiologicalRules factory class."""

    def test_r1_constrained_lof(self):
        """Test R1 rule structure."""
        r1 = BiologicalRules.R1_constrained_lof_developing_cortex()

        assert r1.id == "R1"
        assert r1.name == "Constrained LoF in Developing Cortex"
        assert len(r1.conditions) == 3
        assert r1.conclusion.type == ConclusionType.PATHWAY_DISRUPTION.value
        assert r1.base_confidence >= 0.85
        assert len(r1.evidence_sources) > 0

        # Check condition predicates
        predicates = [c.predicate for c in r1.conditions]
        assert "has_lof_variant" in predicates
        assert "is_constrained" in predicates or "prenatally_expressed" in predicates

    def test_r2_pathway_convergence(self):
        """Test R2 rule structure."""
        r2 = BiologicalRules.R2_pathway_convergence()

        assert r2.id == "R2"
        assert r2.conclusion.type == ConclusionType.PATHWAY_CONVERGENCE.value
        assert "has_multiple_hits" in [c.predicate for c in r2.conditions]

    def test_r3_chd8_cascade(self):
        """Test R3 rule structure."""
        r3 = BiologicalRules.R3_chd8_cascade()

        assert r3.id == "R3"
        assert "CHD8" in r3.name
        assert r3.conclusion.attributes.get("pathway") == "chromatin_regulation"

    def test_r3b_chd8_target(self):
        """Test R3b rule structure."""
        r3b = BiologicalRules.R3b_chd8_target_cascade()

        assert r3b.id == "R3b"
        assert "is_chd8_target" in [c.predicate for c in r3b.conditions]
        assert r3b.conclusion.attributes.get("indirect") is True

    def test_r4_synaptic_excitatory(self):
        """Test R4 rule structure."""
        r4 = BiologicalRules.R4_synaptic_excitatory()

        assert r4.id == "R4"
        assert r4.conclusion.type == ConclusionType.SUBTYPE_INDICATOR.value
        assert r4.conclusion.attributes.get("subtype") == "synaptic_dysfunction"
        assert "is_synaptic_gene" in [c.predicate for c in r4.conditions]

    def test_r4b_synaptic_inhibitory(self):
        """Test R4b rule structure."""
        r4b = BiologicalRules.R4b_synaptic_inhibitory()

        assert r4b.id == "R4b"
        assert r4b.conclusion.attributes.get("cell_type") == "inhibitory"

    def test_r5_paralog_compensation(self):
        """Test R5 rule structure."""
        r5 = BiologicalRules.R5_compensatory_paralog()

        assert r5.id == "R5"
        assert r5.conclusion.type == ConclusionType.EFFECT_MODIFIER.value
        assert "has_paralog" in [c.predicate for c in r5.conditions]

        # Check for negated condition (paralog NOT disrupted)
        negated = [c for c in r5.conditions if c.negated]
        assert len(negated) >= 1

    def test_r6_drug_pathway_target(self):
        """Test R6 rule structure."""
        r6 = BiologicalRules.R6_drug_pathway_target()

        assert r6.id == "R6"
        assert r6.conclusion.type == ConclusionType.THERAPEUTIC_HYPOTHESIS.value
        assert r6.conclusion.attributes.get("requires_validation") is True
        assert r6.base_confidence < 0.7  # Lower confidence for hypothesis

    def test_r7_sfari_high_confidence(self):
        """Test R7 rule structure."""
        r7 = BiologicalRules.R7_sfari_high_confidence()

        assert r7.id == "R7"
        assert "is_high_confidence_sfari" in [c.predicate for c in r7.conditions]

    def test_get_all_rules(self):
        """Test getting all rules."""
        rules = BiologicalRules.get_all_rules()

        assert len(rules) >= 6  # At least R1-R6
        rule_ids = [r.id for r in rules]
        assert "R1" in rule_ids
        assert "R2" in rule_ids
        assert "R3" in rule_ids
        assert "R4" in rule_ids
        assert "R5" in rule_ids
        assert "R6" in rule_ids

    def test_get_core_rules(self):
        """Test getting core rules (without variants)."""
        core = BiologicalRules.get_core_rules()

        assert len(core) == 6
        rule_ids = [r.id for r in core]
        assert "R3b" not in rule_ids
        assert "R4b" not in rule_ids

    def test_get_rules_by_conclusion_type(self):
        """Test filtering rules by conclusion type."""
        pathway_rules = BiologicalRules.get_rules_by_conclusion_type(
            ConclusionType.PATHWAY_DISRUPTION
        )
        assert len(pathway_rules) >= 2

        subtype_rules = BiologicalRules.get_rules_by_conclusion_type(
            ConclusionType.SUBTYPE_INDICATOR
        )
        assert len(subtype_rules) >= 1

        therapeutic_rules = BiologicalRules.get_rules_by_conclusion_type(
            ConclusionType.THERAPEUTIC_HYPOTHESIS
        )
        assert len(therapeutic_rules) >= 1

    def test_get_rule_by_id(self):
        """Test getting specific rule by ID."""
        r1 = BiologicalRules.get_rule_by_id("R1")
        assert r1 is not None
        assert r1.name == "Constrained LoF in Developing Cortex"

        r3b = BiologicalRules.get_rule_by_id("R3b")
        assert r3b is not None
        assert r3b.id == "R3b"

        unknown = BiologicalRules.get_rule_by_id("UNKNOWN")
        assert unknown is None

    def test_rules_have_unique_ids(self):
        """Test that all rules have unique IDs."""
        rules = BiologicalRules.get_all_rules()
        ids = [r.id for r in rules]
        assert len(ids) == len(set(ids))

    def test_rules_have_evidence_sources(self):
        """Test that all rules have evidence sources."""
        rules = BiologicalRules.get_all_rules()
        for rule in rules:
            assert len(rule.evidence_sources) > 0, f"Rule {rule.id} has no evidence sources"

    def test_rule_priorities(self):
        """Test that rules have appropriate priorities."""
        rules = BiologicalRules.get_all_rules()

        # R1 and R3 should be high priority (direct mechanism)
        r1 = BiologicalRules.get_rule_by_id("R1")
        r6 = BiologicalRules.get_rule_by_id("R6")

        assert r1.priority > r6.priority  # R1 should be higher than R6


class TestConclusionType:
    """Tests for ConclusionType enum."""

    def test_conclusion_types(self):
        """Test all conclusion types exist."""
        assert ConclusionType.PATHWAY_DISRUPTION.value == "pathway_disruption"
        assert ConclusionType.PATHWAY_CONVERGENCE.value == "pathway_convergence"
        assert ConclusionType.SUBTYPE_INDICATOR.value == "subtype_indicator"
        assert ConclusionType.EFFECT_MODIFIER.value == "effect_modifier"
        assert ConclusionType.THERAPEUTIC_HYPOTHESIS.value == "therapeutic_hypothesis"
