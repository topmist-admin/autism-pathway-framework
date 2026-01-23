"""Tests for rule engine."""

import pytest
import sys
from pathlib import Path

# Add module to path
module_dir = Path(__file__).parent.parent
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

from rule_engine import (
    RuleEngine,
    FiredRule,
    IndividualData,
    BiologicalContext,
)
from biological_rules import (
    Rule,
    Conclusion,
    ConclusionType,
    BiologicalRules,
)
from conditions import Condition


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


class MockGeneConstraints:
    """Mock gene constraints for testing."""

    def __init__(self):
        self.pli_scores = {
            "SHANK3": 0.99,
            "CHD8": 0.98,
            "SCN2A": 0.95,
            "ARID1B": 0.92,
            "LOWPLI": 0.3,
        }

    def get_pli(self, gene_id: str) -> float:
        return self.pli_scores.get(gene_id)

    def is_constrained(self, gene_id: str, threshold: float = 0.9) -> bool:
        pli = self.pli_scores.get(gene_id)
        return pli is not None and pli >= threshold


class MockDevelopmentalExpression:
    """Mock developmental expression for testing."""

    def __init__(self):
        self.prenatal_genes = {"SHANK3", "CHD8", "SCN2A", "ARID1B", "GRIN2A"}

    def is_prenatally_expressed(self, gene_id: str, threshold: float = 1.0) -> bool:
        return gene_id in self.prenatal_genes

    def get_prenatal_expression(self, gene_id: str) -> float:
        if gene_id in self.prenatal_genes:
            return 2.5
        return 0.0


class MockSingleCellAtlas:
    """Mock single cell atlas for testing."""

    def __init__(self):
        self.excitatory_genes = {"SHANK3", "GRIN2A", "CAMK2A"}
        self.inhibitory_genes = {"GAD1", "GAD2", "PVALB"}

    def is_enriched_in(self, gene_id: str, cell_type: str, threshold: float = 2.0) -> bool:
        if cell_type == "excitatory_neuron":
            return gene_id in self.excitatory_genes
        if cell_type == "inhibitory_neuron":
            return gene_id in self.inhibitory_genes
        return False

    def get_expression(self, gene_id: str, cell_type: str = None) -> float:
        return 2.0 if gene_id in self.excitatory_genes or gene_id in self.inhibitory_genes else 0.5


class MockSFARIGenes:
    """Mock SFARI genes for testing."""

    def __init__(self):
        self.scores = {
            "SHANK3": 1,
            "CHD8": 1,
            "SCN2A": 1,
            "ARID1B": 1,
            "GRIN2A": 2,
        }

    def get_score(self, gene_id: str) -> int:
        return self.scores.get(gene_id)

    def is_sfari_gene(self, gene_id: str) -> bool:
        return gene_id in self.scores

    def is_high_confidence(self, gene_id: str) -> bool:
        return self.scores.get(gene_id) == 1


class MockPathwayDB:
    """Mock pathway database for testing."""

    def __init__(self):
        self.pathways = {
            "synaptic_transmission": {"SHANK3", "GRIN2A", "NLGN1", "NRXN1"},
            "chromatin_regulation": {"CHD8", "ARID1B", "SMARCC2", "KMT2A"},
        }

    def get_pathway_genes(self, pathway_id: str) -> set:
        return self.pathways.get(pathway_id, set())


class TestIndividualData:
    """Tests for IndividualData."""

    def test_individual_creation(self):
        """Test basic individual creation."""
        individual = IndividualData(sample_id="TEST_001")
        assert individual.sample_id == "TEST_001"
        assert len(individual.variants) == 0

    def test_individual_with_variants(self):
        """Test individual with variants."""
        variants = [
            MockVariant("SHANK3", is_lof=True),
            MockVariant("CHD8", is_damaging=True),
        ]
        individual = IndividualData(
            sample_id="TEST_001",
            variants=variants,
        )
        assert len(individual.variants) == 2

    def test_get_gene_variants(self):
        """Test getting variants for a gene."""
        variants = [
            MockVariant("SHANK3", is_lof=True),
            MockVariant("SHANK3", is_missense=True),
            MockVariant("CHD8", is_damaging=True),
        ]
        individual = IndividualData(sample_id="TEST_001", variants=variants)

        shank3_vars = individual.get_gene_variants("SHANK3")
        assert len(shank3_vars) == 2

        chd8_vars = individual.get_gene_variants("CHD8")
        assert len(chd8_vars) == 1

    def test_get_pathway_score(self):
        """Test getting pathway score."""
        individual = IndividualData(
            sample_id="TEST_001",
            pathway_scores={"synaptic": 2.5, "chromatin": 1.8},
        )
        assert individual.get_pathway_score("synaptic") == 2.5
        assert individual.get_pathway_score("unknown") is None

    def test_get_genes_with_damaging_variants(self):
        """Test getting genes with damaging variants."""
        variants = [
            MockVariant("SHANK3", is_lof=True),
            MockVariant("CHD8", is_damaging=True),
            MockVariant("OTHER", is_missense=True),  # Not damaging
        ]
        individual = IndividualData(sample_id="TEST_001", variants=variants)

        damaging_genes = individual.get_genes_with_damaging_variants()
        assert "SHANK3" in damaging_genes
        assert "CHD8" in damaging_genes


class TestBiologicalContext:
    """Tests for BiologicalContext."""

    @pytest.fixture
    def context(self):
        """Create test context."""
        ctx = BiologicalContext(
            gene_constraints=MockGeneConstraints(),
            developmental_expression=MockDevelopmentalExpression(),
            single_cell_atlas=MockSingleCellAtlas(),
            sfari_genes=MockSFARIGenes(),
            pathway_db=MockPathwayDB(),
        )
        ctx.chd8_targets = {"ARID1B", "SMARCC2", "KMT2A"}
        ctx.syngo_genes = {"SHANK3", "NLGN1", "NRXN1", "GRIN2A"}
        ctx.paralog_map = {"SHANK3": ["SHANK1", "SHANK2"]}
        return ctx

    def test_is_gene_constrained(self, context):
        """Test gene constraint check."""
        assert context.is_gene_constrained("SHANK3") is True
        assert context.is_gene_constrained("LOWPLI") is False

    def test_get_pli_score(self, context):
        """Test getting pLI score."""
        assert context.get_pli_score("SHANK3") == 0.99
        assert context.get_pli_score("UNKNOWN") is None

    def test_is_sfari_gene(self, context):
        """Test SFARI gene check."""
        assert context.is_sfari_gene("SHANK3") is True
        assert context.is_sfari_gene("UNKNOWN") is False

    def test_is_prenatally_expressed(self, context):
        """Test prenatal expression check."""
        assert context.is_prenatally_expressed("SHANK3") is True
        assert context.is_prenatally_expressed("NOTPRENATAL") is False

    def test_is_enriched_in_cell_type(self, context):
        """Test cell type enrichment."""
        assert context.is_enriched_in_cell_type("SHANK3", "excitatory_neuron") is True
        assert context.is_enriched_in_cell_type("GAD1", "inhibitory_neuron") is True
        assert context.is_enriched_in_cell_type("SHANK3", "inhibitory_neuron") is False

    def test_get_pathway_genes(self, context):
        """Test getting pathway genes."""
        genes = context.get_pathway_genes("synaptic_transmission")
        assert "SHANK3" in genes
        assert "GRIN2A" in genes

    def test_is_chd8_target(self, context):
        """Test CHD8 target check."""
        assert context.is_chd8_target("ARID1B") is True
        assert context.is_chd8_target("SHANK3") is False

    def test_is_synaptic_gene(self, context):
        """Test synaptic gene check."""
        assert context.is_synaptic_gene("SHANK3") is True
        assert context.is_synaptic_gene("CHD8") is False

    def test_get_paralogs(self, context):
        """Test getting paralogs."""
        paralogs = context.get_paralogs("SHANK3")
        assert "SHANK1" in paralogs
        assert "SHANK2" in paralogs


class TestFiredRule:
    """Tests for FiredRule."""

    def test_fired_rule_creation(self):
        """Test fired rule creation."""
        rule = Rule(
            id="TEST",
            name="Test Rule",
            description="Test",
            conditions=[Condition(predicate="test", arguments={})],
            conclusion=Conclusion(type="test", attributes={}),
            base_confidence=0.8,
        )
        fired = FiredRule(
            rule=rule,
            bindings={"G": "SHANK3"},
            explanation="Test explanation",
        )
        assert fired.rule.id == "TEST"
        assert fired.bindings["G"] == "SHANK3"
        assert fired.confidence == 0.8  # base_confidence * modifier (1.0)

    def test_fired_rule_confidence_calculation(self):
        """Test confidence calculation with modifier."""
        rule = Rule(
            id="TEST",
            name="Test",
            description="Test",
            conditions=[Condition(predicate="test", arguments={})],
            conclusion=Conclusion(type="test", attributes={}, confidence_modifier=0.7),
            base_confidence=0.8,
        )
        fired = FiredRule(rule=rule)
        assert fired.confidence == pytest.approx(0.56)  # 0.8 * 0.7

    def test_fired_rule_to_dict(self):
        """Test fired rule serialization."""
        rule = Rule(
            id="TEST",
            name="Test",
            description="Test",
            conditions=[Condition(predicate="test", arguments={})],
            conclusion=Conclusion(type="pathway_disruption", attributes={"key": "val"}),
        )
        fired = FiredRule(
            rule=rule,
            bindings={"gene": "SHANK3"},
            evidence={"pLI": 0.99},
        )
        d = fired.to_dict()
        assert d["rule_id"] == "TEST"
        assert d["bindings"]["gene"] == "SHANK3"
        assert d["evidence"]["pLI"] == 0.99


class TestRuleEngine:
    """Tests for RuleEngine."""

    @pytest.fixture
    def context(self):
        """Create test context."""
        ctx = BiologicalContext(
            gene_constraints=MockGeneConstraints(),
            developmental_expression=MockDevelopmentalExpression(),
            single_cell_atlas=MockSingleCellAtlas(),
            sfari_genes=MockSFARIGenes(),
            pathway_db=MockPathwayDB(),
        )
        ctx.chd8_targets = {"ARID1B", "SMARCC2", "KMT2A"}
        ctx.syngo_genes = {"SHANK3", "NLGN1", "NRXN1", "GRIN2A"}
        ctx.paralog_map = {"SHANK3": ["SHANK1", "SHANK2"]}
        return ctx

    @pytest.fixture
    def engine(self, context):
        """Create rule engine with all rules."""
        rules = BiologicalRules.get_all_rules()
        return RuleEngine(rules, context)

    @pytest.fixture
    def individual_shank3_lof(self):
        """Create individual with SHANK3 LoF."""
        return IndividualData(
            sample_id="SHANK3_LOF",
            variants=[MockVariant("SHANK3", is_lof=True)],
            gene_burdens={"SHANK3": 1.0},
            pathway_scores={"synaptic_transmission": 2.5},
        )

    @pytest.fixture
    def individual_chd8_damaging(self):
        """Create individual with CHD8 damaging variant."""
        return IndividualData(
            sample_id="CHD8_DAM",
            variants=[MockVariant("CHD8", is_damaging=True)],
            gene_burdens={"CHD8": 1.0},
            pathway_scores={"chromatin_regulation": 2.0},
        )

    def test_engine_creation(self, engine):
        """Test engine creation."""
        assert len(engine.rules) >= 6
        assert engine.context is not None

    def test_rules_sorted_by_priority(self, engine):
        """Test rules are sorted by priority (descending)."""
        priorities = [r.priority for r in engine.rules]
        assert priorities == sorted(priorities, reverse=True)

    def test_evaluate_r1_constrained_lof(self, engine, individual_shank3_lof):
        """Test R1 rule fires for constrained LoF."""
        fired = engine.evaluate(individual_shank3_lof)

        # Should fire R1 (constrained LoF in developing cortex)
        r1_fired = [fr for fr in fired if fr.rule.id == "R1"]
        assert len(r1_fired) >= 1
        assert r1_fired[0].bindings.get("gene") == "SHANK3" or r1_fired[0].bindings.get("G") == "SHANK3"

    def test_evaluate_r3_chd8_cascade(self, engine, individual_chd8_damaging):
        """Test R3 rule fires for CHD8 damaging."""
        fired = engine.evaluate(individual_chd8_damaging)

        # Should fire R3 (CHD8 cascade)
        r3_fired = [fr for fr in fired if fr.rule.id == "R3"]
        assert len(r3_fired) >= 1

    def test_evaluate_r4_synaptic(self, engine, individual_shank3_lof):
        """Test R4 rule fires for synaptic gene."""
        fired = engine.evaluate(individual_shank3_lof)

        # SHANK3 is synaptic and enriched in excitatory neurons
        r4_fired = [fr for fr in fired if fr.rule.id == "R4"]
        assert len(r4_fired) >= 1

    def test_evaluate_r7_sfari(self, engine, individual_shank3_lof):
        """Test R7 rule fires for SFARI gene."""
        fired = engine.evaluate(individual_shank3_lof)

        # SHANK3 is high-confidence SFARI
        r7_fired = [fr for fr in fired if fr.rule.id == "R7"]
        assert len(r7_fired) >= 1

    def test_evaluate_specific_rules(self, engine, individual_shank3_lof):
        """Test evaluating specific rules only."""
        fired = engine.evaluate(individual_shank3_lof, rule_ids=["R1", "R7"])

        rule_ids = [fr.rule.id for fr in fired]
        # Should only have R1 and R7
        for rid in rule_ids:
            assert rid in ["R1", "R7"]

    def test_evaluate_batch(self, engine, individual_shank3_lof, individual_chd8_damaging):
        """Test batch evaluation."""
        cohort = [individual_shank3_lof, individual_chd8_damaging]
        results = engine.evaluate_batch(cohort)

        assert "SHANK3_LOF" in results
        assert "CHD8_DAM" in results
        assert len(results["SHANK3_LOF"]) > 0
        assert len(results["CHD8_DAM"]) > 0

    def test_no_rules_fire_empty_individual(self, engine):
        """Test that no rules fire for empty individual."""
        empty = IndividualData(sample_id="EMPTY")
        fired = engine.evaluate(empty)
        assert len(fired) == 0

    def test_explain_fired_rule(self, engine, individual_shank3_lof):
        """Test generating explanation for fired rule."""
        fired = engine.evaluate(individual_shank3_lof)
        assert len(fired) > 0

        explanation = engine.explain(fired[0])
        assert len(explanation) > 0
        assert fired[0].rule.name in explanation

    def test_get_summary(self, engine, individual_shank3_lof):
        """Test getting summary of fired rules."""
        fired = engine.evaluate(individual_shank3_lof)
        summary = engine.get_summary(fired)

        assert "total_fired" in summary
        assert "by_conclusion_type" in summary
        assert "by_rule" in summary
        assert "genes_affected" in summary
        assert summary["total_fired"] == len(fired)

    def test_get_rules_by_type(self, engine):
        """Test getting rules by conclusion type."""
        pathway_rules = engine.get_rules_by_type(ConclusionType.PATHWAY_DISRUPTION)
        assert len(pathway_rules) >= 1

        subtype_rules = engine.get_rules_by_type(ConclusionType.SUBTYPE_INDICATOR)
        assert len(subtype_rules) >= 1
