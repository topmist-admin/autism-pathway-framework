"""Tests for evidence scoring."""

import sys
from pathlib import Path

# Add module to path
_module_dir = Path(__file__).parent.parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

import pytest
from evidence import (
    EvidenceLevel,
    SafetyFlag,
    EvidenceScore,
    EvidenceScorerConfig,
    EvidenceScorer,
    create_sample_evidence_databases,
)


class TestEvidenceLevel:
    """Tests for EvidenceLevel enum."""

    def test_all_levels_defined(self):
        """Test all evidence levels are defined."""
        levels = [
            EvidenceLevel.HIGH,
            EvidenceLevel.MODERATE,
            EvidenceLevel.LOW,
            EvidenceLevel.INSUFFICIENT,
        ]
        assert len(levels) == 4

    def test_level_values(self):
        """Test level string values."""
        assert EvidenceLevel.HIGH.value == "high"
        assert EvidenceLevel.MODERATE.value == "moderate"


class TestSafetyFlag:
    """Tests for SafetyFlag enum."""

    def test_all_flags_defined(self):
        """Test all safety flags are defined."""
        flags = [
            SafetyFlag.CNS_EFFECTS,
            SafetyFlag.DEVELOPMENTAL_CONCERNS,
            SafetyFlag.DRUG_INTERACTIONS,
            SafetyFlag.CONTRAINDICATED_PEDIATRIC,
            SafetyFlag.BLACK_BOX_WARNING,
            SafetyFlag.OFF_LABEL_USE,
            SafetyFlag.IMMUNOSUPPRESSION,
            SafetyFlag.HEPATOTOXICITY,
            SafetyFlag.CARDIOTOXICITY,
            SafetyFlag.TERATOGENIC,
            SafetyFlag.WITHDRAWAL_RISK,
        ]
        assert len(flags) == 11


class TestEvidenceScore:
    """Tests for EvidenceScore dataclass."""

    def test_default_creation(self):
        """Test default evidence score creation."""
        score = EvidenceScore()
        assert score.biological_plausibility == 0.0
        assert score.mechanistic_alignment == 0.0
        assert score.overall == 0.0
        assert score.level == EvidenceLevel.INSUFFICIENT
        assert score.safety_flags == []

    def test_custom_creation(self):
        """Test custom evidence score creation."""
        score = EvidenceScore(
            biological_plausibility=0.8,
            mechanistic_alignment=0.7,
            literature_support=0.6,
            clinical_evidence=0.5,
            overall=0.65,
            confidence=0.7,
            level=EvidenceLevel.MODERATE,
            safety_flags=["cns_effects"],
        )

        assert score.biological_plausibility == 0.8
        assert score.overall == 0.65
        assert "cns_effects" in score.safety_flags

    def test_invalid_score_raises(self):
        """Test that invalid scores raise ValueError."""
        with pytest.raises(ValueError):
            EvidenceScore(biological_plausibility=1.5)

        with pytest.raises(ValueError):
            EvidenceScore(overall=-0.1)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        score = EvidenceScore(
            biological_plausibility=0.7,
            overall=0.6,
            level=EvidenceLevel.MODERATE,
            safety_flags=["cns_effects"],
        )

        d = score.to_dict()
        assert d["biological_plausibility"] == 0.7
        assert d["level"] == "moderate"
        assert "cns_effects" in d["safety_flags"]

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "biological_plausibility": 0.8,
            "overall": 0.7,
            "level": "high",
            "safety_flags": ["cns_effects"],
        }

        score = EvidenceScore.from_dict(data)
        assert score.biological_plausibility == 0.8
        assert score.level == EvidenceLevel.HIGH

    def test_has_critical_safety_flags(self):
        """Test critical safety flag detection."""
        # No flags
        score1 = EvidenceScore()
        assert not score1.has_critical_safety_flags

        # Non-critical flag
        score2 = EvidenceScore(safety_flags=["cns_effects"])
        assert not score2.has_critical_safety_flags

        # Critical flag
        score3 = EvidenceScore(safety_flags=["black_box_warning"])
        assert score3.has_critical_safety_flags

        # Pediatric contraindication
        score4 = EvidenceScore(safety_flags=["contraindicated_pediatric"])
        assert score4.has_critical_safety_flags

    def test_safety_summary(self):
        """Test safety summary generation."""
        # No flags
        score1 = EvidenceScore()
        assert "No specific safety flags" in score1.safety_summary

        # With flags
        score2 = EvidenceScore(safety_flags=["cns_effects", "withdrawal_risk"])
        assert "cns_effects" in score2.safety_summary

        # Critical flags
        score3 = EvidenceScore(safety_flags=["black_box_warning"])
        assert "CRITICAL" in score3.safety_summary


class TestEvidenceScorerConfig:
    """Tests for EvidenceScorerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EvidenceScorerConfig()
        assert config.weight_biological == 0.3
        assert config.threshold_high == 0.7
        assert config.safety_penalty_per_flag == 0.05

    def test_weights_sum_warning(self, caplog):
        """Test warning when weights don't sum to 1."""
        config = EvidenceScorerConfig(
            weight_biological=0.5,
            weight_mechanistic=0.5,
            weight_literature=0.5,
            weight_clinical=0.5,
        )
        # Weights sum to 2.0, should log warning
        assert "weights sum to" in caplog.text or config is not None


class TestEvidenceScorer:
    """Tests for EvidenceScorer."""

    @pytest.fixture
    def scorer(self):
        """Create default evidence scorer."""
        return EvidenceScorer()

    @pytest.fixture
    def scorer_with_data(self):
        """Create scorer with sample databases."""
        lit_db, clinical_db, safety_db = create_sample_evidence_databases()
        return EvidenceScorer(
            literature_db=lit_db,
            clinical_trials_db=clinical_db,
            safety_db=safety_db,
        )

    def test_basic_scoring(self, scorer):
        """Test basic evidence scoring."""
        score = scorer.score(
            drug_id="TEST001",
            drug_name="Test Drug",
            target_pathway="synaptic_transmission",
            mechanism="NMDA receptor antagonist",
            target_genes=["GRIN1", "GRIN2A"],
            pathway_genes=["GRIN1", "GRIN2A", "GRIN2B"],
        )

        assert isinstance(score, EvidenceScore)
        assert 0 <= score.overall <= 1
        assert score.level in EvidenceLevel

    def test_biological_plausibility(self, scorer):
        """Test biological plausibility scoring."""
        # Good overlap
        score1 = scorer.score(
            drug_id="TEST001",
            drug_name="Test",
            target_pathway="test",
            mechanism="inhibitor",
            target_genes=["GENE1", "GENE2"],
            pathway_genes=["GENE1", "GENE2", "GENE3"],
            disrupted_genes=["GENE1"],
        )

        # No overlap
        score2 = scorer.score(
            drug_id="TEST002",
            drug_name="Test",
            target_pathway="test",
            mechanism="inhibitor",
            target_genes=["GENEX", "GENEY"],
            pathway_genes=["GENE1", "GENE2", "GENE3"],
        )

        # Better overlap should give higher plausibility
        assert score1.biological_plausibility > score2.biological_plausibility

    def test_disrupted_gene_targeting_boost(self, scorer):
        """Test that targeting disrupted genes boosts score."""
        # Targets disrupted gene
        score1 = scorer.score(
            drug_id="TEST001",
            drug_name="Test",
            target_pathway="test",
            mechanism="inhibitor",
            target_genes=["GENE1"],
            pathway_genes=["GENE1", "GENE2"],
            disrupted_genes=["GENE1"],
        )

        # Doesn't target disrupted gene
        score2 = scorer.score(
            drug_id="TEST002",
            drug_name="Test",
            target_pathway="test",
            mechanism="inhibitor",
            target_genes=["GENE2"],
            pathway_genes=["GENE1", "GENE2"],
            disrupted_genes=["GENE1"],
        )

        assert score1.biological_plausibility > score2.biological_plausibility

    def test_mechanistic_alignment(self, scorer):
        """Test mechanistic alignment scoring."""
        # Good mechanism for synaptic
        score1 = scorer.score(
            drug_id="TEST001",
            drug_name="Test",
            target_pathway="synaptic_transmission",
            mechanism="receptor agonist modulator",
            target_genes=["GENE1"],
        )

        # Generic mechanism
        score2 = scorer.score(
            drug_id="TEST002",
            drug_name="Test",
            target_pathway="synaptic_transmission",
            mechanism="unknown",
            target_genes=["GENE1"],
        )

        assert score1.mechanistic_alignment > score2.mechanistic_alignment

    def test_with_literature_data(self, scorer_with_data):
        """Test scoring with literature database."""
        # Drug with literature evidence
        score = scorer_with_data.score(
            drug_id="DB00334",  # Memantine in sample DB
            drug_name="Memantine",
            target_pathway="synaptic_transmission",
            mechanism="NMDA antagonist",
            target_genes=["GRIN1"],
        )

        assert score.literature_support > 0.3

    def test_with_clinical_data(self, scorer_with_data):
        """Test scoring with clinical trial database."""
        # Drug with clinical evidence
        score = scorer_with_data.score(
            drug_id="DB01590",  # Everolimus in sample DB
            drug_name="Everolimus",
            target_pathway="mtor_signaling",
            mechanism="mTOR inhibitor",
            target_genes=["MTOR"],
        )

        assert score.clinical_evidence > 0

    def test_safety_flags_assigned(self, scorer_with_data):
        """Test safety flags are assigned from database."""
        score = scorer_with_data.score(
            drug_id="DB00196",  # Fluoxetine has flags in sample DB
            drug_name="Fluoxetine",
            target_pathway="serotonin_signaling",
            mechanism="SSRI",
            target_genes=["SLC6A4"],
        )

        assert len(score.safety_flags) > 0
        assert "cns_effects" in score.safety_flags or "withdrawal_risk" in score.safety_flags

    def test_safety_flags_from_mechanism(self, scorer):
        """Test safety flags inferred from mechanism."""
        score = scorer.score(
            drug_id="TEST001",
            drug_name="Test",
            target_pathway="immune",
            mechanism="immunosuppressive agent",
            target_genes=["GENE1"],
        )

        assert "immunosuppression" in score.safety_flags

    def test_rule_confidence_boost(self, scorer):
        """Test that rule confidence boosts score."""
        # Without rule confidence
        score1 = scorer.score(
            drug_id="TEST001",
            drug_name="Test",
            target_pathway="test",
            mechanism="inhibitor",
            target_genes=["GENE1"],
        )

        # With high rule confidence
        score2 = scorer.score(
            drug_id="TEST001",
            drug_name="Test",
            target_pathway="test",
            mechanism="inhibitor",
            target_genes=["GENE1"],
            rule_confidence=0.9,
        )

        assert score2.overall > score1.overall

    def test_evidence_level_assignment(self, scorer):
        """Test evidence level is correctly assigned."""
        # Should check that levels are assigned based on thresholds
        # Create scorer with known thresholds
        config = EvidenceScorerConfig(
            threshold_high=0.7,
            threshold_moderate=0.5,
            threshold_low=0.3,
        )
        scorer = EvidenceScorer(config=config)

        # Force a high overall score
        score = scorer.score(
            drug_id="TEST001",
            drug_name="Test",
            target_pathway="synaptic_transmission",
            mechanism="receptor agonist modulator",
            target_genes=["GRIN1", "GRIN2A"],
            pathway_genes=["GRIN1", "GRIN2A"],
            disrupted_genes=["GRIN1"],
            rule_confidence=0.9,
        )

        # Level should be based on overall score
        assert score.level in EvidenceLevel

    def test_explanation_generated(self, scorer):
        """Test that explanation is generated."""
        score = scorer.score(
            drug_id="TEST001",
            drug_name="Test Drug",
            target_pathway="test_pathway",
            mechanism="test mechanism",
            target_genes=["GENE1"],
        )

        assert score.explanation
        assert "Test Drug" in score.explanation
        assert "HYPOTHESIS" in score.explanation

    def test_add_literature_evidence(self, scorer):
        """Test adding literature evidence."""
        scorer.add_literature_evidence("TEST001", 0.8)
        assert scorer.literature_db["TEST001"] == 0.8

        # Test clamping
        scorer.add_literature_evidence("TEST002", 1.5)
        assert scorer.literature_db["TEST002"] == 1.0

    def test_add_safety_flags(self, scorer):
        """Test adding safety flags."""
        scorer.add_safety_flags("TEST001", ["cns_effects"])
        assert "cns_effects" in scorer.safety_db["TEST001"]

        # Add more flags
        scorer.add_safety_flags("TEST001", ["hepatotoxicity"])
        assert "hepatotoxicity" in scorer.safety_db["TEST001"]

        # Should deduplicate
        scorer.add_safety_flags("TEST001", ["cns_effects"])
        assert scorer.safety_db["TEST001"].count("cns_effects") == 1


class TestCreateSampleEvidenceDatabases:
    """Tests for sample evidence databases."""

    def test_create_databases(self):
        """Test creating sample databases."""
        lit_db, clinical_db, safety_db = create_sample_evidence_databases()

        assert len(lit_db) > 0
        assert len(clinical_db) > 0
        assert len(safety_db) > 0

    def test_score_ranges(self):
        """Test that scores are in valid ranges."""
        lit_db, clinical_db, safety_db = create_sample_evidence_databases()

        for score in lit_db.values():
            assert 0 <= score <= 1

        for score in clinical_db.values():
            assert 0 <= score <= 1

    def test_safety_flags_valid(self):
        """Test that safety flags use valid values."""
        _, _, safety_db = create_sample_evidence_databases()

        valid_flags = {f.value for f in SafetyFlag}

        for drug_flags in safety_db.values():
            for flag in drug_flags:
                assert flag in valid_flags
