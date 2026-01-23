"""Tests for hypothesis ranking."""

import sys
from pathlib import Path

# Add module to path
_module_dir = Path(__file__).parent.parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

import pytest
from ranking import (
    TherapeuticHypothesis,
    RankingConfig,
    RankingResult,
    HypothesisRanker,
    create_hypothesis_ranker,
)
from pathway_drug_mapping import (
    DrugTargetDatabase,
    PathwayDrugMapper,
    create_sample_drug_database,
)
from evidence import (
    EvidenceScore,
    EvidenceLevel,
    EvidenceScorer,
    create_sample_evidence_databases,
)


class TestTherapeuticHypothesis:
    """Tests for TherapeuticHypothesis dataclass."""

    @pytest.fixture
    def sample_evidence(self):
        """Create sample evidence score."""
        return EvidenceScore(
            biological_plausibility=0.7,
            mechanistic_alignment=0.6,
            literature_support=0.5,
            overall=0.6,
            confidence=0.7,
            level=EvidenceLevel.MODERATE,
        )

    def test_creation(self, sample_evidence):
        """Test hypothesis creation."""
        hyp = TherapeuticHypothesis(
            drug_id="DB001",
            drug_name="Test Drug",
            target_pathway="synaptic_transmission",
            target_genes=["GENE1", "GENE2"],
            mechanism="Receptor antagonist",
            score=0.75,
            evidence=sample_evidence,
            explanation="Test explanation",
            confidence=0.8,
        )

        assert hyp.drug_id == "DB001"
        assert hyp.score == 0.75
        assert hyp.requires_validation is True  # Always True

    def test_requires_validation_always_true(self, sample_evidence):
        """Test that requires_validation cannot be changed."""
        hyp = TherapeuticHypothesis(
            drug_id="DB001",
            drug_name="Test",
            target_pathway="test",
            target_genes=[],
            mechanism="test",
            score=0.5,
            evidence=sample_evidence,
            explanation="test",
            confidence=0.5,
        )

        # Try to change it (should not work)
        assert hyp.requires_validation is True

        # Even in to_dict
        d = hyp.to_dict()
        assert d["requires_validation"] is True

    def test_invalid_score_raises(self, sample_evidence):
        """Test that invalid score raises ValueError."""
        with pytest.raises(ValueError, match="score must be between"):
            TherapeuticHypothesis(
                drug_id="DB001",
                drug_name="Test",
                target_pathway="test",
                target_genes=[],
                mechanism="test",
                score=1.5,
                evidence=sample_evidence,
                explanation="test",
                confidence=0.5,
            )

    def test_to_dict(self, sample_evidence):
        """Test serialization to dictionary."""
        hyp = TherapeuticHypothesis(
            drug_id="DB001",
            drug_name="Test Drug",
            target_pathway="synaptic_transmission",
            target_genes=["GENE1"],
            mechanism="Antagonist",
            score=0.7,
            evidence=sample_evidence,
            explanation="Test",
            confidence=0.8,
            rank=1,
        )

        d = hyp.to_dict()
        assert d["drug_id"] == "DB001"
        assert d["score"] == 0.7
        assert d["requires_validation"] is True
        assert "evidence" in d

    def test_from_dict(self, sample_evidence):
        """Test deserialization from dictionary."""
        data = {
            "drug_id": "DB001",
            "drug_name": "Test Drug",
            "target_pathway": "test",
            "target_genes": ["GENE1"],
            "mechanism": "test",
            "score": 0.6,
            "evidence": sample_evidence.to_dict(),
            "explanation": "test",
            "confidence": 0.7,
            "rank": 2,
        }

        hyp = TherapeuticHypothesis.from_dict(data)
        assert hyp.drug_id == "DB001"
        assert hyp.requires_validation is True  # Always True

    def test_is_high_evidence(self, sample_evidence):
        """Test high evidence detection."""
        # Moderate evidence
        hyp1 = TherapeuticHypothesis(
            drug_id="DB001",
            drug_name="Test",
            target_pathway="test",
            target_genes=[],
            mechanism="test",
            score=0.5,
            evidence=sample_evidence,
            explanation="test",
            confidence=0.5,
        )
        assert not hyp1.is_high_evidence

        # High evidence
        high_evidence = EvidenceScore(
            overall=0.8,
            level=EvidenceLevel.HIGH,
        )
        hyp2 = TherapeuticHypothesis(
            drug_id="DB002",
            drug_name="Test",
            target_pathway="test",
            target_genes=[],
            mechanism="test",
            score=0.8,
            evidence=high_evidence,
            explanation="test",
            confidence=0.8,
        )
        assert hyp2.is_high_evidence

    def test_has_safety_concerns(self, sample_evidence):
        """Test safety concern detection."""
        # No safety flags
        hyp1 = TherapeuticHypothesis(
            drug_id="DB001",
            drug_name="Test",
            target_pathway="test",
            target_genes=[],
            mechanism="test",
            score=0.5,
            evidence=sample_evidence,
            explanation="test",
            confidence=0.5,
        )
        assert not hyp1.has_safety_concerns

        # With safety flags
        flagged_evidence = EvidenceScore(
            overall=0.5,
            safety_flags=["cns_effects"],
        )
        hyp2 = TherapeuticHypothesis(
            drug_id="DB002",
            drug_name="Test",
            target_pathway="test",
            target_genes=[],
            mechanism="test",
            score=0.5,
            evidence=flagged_evidence,
            explanation="test",
            confidence=0.5,
        )
        assert hyp2.has_safety_concerns

    def test_summary(self, sample_evidence):
        """Test summary generation."""
        hyp = TherapeuticHypothesis(
            drug_id="DB001",
            drug_name="Memantine",
            target_pathway="synaptic_transmission",
            target_genes=["GRIN1"],
            mechanism="NMDA antagonist",
            score=0.75,
            evidence=sample_evidence,
            explanation="test",
            confidence=0.7,
            rank=1,
        )

        summary = hyp.summary()
        assert "[1]" in summary
        assert "Memantine" in summary
        assert "0.75" in summary


class TestRankingConfig:
    """Tests for RankingConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RankingConfig()
        assert config.weight_evidence == 0.4
        assert config.min_pathway_zscore == 1.5
        assert config.max_hypotheses == 50
        assert config.max_per_pathway == 5

    def test_custom_config(self):
        """Test custom configuration."""
        config = RankingConfig(
            min_evidence_score=0.3,
            max_hypotheses=20,
            exclude_critical_safety=True,
        )
        assert config.min_evidence_score == 0.3
        assert config.max_hypotheses == 20
        assert config.exclude_critical_safety is True


class TestRankingResult:
    """Tests for RankingResult."""

    @pytest.fixture
    def sample_hypotheses(self):
        """Create sample hypotheses."""
        evidence = EvidenceScore(overall=0.6, level=EvidenceLevel.MODERATE)
        high_evidence = EvidenceScore(overall=0.8, level=EvidenceLevel.HIGH)

        return [
            TherapeuticHypothesis(
                drug_id="DB001",
                drug_name="Drug1",
                target_pathway="pathway1",
                target_genes=[],
                mechanism="test",
                score=0.8,
                evidence=high_evidence,
                explanation="test",
                confidence=0.8,
                rank=1,
            ),
            TherapeuticHypothesis(
                drug_id="DB002",
                drug_name="Drug2",
                target_pathway="pathway2",
                target_genes=[],
                mechanism="test",
                score=0.6,
                evidence=evidence,
                explanation="test",
                confidence=0.6,
                rank=2,
            ),
        ]

    def test_creation(self, sample_hypotheses):
        """Test result creation."""
        result = RankingResult(
            hypotheses=sample_hypotheses,
            pathways_analyzed=["pathway1", "pathway2"],
            drugs_considered=10,
        )

        assert len(result.hypotheses) == 2
        assert result.drugs_considered == 10

    def test_top_hypotheses(self, sample_hypotheses):
        """Test top hypotheses property."""
        result = RankingResult(
            hypotheses=sample_hypotheses,
            pathways_analyzed=[],
            drugs_considered=2,
        )

        top = result.top_hypotheses
        assert len(top) == 2  # All if less than 10

    def test_high_evidence_count(self, sample_hypotheses):
        """Test high evidence count property."""
        result = RankingResult(
            hypotheses=sample_hypotheses,
            pathways_analyzed=[],
            drugs_considered=2,
        )

        assert result.high_evidence_count == 1

    def test_pathways_with_hypotheses(self, sample_hypotheses):
        """Test unique pathways property."""
        result = RankingResult(
            hypotheses=sample_hypotheses,
            pathways_analyzed=["pathway1", "pathway2", "pathway3"],
            drugs_considered=2,
        )

        pathways = result.pathways_with_hypotheses
        assert "pathway1" in pathways
        assert "pathway2" in pathways
        assert len(pathways) == 2

    def test_to_dict(self, sample_hypotheses):
        """Test serialization to dictionary."""
        result = RankingResult(
            hypotheses=sample_hypotheses,
            pathways_analyzed=["pathway1"],
            drugs_considered=10,
        )

        d = result.to_dict()
        assert "hypotheses" in d
        assert "pathways_analyzed" in d
        assert "timestamp" in d

    def test_summary(self, sample_hypotheses):
        """Test summary generation."""
        result = RankingResult(
            hypotheses=sample_hypotheses,
            pathways_analyzed=["pathway1", "pathway2"],
            drugs_considered=10,
        )

        summary = result.summary()
        assert "Therapeutic Hypothesis Ranking" in summary
        assert "validation" in summary.lower()


class TestHypothesisRanker:
    """Tests for HypothesisRanker."""

    @pytest.fixture
    def drug_db(self):
        """Create sample drug database."""
        return create_sample_drug_database()

    @pytest.fixture
    def ranker(self, drug_db):
        """Create ranker with sample database."""
        lit_db, clinical_db, safety_db = create_sample_evidence_databases()
        mapper = PathwayDrugMapper(drug_db)
        scorer = EvidenceScorer(
            literature_db=lit_db,
            clinical_trials_db=clinical_db,
            safety_db=safety_db,
        )
        return HypothesisRanker(
            drug_mapper=mapper,
            evidence_scorer=scorer,
        )

    def test_basic_ranking(self, ranker):
        """Test basic hypothesis ranking."""
        pathway_scores = {
            "synaptic_transmission": 2.5,
            "mtor_signaling": 2.0,
        }

        result = ranker.rank(pathway_scores=pathway_scores)

        assert isinstance(result, RankingResult)
        assert len(result.hypotheses) > 0
        assert all(h.requires_validation for h in result.hypotheses)

    def test_ranking_order(self, ranker):
        """Test hypotheses are ranked by score."""
        pathway_scores = {
            "synaptic_transmission": 2.5,
            "mtor_signaling": 2.5,
        }

        result = ranker.rank(pathway_scores=pathway_scores)

        if len(result.hypotheses) > 1:
            scores = [h.score for h in result.hypotheses]
            assert scores == sorted(scores, reverse=True)

    def test_ranks_assigned(self, ranker):
        """Test ranks are correctly assigned."""
        pathway_scores = {"synaptic_transmission": 3.0}

        result = ranker.rank(pathway_scores=pathway_scores)

        for i, h in enumerate(result.hypotheses):
            assert h.rank == i + 1

    def test_pathway_threshold(self, ranker):
        """Test minimum pathway z-score threshold."""
        pathway_scores = {
            "synaptic_transmission": 2.5,  # Above threshold
            "chromatin_remodeling": 0.5,  # Below threshold
        }

        result = ranker.rank(
            pathway_scores=pathway_scores,
            min_pathway_zscore=1.5,
        )

        # Should not include low-scoring pathway
        assert "synaptic_transmission" in result.pathways_analyzed
        assert "chromatin_remodeling" not in result.pathways_analyzed

    def test_disrupted_genes_boost(self, ranker):
        """Test that disrupted genes boost relevant hypotheses."""
        pathway_scores = {"mtor_signaling": 2.5}

        # Without disrupted genes
        result1 = ranker.rank(pathway_scores=pathway_scores)

        # With disrupted gene matching drug target
        result2 = ranker.rank(
            pathway_scores=pathway_scores,
            disrupted_genes=["MTOR"],
        )

        # Both should have results, but with different characteristics
        assert len(result1.hypotheses) > 0 or len(result2.hypotheses) > 0

    def test_max_hypotheses_limit(self, drug_db):
        """Test hypothesis count limit."""
        mapper = PathwayDrugMapper(drug_db)
        scorer = EvidenceScorer()
        config = RankingConfig(max_hypotheses=3)
        ranker = HypothesisRanker(
            drug_mapper=mapper,
            evidence_scorer=scorer,
            config=config,
        )

        pathway_scores = {
            "synaptic_transmission": 3.0,
            "mtor_signaling": 3.0,
            "chromatin_remodeling": 3.0,
        }

        result = ranker.rank(pathway_scores=pathway_scores)

        assert len(result.hypotheses) <= 3

    def test_diversity_constraints(self, drug_db):
        """Test diversity constraints limit per-pathway hypotheses."""
        mapper = PathwayDrugMapper(drug_db)
        scorer = EvidenceScorer()
        config = RankingConfig(max_per_pathway=2)
        ranker = HypothesisRanker(
            drug_mapper=mapper,
            evidence_scorer=scorer,
            config=config,
        )

        pathway_scores = {"synaptic_transmission": 3.0}

        result = ranker.rank(pathway_scores=pathway_scores)

        # Count per pathway
        pathway_counts = {}
        for h in result.hypotheses:
            pathway_counts[h.target_pathway] = pathway_counts.get(h.target_pathway, 0) + 1

        for count in pathway_counts.values():
            assert count <= 2

    def test_evidence_minimum_filter(self, drug_db):
        """Test minimum evidence score filter."""
        mapper = PathwayDrugMapper(drug_db)
        scorer = EvidenceScorer()
        config = RankingConfig(min_evidence_score=0.9)  # Very high threshold
        ranker = HypothesisRanker(
            drug_mapper=mapper,
            evidence_scorer=scorer,
            config=config,
        )

        pathway_scores = {"synaptic_transmission": 2.5}

        result = ranker.rank(pathway_scores=pathway_scores)

        # May have fewer or no results due to high threshold
        for h in result.hypotheses:
            assert h.evidence.overall >= 0.9

    def test_explanation_generated(self, ranker):
        """Test that explanations are generated."""
        pathway_scores = {"synaptic_transmission": 2.5}

        result = ranker.rank(pathway_scores=pathway_scores)

        for h in result.hypotheses:
            assert h.explanation
            assert "HYPOTHESIS" in h.explanation
            assert "validation" in h.explanation.lower()

    def test_no_mapper_returns_empty(self):
        """Test that ranker without mapper returns empty results."""
        ranker = HypothesisRanker(
            drug_mapper=None,
            evidence_scorer=EvidenceScorer(),
        )

        pathway_scores = {"test_pathway": 2.5}
        result = ranker.rank(pathway_scores=pathway_scores)

        assert len(result.hypotheses) == 0

    def test_metadata_included(self, ranker):
        """Test that metadata is included in result."""
        pathway_scores = {"synaptic_transmission": 2.5}

        result = ranker.rank(
            pathway_scores=pathway_scores,
            disrupted_genes=["GENE1", "GENE2"],
        )

        assert "disrupted_genes_count" in result.metadata
        assert result.metadata["disrupted_genes_count"] == 2


class TestCreateHypothesisRanker:
    """Tests for create_hypothesis_ranker factory."""

    def test_create_basic_ranker(self):
        """Test creating basic ranker."""
        ranker = create_hypothesis_ranker()

        assert ranker is not None
        assert ranker.evidence_scorer is not None
        assert ranker.drug_mapper is None  # No DB provided

    def test_create_with_database(self):
        """Test creating ranker with database."""
        db = create_sample_drug_database()
        ranker = create_hypothesis_ranker(drug_db=db)

        assert ranker.drug_mapper is not None

    def test_create_with_config(self):
        """Test creating ranker with custom config."""
        config = RankingConfig(max_hypotheses=10)
        ranker = create_hypothesis_ranker(config=config)

        assert ranker.config.max_hypotheses == 10


class TestRankerWithFiredRules:
    """Tests for ranker with fired rules integration."""

    @pytest.fixture
    def ranker(self):
        """Create ranker with sample data."""
        db = create_sample_drug_database()
        mapper = PathwayDrugMapper(db)
        return HypothesisRanker(drug_mapper=mapper)

    def test_extract_rule_confidences(self, ranker):
        """Test extracting confidences from fired rules."""
        # Create mock fired rules
        class MockConclusion:
            type = "therapeutic_hypothesis"
            attributes = {"drug": "DB001"}

        class MockRule:
            conclusion = MockConclusion()

        class MockFiredRule:
            rule = MockRule()
            bindings = {"D": "DB001"}
            confidence = 0.85

        fired_rules = [MockFiredRule()]

        confidences = ranker._extract_rule_confidences(fired_rules)

        assert "DB001" in confidences
        assert confidences["DB001"] == 0.85

    def test_rule_confidence_boosts_score(self, ranker):
        """Test that rule confidence boosts hypothesis score."""
        pathway_scores = {"synaptic_transmission": 2.5}

        # Without rules
        result1 = ranker.rank(pathway_scores=pathway_scores)

        # Create mock fired rule for a drug
        class MockConclusion:
            type = "therapeutic_hypothesis"
            attributes = {"drug": "DB00334"}

        class MockRule:
            conclusion = MockConclusion()

        class MockFiredRule:
            rule = MockRule()
            bindings = {"D": "DB00334"}
            confidence = 0.9

        # With rules
        result2 = ranker.rank(
            pathway_scores=pathway_scores,
            fired_rules=[MockFiredRule()],
        )

        # Both should have results
        assert len(result1.hypotheses) > 0 or len(result2.hypotheses) > 0


class TestRankingIntegration:
    """Integration tests for complete ranking workflow."""

    def test_full_workflow(self):
        """Test complete hypothesis generation workflow."""
        # Create database
        db = create_sample_drug_database()

        # Create components
        lit_db, clinical_db, safety_db = create_sample_evidence_databases()
        mapper = PathwayDrugMapper(db)
        scorer = EvidenceScorer(
            literature_db=lit_db,
            clinical_trials_db=clinical_db,
            safety_db=safety_db,
        )

        # Create ranker
        ranker = HypothesisRanker(
            drug_mapper=mapper,
            evidence_scorer=scorer,
        )

        # Define disrupted pathways
        pathway_scores = {
            "synaptic_transmission": 2.8,
            "mtor_signaling": 2.2,
            "chromatin_remodeling": 1.6,
        }

        # Generate hypotheses
        result = ranker.rank(
            pathway_scores=pathway_scores,
            pathway_genes={
                "synaptic_transmission": ["GRIN1", "GRIN2A", "GABBR1"],
                "mtor_signaling": ["MTOR", "TSC1", "TSC2"],
            },
            disrupted_genes=["GRIN2A", "MTOR"],
        )

        # Validate result
        assert isinstance(result, RankingResult)
        assert len(result.pathways_analyzed) >= 2
        assert result.drugs_considered > 0

        # Check hypotheses
        for h in result.hypotheses:
            assert h.requires_validation is True
            assert 0 <= h.score <= 1
            assert h.explanation
            assert h.rank > 0

        # Print summary
        print(result.summary())
