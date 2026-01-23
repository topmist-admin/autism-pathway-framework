"""
Tests for Session 18 Pipelines: Therapeutic Hypothesis & Causal Analysis.

These tests verify the pipeline components and configuration using mock data.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pytest

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# =============================================================================
# Test TherapeuticConfig
# =============================================================================

class TestTherapeuticConfig:
    """Tests for TherapeuticConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from pipelines.therapeutic_hypothesis import TherapeuticConfig

        config = TherapeuticConfig()
        assert config.enable_rules is True
        assert config.enable_causal_validation is True
        assert config.use_sample_database is True
        assert config.min_pathway_zscore == 1.5
        assert config.max_hypotheses == 50

    def test_custom_values(self):
        """Test custom configuration values."""
        from pipelines.therapeutic_hypothesis import TherapeuticConfig

        config = TherapeuticConfig(
            enable_rules=False,
            enable_causal_validation=False,
            min_pathway_zscore=2.0,
            max_hypotheses=100,
        )
        assert config.enable_rules is False
        assert config.enable_causal_validation is False
        assert config.min_pathway_zscore == 2.0
        assert config.max_hypotheses == 100


class TestTherapeuticPipelineConfig:
    """Tests for TherapeuticPipelineConfig dataclass."""

    def test_creation(self):
        """Test creating complete pipeline configuration."""
        from pipelines.therapeutic_hypothesis import (
            TherapeuticPipelineConfig,
            TherapeuticConfig,
        )
        from pipelines.subtype_discovery import DataConfig

        config = TherapeuticPipelineConfig(
            data=DataConfig(vcf_path="test.vcf", pathway_gmt_path="test.gmt"),
            therapeutic=TherapeuticConfig(enable_rules=True),
            verbose=False,
        )

        assert config.data.vcf_path == "test.vcf"
        assert config.therapeutic.enable_rules is True
        assert config.verbose is False


# =============================================================================
# Test CausalAnalysisConfig
# =============================================================================

class TestCausalAnalysisConfig:
    """Tests for CausalAnalysisConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from pipelines.causal_analysis import CausalAnalysisConfig

        config = CausalAnalysisConfig()
        assert config.sample_id == "individual"
        assert config.variant_genes == []
        assert config.disrupted_pathways == []
        assert config.use_sample_model is True
        assert config.run_intervention_analysis is True
        assert config.run_counterfactual_analysis is True
        assert config.run_mediation_analysis is True

    def test_with_genetic_data(self):
        """Test configuration with genetic data."""
        from pipelines.causal_analysis import CausalAnalysisConfig

        config = CausalAnalysisConfig(
            sample_id="PATIENT_001",
            variant_genes=["SHANK3", "CHD8", "SCN2A"],
            disrupted_pathways=["synaptic_transmission", "chromatin_remodeling"],
            gene_effects={"SHANK3": 0.8, "CHD8": 0.7},
        )

        assert config.sample_id == "PATIENT_001"
        assert len(config.variant_genes) == 3
        assert len(config.disrupted_pathways) == 2
        assert config.gene_effects["SHANK3"] == 0.8


# =============================================================================
# Test Result Dataclasses
# =============================================================================

class TestIndividualAnalysis:
    """Tests for IndividualAnalysis dataclass."""

    def test_creation(self):
        """Test creating individual analysis result."""
        from pipelines.therapeutic_hypothesis import IndividualAnalysis

        analysis = IndividualAnalysis(
            sample_id="S1",
            fired_rules=[Mock()],
            subtype_assignment=2,
            disrupted_pathways=["pathway1", "pathway2"],
        )

        assert analysis.sample_id == "S1"
        assert analysis.n_rules_fired == 1
        assert analysis.subtype_assignment == 2
        assert len(analysis.disrupted_pathways) == 2

    def test_summary(self):
        """Test summary generation."""
        from pipelines.therapeutic_hypothesis import IndividualAnalysis

        analysis = IndividualAnalysis(
            sample_id="S1",
            fired_rules=[],
            subtype_assignment=0,
            disrupted_pathways=["pathway1"],
        )

        summary = analysis.summary()
        assert "Individual: S1" in summary
        assert "Subtype: 0" in summary


class TestCausalValidation:
    """Tests for CausalValidation dataclass."""

    def test_creation(self):
        """Test creating causal validation result."""
        from pipelines.therapeutic_hypothesis import CausalValidation

        validation = CausalValidation(
            hypothesis_id="drug_001",
            intervention_effect=0.35,
            causal_confidence=0.7,
            is_causally_supported=True,
            explanation="Test explanation",
        )

        assert validation.hypothesis_id == "drug_001"
        assert validation.intervention_effect == 0.35
        assert validation.is_causally_supported is True

    def test_serialization(self):
        """Test dictionary serialization."""
        from pipelines.therapeutic_hypothesis import CausalValidation

        validation = CausalValidation(
            hypothesis_id="drug_001",
            intervention_effect=0.35,
            causal_confidence=0.7,
            is_causally_supported=True,
        )

        data = validation.to_dict()
        assert data["hypothesis_id"] == "drug_001"
        assert data["intervention_effect"] == 0.35


class TestInterventionAnalysis:
    """Tests for InterventionAnalysis dataclass."""

    def test_creation(self):
        """Test creating intervention analysis result."""
        from pipelines.causal_analysis import InterventionAnalysis

        analysis = InterventionAnalysis(
            interventions={"pathway1": 0.0},
            outcome="asd_phenotype",
            effect=-0.25,
            explanation="Test explanation",
        )

        assert analysis.interventions == {"pathway1": 0.0}
        assert analysis.outcome == "asd_phenotype"
        assert analysis.effect == -0.25


class TestCounterfactualAnalysis:
    """Tests for CounterfactualAnalysis dataclass."""

    def test_creation(self):
        """Test creating counterfactual analysis result."""
        from pipelines.causal_analysis import CounterfactualAnalysis

        analysis = CounterfactualAnalysis(
            factual={"gene_function": 0.3},
            counterfactual={"gene_function": 1.0},
            outcome_variable="asd_phenotype",
            factual_outcome=0.8,
            counterfactual_outcome=0.3,
            change=-0.5,
        )

        assert analysis.factual_outcome == 0.8
        assert analysis.counterfactual_outcome == 0.3
        assert analysis.change == -0.5


class TestPathwayMediationAnalysis:
    """Tests for PathwayMediationAnalysis dataclass."""

    def test_creation(self):
        """Test creating pathway mediation analysis result."""
        from pipelines.causal_analysis import PathwayMediationAnalysis

        analysis = PathwayMediationAnalysis(
            gene="SHANK3",
            pathway="synaptic_transmission",
            phenotype="asd_phenotype",
            total_effect=0.6,
            direct_effect=0.2,
            indirect_effect=0.4,
            proportion_mediated=0.67,
        )

        assert analysis.gene == "SHANK3"
        assert analysis.pathway == "synaptic_transmission"
        assert analysis.proportion_mediated == 0.67


# =============================================================================
# Test TherapeuticHypothesisResult
# =============================================================================

class TestTherapeuticHypothesisResult:
    """Tests for TherapeuticHypothesisResult dataclass."""

    def test_properties(self):
        """Test result properties."""
        from pipelines.therapeutic_hypothesis import (
            TherapeuticHypothesisResult,
            CausalValidation,
        )

        # Create mock objects
        mock_subtype_result = Mock()
        mock_subtype_result.n_subtypes = 3
        mock_subtype_result.pathway_scores = Mock()
        mock_subtype_result.pathway_scores.pathways = ["P1", "P2"]

        mock_ranking_result = Mock()
        mock_ranking_result.hypotheses = [Mock(), Mock()]
        mock_ranking_result.pathways_with_hypotheses = ["P1"]
        mock_ranking_result.high_evidence_count = 1
        mock_ranking_result.top_hypotheses = [Mock()]
        mock_ranking_result.top_hypotheses[0].summary.return_value = "Drug A: 0.85"

        causal_validations = [
            CausalValidation("d1", 0.3, None, 0.7, True),
            CausalValidation("d2", 0.1, None, 0.3, False),
        ]

        result = TherapeuticHypothesisResult(
            subtype_result=mock_subtype_result,
            individual_analyses={"S1": Mock(), "S2": Mock()},
            all_fired_rules=[Mock(), Mock(), Mock()],
            ranking_result=mock_ranking_result,
            causal_validations=causal_validations,
            pathway_hypotheses_map={"P1": [Mock()]},
            subtype_hypotheses_map={0: [Mock()], 1: []},
        )

        assert result.n_individuals == 2
        assert result.n_hypotheses == 2
        assert result.n_causally_validated == 1


# =============================================================================
# Test CausalAnalysisResult
# =============================================================================

class TestCausalAnalysisResult:
    """Tests for CausalAnalysisResult dataclass."""

    def test_creation(self):
        """Test creating causal analysis result."""
        from pipelines.causal_analysis import (
            CausalAnalysisResult,
            InterventionAnalysis,
            CounterfactualAnalysis,
            PathwayMediationAnalysis,
        )

        result = CausalAnalysisResult(
            sample_id="PATIENT_001",
            variant_genes=["SHANK3", "CHD8"],
            disrupted_pathways=["synaptic_transmission"],
            causal_model_nodes=["node1", "node2", "node3"],
            causal_model_edges=5,
            intervention_analyses=[
                InterventionAnalysis({"p1": 0.0}, "outcome", -0.2)
            ],
            counterfactual_analyses=[],
            mediation_analyses=[],
            key_causal_drivers=["SHANK3 (effect: 0.5)"],
            intervention_opportunities=[{"target": "synaptic_transmission"}],
        )

        assert result.sample_id == "PATIENT_001"
        assert result.n_interventions == 1
        assert result.n_counterfactuals == 0
        assert len(result.key_causal_drivers) == 1

    def test_summary(self):
        """Test summary generation."""
        from pipelines.causal_analysis import CausalAnalysisResult

        result = CausalAnalysisResult(
            sample_id="PATIENT_001",
            variant_genes=["SHANK3"],
            disrupted_pathways=["synaptic_transmission"],
            causal_model_nodes=["n1", "n2"],
            causal_model_edges=3,
            intervention_analyses=[],
            counterfactual_analyses=[],
            mediation_analyses=[],
            key_causal_drivers=[],
            intervention_opportunities=[],
        )

        summary = result.summary
        assert "CAUSAL ANALYSIS PIPELINE RESULTS" in summary
        assert "PATIENT_001" in summary
        assert "SHANK3" in summary


# =============================================================================
# Test Pipeline Initialization
# =============================================================================

class TestTherapeuticHypothesisPipeline:
    """Tests for TherapeuticHypothesisPipeline initialization."""

    def test_initialization(self):
        """Test pipeline initialization."""
        from pipelines.therapeutic_hypothesis import (
            TherapeuticHypothesisPipeline,
            TherapeuticPipelineConfig,
            TherapeuticConfig,
        )
        from pipelines.subtype_discovery import DataConfig

        config = TherapeuticPipelineConfig(
            data=DataConfig(vcf_path="test.vcf", pathway_gmt_path="test.gmt"),
            therapeutic=TherapeuticConfig(enable_rules=True),
            verbose=False,
        )

        pipeline = TherapeuticHypothesisPipeline(config)
        assert pipeline.config == config
        assert pipeline._subtype_pipeline is None  # Lazy loading


class TestCausalAnalysisPipeline:
    """Tests for CausalAnalysisPipeline initialization."""

    def test_initialization(self):
        """Test pipeline initialization."""
        from pipelines.causal_analysis import (
            CausalAnalysisPipeline,
            CausalAnalysisConfig,
        )

        config = CausalAnalysisConfig(
            sample_id="TEST_001",
            variant_genes=["SHANK3"],
            disrupted_pathways=["synaptic_transmission"],
            verbose=False,
        )

        pipeline = CausalAnalysisPipeline(config)
        assert pipeline.config == config
        assert pipeline._scm is None  # Lazy loading


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_run_therapeutic_hypothesis_signature(self):
        """Test that convenience function has correct signature."""
        from pipelines.therapeutic_hypothesis import run_therapeutic_hypothesis
        import inspect

        sig = inspect.signature(run_therapeutic_hypothesis)
        params = list(sig.parameters.keys())

        assert "vcf_path" in params
        assert "pathway_path" in params
        assert "enable_rules" in params
        assert "enable_causal" in params

    def test_run_causal_analysis_signature(self):
        """Test that convenience function has correct signature."""
        from pipelines.causal_analysis import run_causal_analysis
        import inspect

        sig = inspect.signature(run_causal_analysis)
        params = list(sig.parameters.keys())

        assert "sample_id" in params
        assert "variant_genes" in params
        assert "disrupted_pathways" in params


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
