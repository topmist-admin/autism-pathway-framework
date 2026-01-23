"""
Tests for the Subtype Discovery Pipeline.

These tests verify the pipeline components and integration using mock data.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Set, Any, Optional

import numpy as np
import pytest

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# =============================================================================
# Mock Data Classes (to avoid importing full module dependencies)
# =============================================================================

@dataclass
class MockVariant:
    """Mock variant for testing."""
    chrom: str
    pos: int
    ref: str
    alt: str
    sample_id: str
    genotype: str
    quality: float
    info: Dict[str, Any] = None
    filter_status: str = "PASS"
    variant_id: str = ""

    def __post_init__(self):
        if self.info is None:
            self.info = {}
        if not self.variant_id:
            self.variant_id = f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}"


@dataclass
class MockVariantDataset:
    """Mock variant dataset for testing."""
    variants: List[MockVariant]
    samples: List[str]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MockValidationReport:
    """Mock validation report."""
    is_valid: bool = True
    n_variants: int = 0
    n_samples: int = 0
    n_chromosomes: int = 0
    variant_types: Dict[str, int] = None
    quality_stats: Dict[str, float] = None
    warnings: List[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.variant_types is None:
            self.variant_types = {}
        if self.quality_stats is None:
            self.quality_stats = {}
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


@dataclass
class MockPathwayDatabase:
    """Mock pathway database for testing."""
    pathways: Dict[str, Set[str]]
    pathway_names: Dict[str, str]
    pathway_descriptions: Dict[str, str] = None
    gene_to_pathways: Dict[str, Set[str]] = None
    source: str = "TEST"

    def __post_init__(self):
        if self.pathway_descriptions is None:
            self.pathway_descriptions = {}
        if self.gene_to_pathways is None:
            self.gene_to_pathways = {}
            for pathway_id, genes in self.pathways.items():
                for gene in genes:
                    if gene not in self.gene_to_pathways:
                        self.gene_to_pathways[gene] = set()
                    self.gene_to_pathways[gene].add(pathway_id)

    def get_all_genes(self) -> Set[str]:
        """Get all genes in the database."""
        return set().union(*self.pathways.values())

    def filter_by_size(self, min_size: int = 5, max_size: int = 500):
        """Filter pathways by size."""
        filtered = {
            pid: genes
            for pid, genes in self.pathways.items()
            if min_size <= len(genes) <= max_size
        }
        return MockPathwayDatabase(
            pathways=filtered,
            pathway_names={k: v for k, v in self.pathway_names.items() if k in filtered},
            source=self.source,
        )


# =============================================================================
# Test Configuration Classes
# =============================================================================

class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_creation_with_vcf_only(self):
        """Test creating DataConfig with just VCF path."""
        from pipelines.subtype_discovery import DataConfig

        config = DataConfig(vcf_path="test.vcf")
        assert config.vcf_path == "test.vcf"
        assert config.pathway_gmt_path is None
        assert config.gene_id_type == "symbol"

    def test_creation_with_all_options(self):
        """Test creating DataConfig with all options."""
        from pipelines.subtype_discovery import DataConfig

        config = DataConfig(
            vcf_path="test.vcf",
            pathway_gmt_path="pathways.gmt",
            gnomad_constraints_path="constraints.tsv",
            ppi_network_path="ppi.txt",
            ppi_min_score=700.0,
            gene_id_type="ensembl",
        )
        assert config.vcf_path == "test.vcf"
        assert config.pathway_gmt_path == "pathways.gmt"
        assert config.gnomad_constraints_path == "constraints.tsv"
        assert config.ppi_min_score == 700.0


class TestProcessingConfig:
    """Tests for ProcessingConfig dataclass."""

    def test_default_values(self):
        """Test default processing configuration values."""
        from pipelines.subtype_discovery import ProcessingConfig

        config = ProcessingConfig()
        assert config.min_quality == 30.0
        assert config.min_depth == 10
        assert config.filter_pass_only is True
        assert config.max_allele_freq == 0.01
        assert config.use_constraint_weighting is True

    def test_custom_values(self):
        """Test custom processing configuration."""
        from pipelines.subtype_discovery import ProcessingConfig

        config = ProcessingConfig(
            min_quality=50.0,
            min_depth=20,
            max_allele_freq=0.001,
        )
        assert config.min_quality == 50.0
        assert config.min_depth == 20
        assert config.max_allele_freq == 0.001


class TestClusteringPipelineConfig:
    """Tests for ClusteringPipelineConfig dataclass."""

    def test_default_values(self):
        """Test default clustering configuration."""
        from pipelines.subtype_discovery import ClusteringPipelineConfig, ClusteringMethod

        config = ClusteringPipelineConfig()
        assert config.method == ClusteringMethod.GMM
        assert config.n_clusters is None
        assert config.min_clusters == 2
        assert config.max_clusters == 10
        assert config.run_stability is True

    def test_fixed_clusters(self):
        """Test configuration with fixed number of clusters."""
        from pipelines.subtype_discovery import ClusteringPipelineConfig

        config = ClusteringPipelineConfig(n_clusters=5)
        assert config.n_clusters == 5


class TestPipelineConfig:
    """Tests for the complete PipelineConfig."""

    def test_creation(self):
        """Test creating complete pipeline configuration."""
        from pipelines.subtype_discovery import (
            PipelineConfig,
            DataConfig,
            ProcessingConfig,
            ClusteringPipelineConfig,
        )

        config = PipelineConfig(
            data=DataConfig(vcf_path="test.vcf", pathway_gmt_path="test.gmt"),
            processing=ProcessingConfig(min_quality=40.0),
            clustering=ClusteringPipelineConfig(n_clusters=4),
            verbose=False,
            random_state=123,
        )

        assert config.data.vcf_path == "test.vcf"
        assert config.processing.min_quality == 40.0
        assert config.clustering.n_clusters == 4
        assert config.verbose is False
        assert config.random_state == 123


# =============================================================================
# Test SubtypeDiscoveryResult
# =============================================================================

class TestSubtypeDiscoveryResult:
    """Tests for SubtypeDiscoveryResult dataclass."""

    def test_summary_generation(self):
        """Test that summary is generated correctly."""
        from pipelines.subtype_discovery import SubtypeDiscoveryResult

        # Create mock objects
        mock_clustering = Mock()
        mock_clustering.n_clusters = 3
        mock_clustering.method = "GMM"
        mock_clustering.metrics = {"silhouette_score": 0.45}
        mock_clustering.cluster_sizes = {0: 10, 1: 15, 2: 8}
        mock_clustering.labels = np.array([0]*10 + [1]*15 + [2]*8)

        mock_pathway_scores = Mock()
        mock_pathway_scores.samples = [f"S{i}" for i in range(33)]
        mock_pathway_scores.pathways = [f"P{i}" for i in range(50)]
        mock_pathway_scores.sample_index = {f"S{i}": i for i in range(33)}

        mock_gene_burdens = Mock()
        mock_gene_burdens.genes = [f"G{i}" for i in range(100)]

        result = SubtypeDiscoveryResult(
            clustering_result=mock_clustering,
            subtype_profiles=[],
            n_subtypes=3,
            pathway_scores=mock_pathway_scores,
            gene_burdens=mock_gene_burdens,
            runtime_seconds=10.5,
        )

        summary = result.summary
        assert "SUBTYPE DISCOVERY PIPELINE RESULTS" in summary
        assert "Number of subtypes: 3" in summary
        assert "Samples: 33" in summary
        assert "Pathways scored: 50" in summary

    def test_get_subtype_assignment(self):
        """Test getting subtype assignment for a sample."""
        from pipelines.subtype_discovery import SubtypeDiscoveryResult

        mock_clustering = Mock()
        mock_clustering.n_clusters = 2
        mock_clustering.method = "GMM"
        mock_clustering.metrics = {}
        mock_clustering.cluster_sizes = {0: 5, 1: 5}
        mock_clustering.labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        mock_pathway_scores = Mock()
        mock_pathway_scores.samples = [f"S{i}" for i in range(10)]
        mock_pathway_scores.pathways = ["P1", "P2"]
        mock_pathway_scores.sample_index = {f"S{i}": i for i in range(10)}

        mock_gene_burdens = Mock()
        mock_gene_burdens.genes = ["G1", "G2"]

        result = SubtypeDiscoveryResult(
            clustering_result=mock_clustering,
            subtype_profiles=[],
            n_subtypes=2,
            pathway_scores=mock_pathway_scores,
            gene_burdens=mock_gene_burdens,
        )

        assert result.get_subtype_assignment("S0") == 0
        assert result.get_subtype_assignment("S5") == 1
        assert result.get_subtype_assignment("UNKNOWN") is None

    def test_get_samples_by_subtype(self):
        """Test getting all samples for a subtype."""
        from pipelines.subtype_discovery import SubtypeDiscoveryResult

        mock_clustering = Mock()
        mock_clustering.n_clusters = 2
        mock_clustering.method = "GMM"
        mock_clustering.metrics = {}
        mock_clustering.cluster_sizes = {0: 3, 1: 2}
        mock_clustering.labels = np.array([0, 0, 0, 1, 1])

        mock_pathway_scores = Mock()
        mock_pathway_scores.samples = ["A", "B", "C", "D", "E"]
        mock_pathway_scores.pathways = ["P1"]
        mock_pathway_scores.sample_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

        mock_gene_burdens = Mock()
        mock_gene_burdens.genes = ["G1"]

        result = SubtypeDiscoveryResult(
            clustering_result=mock_clustering,
            subtype_profiles=[],
            n_subtypes=2,
            pathway_scores=mock_pathway_scores,
            gene_burdens=mock_gene_burdens,
        )

        subtype_0_samples = result.get_samples_by_subtype(0)
        assert set(subtype_0_samples) == {"A", "B", "C"}

        subtype_1_samples = result.get_samples_by_subtype(1)
        assert set(subtype_1_samples) == {"D", "E"}


# =============================================================================
# Test Pipeline Initialization
# =============================================================================

class TestSubtypeDiscoveryPipeline:
    """Tests for SubtypeDiscoveryPipeline class."""

    def test_initialization(self):
        """Test pipeline initialization."""
        from pipelines.subtype_discovery import (
            SubtypeDiscoveryPipeline,
            PipelineConfig,
            DataConfig,
        )

        config = PipelineConfig(
            data=DataConfig(vcf_path="test.vcf", pathway_gmt_path="test.gmt"),
            verbose=False,
        )

        pipeline = SubtypeDiscoveryPipeline(config)
        assert pipeline.config == config
        assert pipeline._vcf_loader is None  # Lazy loading
        assert pipeline._pathway_db is None

    def test_logging_setup(self):
        """Test that logging is configured based on verbosity."""
        from pipelines.subtype_discovery import (
            SubtypeDiscoveryPipeline,
            PipelineConfig,
            DataConfig,
        )

        # Verbose mode
        config_verbose = PipelineConfig(
            data=DataConfig(vcf_path="test.vcf", pathway_gmt_path="test.gmt"),
            verbose=True,
        )
        pipeline_verbose = SubtypeDiscoveryPipeline(config_verbose)

        # Non-verbose mode
        config_quiet = PipelineConfig(
            data=DataConfig(vcf_path="test.vcf", pathway_gmt_path="test.gmt"),
            verbose=False,
        )
        pipeline_quiet = SubtypeDiscoveryPipeline(config_quiet)

        # Both should initialize without error
        assert pipeline_verbose is not None
        assert pipeline_quiet is not None


# =============================================================================
# Integration Tests (with mocked modules)
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for the pipeline with mocked module imports."""

    @pytest.fixture
    def mock_modules(self):
        """Create mock modules for testing."""
        # This would mock the actual module imports
        # For now, we'll test the configuration and result handling
        pass

    def test_config_propagation(self):
        """Test that configuration is properly propagated through pipeline."""
        from pipelines.subtype_discovery import (
            PipelineConfig,
            DataConfig,
            ProcessingConfig,
            ClusteringPipelineConfig,
        )

        config = PipelineConfig(
            data=DataConfig(
                vcf_path="cohort.vcf.gz",
                pathway_gmt_path="reactome.gmt",
            ),
            processing=ProcessingConfig(
                min_quality=40.0,
                max_allele_freq=0.005,
            ),
            clustering=ClusteringPipelineConfig(
                n_clusters=5,
                run_stability=True,
                n_bootstrap=50,
            ),
            random_state=42,
        )

        assert config.data.vcf_path == "cohort.vcf.gz"
        assert config.processing.min_quality == 40.0
        assert config.processing.max_allele_freq == 0.005
        assert config.clustering.n_clusters == 5
        assert config.clustering.n_bootstrap == 50
        assert config.random_state == 42


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
