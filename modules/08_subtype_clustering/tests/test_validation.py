"""
Tests for Research Integrity Validation Components

Tests for ConfoundAnalyzer, NegativeControlRunner, and ProvenanceRecord.
"""

import sys
from pathlib import Path
from datetime import datetime
import tempfile
import json

import numpy as np
import pytest

# Add module directory to path
_module_dir = Path(__file__).parent.parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from validation import (
    ConfoundType,
    ConfoundTestResult,
    ConfoundReport,
    ConfoundAnalyzerConfig,
    ConfoundAnalyzer,
    PermutationResult,
    NegativeControlReport,
    NegativeControlConfig,
    NegativeControlRunner,
    ProvenanceRecord,
)
from clustering import ClusteringConfig, ClusteringMethod, SubtypeClusterer


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Generate sample clustering data."""
    np.random.seed(42)
    n_samples = 60
    n_features = 10

    # Create data with 3 clear clusters
    cluster_centers = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 3, 3, 3, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 3, 3, 3, 3],
    ])

    data = []
    labels = []
    for i, center in enumerate(cluster_centers):
        cluster_data = np.random.randn(20, n_features) * 0.5 + center
        data.append(cluster_data)
        labels.extend([i] * 20)

    return np.vstack(data), np.array(labels)


@pytest.fixture
def sample_confounds():
    """Generate sample confound variables."""
    np.random.seed(42)
    n_samples = 60

    return {
        "batch": np.repeat(["A", "B", "C"], 20),
        "site": np.tile(["Site1", "Site2"], 30),
        "age": np.random.uniform(20, 60, n_samples),
        "sex": np.random.choice(["M", "F"], n_samples),
    }


@pytest.fixture
def clusterer():
    """Create a simple clusterer."""
    config = ClusteringConfig(
        method=ClusteringMethod.KMEANS,
        n_clusters=3,
        random_state=42,
    )
    return SubtypeClusterer(config)


# =============================================================================
# Confound Analyzer Tests
# =============================================================================

class TestConfoundAnalyzerConfig:
    """Tests for ConfoundAnalyzerConfig."""

    def test_default_config(self):
        config = ConfoundAnalyzerConfig()
        assert config.significance_threshold == 0.05
        assert config.effect_size_threshold == 0.3
        assert config.apply_bonferroni is True
        assert config.min_samples_per_group == 5

    def test_custom_config(self):
        config = ConfoundAnalyzerConfig(
            significance_threshold=0.01,
            effect_size_threshold=0.5,
            apply_bonferroni=False,
        )
        assert config.significance_threshold == 0.01
        assert config.effect_size_threshold == 0.5
        assert config.apply_bonferroni is False


class TestConfoundTestResult:
    """Tests for ConfoundTestResult dataclass."""

    def test_result_creation(self):
        result = ConfoundTestResult(
            confound_name="batch",
            confound_type=ConfoundType.BATCH,
            test_statistic=10.5,
            p_value=0.001,
            effect_size=0.4,
            test_method="chi-squared",
            is_significant=True,
            interpretation="Significant batch effect detected.",
        )
        assert result.confound_name == "batch"
        assert result.confound_type == ConfoundType.BATCH
        assert result.is_significant is True


class TestConfoundReport:
    """Tests for ConfoundReport dataclass."""

    def test_has_significant_confounds(self):
        result = ConfoundTestResult(
            confound_name="batch",
            confound_type=ConfoundType.BATCH,
            test_statistic=10.5,
            p_value=0.001,
            effect_size=0.4,
            test_method="chi-squared",
            is_significant=True,
            interpretation="Significant.",
        )
        report = ConfoundReport(
            test_results=[result],
            overall_risk="moderate",
            problematic_confounds=["batch"],
            recommendations=["Apply batch correction."],
        )
        assert report.has_significant_confounds is True

    def test_no_significant_confounds(self):
        result = ConfoundTestResult(
            confound_name="batch",
            confound_type=ConfoundType.BATCH,
            test_statistic=1.5,
            p_value=0.5,
            effect_size=0.1,
            test_method="chi-squared",
            is_significant=False,
            interpretation="Not significant.",
        )
        report = ConfoundReport(
            test_results=[result],
            overall_risk="low",
            problematic_confounds=[],
            recommendations=["All clear."],
        )
        assert report.has_significant_confounds is False

    def test_get_summary(self):
        report = ConfoundReport(
            test_results=[],
            overall_risk="low",
            problematic_confounds=[],
            recommendations=["No issues detected."],
        )
        summary = report.get_summary()
        assert "Confound Analysis Report" in summary
        assert "LOW" in summary


class TestConfoundAnalyzer:
    """Tests for ConfoundAnalyzer class."""

    def test_init_default_config(self):
        analyzer = ConfoundAnalyzer()
        assert analyzer.config.significance_threshold == 0.05

    def test_init_custom_config(self):
        config = ConfoundAnalyzerConfig(significance_threshold=0.01)
        analyzer = ConfoundAnalyzer(config)
        assert analyzer.config.significance_threshold == 0.01

    def test_test_cluster_confound_alignment_categorical(self, sample_data, sample_confounds):
        data, labels = sample_data
        analyzer = ConfoundAnalyzer()

        report = analyzer.test_cluster_confound_alignment(
            cluster_labels=labels,
            confounds={"batch": sample_confounds["batch"]},
        )

        assert isinstance(report, ConfoundReport)
        assert len(report.test_results) == 1
        assert report.test_results[0].confound_name == "batch"

    def test_test_cluster_confound_alignment_continuous(self, sample_data, sample_confounds):
        data, labels = sample_data
        analyzer = ConfoundAnalyzer()

        report = analyzer.test_cluster_confound_alignment(
            cluster_labels=labels,
            confounds={"age": sample_confounds["age"]},
        )

        assert isinstance(report, ConfoundReport)
        assert len(report.test_results) == 1

    def test_test_cluster_confound_alignment_multiple(self, sample_data, sample_confounds):
        data, labels = sample_data
        analyzer = ConfoundAnalyzer()

        report = analyzer.test_cluster_confound_alignment(
            cluster_labels=labels,
            confounds=sample_confounds,
        )

        assert len(report.test_results) == 4
        assert report.overall_risk in ["low", "moderate", "high"]

    def test_compute_confound_association_categorical(self, sample_data, sample_confounds):
        data, labels = sample_data
        analyzer = ConfoundAnalyzer()

        stat, p_value, effect = analyzer.compute_confound_association(
            cluster_labels=labels,
            confound_values=sample_confounds["batch"],
            is_categorical=True,
        )

        assert stat >= 0
        assert 0 <= p_value <= 1
        assert 0 <= effect <= 1

    def test_compute_confound_association_continuous(self, sample_data, sample_confounds):
        data, labels = sample_data
        analyzer = ConfoundAnalyzer()

        stat, p_value, effect = analyzer.compute_confound_association(
            cluster_labels=labels,
            confound_values=sample_confounds["age"],
            is_categorical=False,
        )

        assert stat >= 0
        assert 0 <= p_value <= 1

    def test_confound_type_inference(self):
        analyzer = ConfoundAnalyzer()

        # Test type inference
        batch_type = analyzer._infer_confound_type("batch_id", np.array(["A", "B"]))
        assert batch_type == ConfoundType.BATCH

        site_type = analyzer._infer_confound_type("site_name", np.array([1, 2]))
        assert site_type == ConfoundType.SITE

        ancestry_type = analyzer._infer_confound_type("ancestry_pcs", np.array([0.1, 0.2]))
        assert ancestry_type == ConfoundType.ANCESTRY


# =============================================================================
# Negative Control Runner Tests
# =============================================================================

class TestNegativeControlConfig:
    """Tests for NegativeControlConfig."""

    def test_default_config(self):
        config = NegativeControlConfig()
        assert config.n_permutations == 1000
        assert config.significance_threshold == 0.05
        assert config.random_state == 42

    def test_custom_config(self):
        config = NegativeControlConfig(
            n_permutations=500,
            significance_threshold=0.01,
        )
        assert config.n_permutations == 500


class TestPermutationResult:
    """Tests for PermutationResult dataclass."""

    def test_result_creation(self):
        result = PermutationResult(
            observed_metric=0.8,
            null_distribution=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            p_value=0.01,
            metric_name="silhouette_score",
            n_permutations=100,
            is_significant=True,
        )
        assert result.observed_metric == 0.8
        assert result.is_significant is True

    def test_empirical_p_value(self):
        null_dist = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        result = PermutationResult(
            observed_metric=0.85,
            null_distribution=null_dist,
            p_value=0.1,
            metric_name="silhouette_score",
            n_permutations=9,
            is_significant=True,
        )
        # Only 0.9 >= 0.85, so (1 + 1) / (9 + 1) = 0.2
        assert result.empirical_p_value == 0.2

    def test_z_score(self):
        null_dist = np.array([0.0, 0.2, 0.4, 0.6, 0.8])  # mean=0.4, std≈0.28
        result = PermutationResult(
            observed_metric=1.0,
            null_distribution=null_dist,
            p_value=0.01,
            metric_name="silhouette_score",
            n_permutations=5,
            is_significant=True,
        )
        # z = (1.0 - 0.4) / std ≈ 2.1
        assert result.z_score > 2.0


class TestNegativeControlRunner:
    """Tests for NegativeControlRunner class."""

    def test_init_default_config(self):
        runner = NegativeControlRunner()
        assert runner.config.n_permutations == 1000

    def test_init_custom_config(self):
        config = NegativeControlConfig(n_permutations=50)
        runner = NegativeControlRunner(config)
        assert runner.config.n_permutations == 50

    def test_permutation_test(self, sample_data, clusterer):
        data, _ = sample_data
        config = NegativeControlConfig(n_permutations=10, random_state=42)
        runner = NegativeControlRunner(config)

        result = runner.permutation_test(data, clusterer)

        assert isinstance(result, PermutationResult)
        assert len(result.null_distribution) == 10
        assert 0 <= result.p_value <= 1

    def test_random_geneset_baseline(self, sample_data, clusterer):
        data, _ = sample_data
        config = NegativeControlConfig(random_state=42)
        runner = NegativeControlRunner(config)

        result = runner.random_geneset_baseline(
            data, clusterer, n_random_sets=10
        )

        assert "observed_metric" in result
        assert "random_mean" in result
        assert "p_value" in result
        assert "significantly_better" in result

    def test_label_shuffle_test(self, sample_data, clusterer):
        data, labels = sample_data
        config = NegativeControlConfig(n_permutations=10, random_state=42)
        runner = NegativeControlRunner(config)

        result = runner.label_shuffle_test(data, labels, clusterer)

        assert isinstance(result, PermutationResult)
        assert result.metric_name == "silhouette_score"

    def test_run_full_negative_control(self, sample_data, clusterer):
        data, _ = sample_data
        config = NegativeControlConfig(n_permutations=5, random_state=42)
        runner = NegativeControlRunner(config)

        report = runner.run_full_negative_control(data, clusterer)

        assert isinstance(report, NegativeControlReport)
        assert len(report.permutation_results) == 2
        assert "random_mean" in report.random_baseline_results
        assert len(report.recommendations) > 0


# =============================================================================
# Provenance Record Tests
# =============================================================================

class TestProvenanceRecord:
    """Tests for ProvenanceRecord dataclass."""

    def test_create_basic(self):
        record = ProvenanceRecord(
            reference_genome="GRCh38",
            annotation_version="GENCODE v38",
        )
        assert record.reference_genome == "GRCh38"
        assert record.annotation_version == "GENCODE v38"
        assert isinstance(record.timestamp, datetime)

    def test_create_with_databases(self):
        record = ProvenanceRecord(
            reference_genome="GRCh38",
            annotation_version="GENCODE v38",
            pathway_db_versions={
                "GO": "2023-01-01",
                "KEGG": "release95",
                "Reactome": "v84",
            },
        )
        assert len(record.pathway_db_versions) == 3
        assert record.pathway_db_versions["GO"] == "2023-01-01"

    def test_to_dict(self):
        record = ProvenanceRecord(
            reference_genome="GRCh38",
            annotation_version="GENCODE v38",
            cohort_name="Test Cohort",
        )
        d = record.to_dict()

        assert d["reference_genome"] == "GRCh38"
        assert d["cohort_name"] == "Test Cohort"
        assert "timestamp" in d

    def test_from_dict(self):
        data = {
            "reference_genome": "GRCh38",
            "annotation_version": "GENCODE v38",
            "pathway_db_versions": {"GO": "2023-01"},
            "timestamp": "2024-01-15T10:30:00",
        }
        record = ProvenanceRecord.from_dict(data)

        assert record.reference_genome == "GRCh38"
        assert record.pathway_db_versions["GO"] == "2023-01"

    def test_validate_compatibility_same(self):
        record1 = ProvenanceRecord(
            reference_genome="GRCh38",
            annotation_version="GENCODE v38",
        )
        record2 = ProvenanceRecord(
            reference_genome="GRCh38",
            annotation_version="GENCODE v38",
        )

        is_compatible, issues = record1.validate_compatibility(record2)
        assert is_compatible is True
        assert len(issues) == 0

    def test_validate_compatibility_different_genome(self):
        record1 = ProvenanceRecord(
            reference_genome="GRCh38",
            annotation_version="GENCODE v38",
        )
        record2 = ProvenanceRecord(
            reference_genome="GRCh37",
            annotation_version="GENCODE v38",
        )

        is_compatible, issues = record1.validate_compatibility(record2)
        assert is_compatible is False
        assert len(issues) > 0
        assert "Reference genome mismatch" in issues[0]

    def test_validate_compatibility_different_annotation(self):
        record1 = ProvenanceRecord(
            reference_genome="GRCh38",
            annotation_version="GENCODE v38",
        )
        record2 = ProvenanceRecord(
            reference_genome="GRCh38",
            annotation_version="GENCODE v36",
        )

        is_compatible, issues = record1.validate_compatibility(record2)
        assert is_compatible is False
        assert "Annotation version mismatch" in issues[0]

    def test_validate_compatibility_different_databases(self):
        record1 = ProvenanceRecord(
            reference_genome="GRCh38",
            annotation_version="GENCODE v38",
            pathway_db_versions={"GO": "2023-01"},
        )
        record2 = ProvenanceRecord(
            reference_genome="GRCh38",
            annotation_version="GENCODE v38",
            pathway_db_versions={"GO": "2024-01"},
        )

        is_compatible, issues = record1.validate_compatibility(record2)
        assert is_compatible is False
        assert "GO" in issues[0]

    def test_validate_compatibility_strict_mode(self):
        record1 = ProvenanceRecord(
            reference_genome="GRCh38",
            annotation_version="GENCODE v38",
            dependencies={"numpy": "1.24.0"},
        )
        record2 = ProvenanceRecord(
            reference_genome="GRCh38",
            annotation_version="GENCODE v38",
            dependencies={"numpy": "1.25.0"},
        )

        # Non-strict mode: should be compatible
        is_compatible, issues = record1.validate_compatibility(record2, strict=False)
        assert is_compatible is True

        # Strict mode: should fail
        is_compatible, issues = record1.validate_compatibility(record2, strict=True)
        assert is_compatible is False

    def test_get_summary(self):
        record = ProvenanceRecord(
            reference_genome="GRCh38",
            annotation_version="GENCODE v38",
            cohort_name="SPARK",
            pathway_db_versions={"GO": "2023-01"},
        )
        summary = record.get_summary()

        assert "Provenance Record" in summary
        assert "GRCh38" in summary
        assert "SPARK" in summary
        assert "GO" in summary

    def test_save_and_load(self):
        record = ProvenanceRecord(
            reference_genome="GRCh38",
            annotation_version="GENCODE v38",
            cohort_name="Test",
            pathway_db_versions={"GO": "2023-01"},
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            record.save(filepath)

            # Verify file exists and is valid JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
            assert data["reference_genome"] == "GRCh38"

            # Load and verify
            loaded = ProvenanceRecord.load(filepath)
            assert loaded.reference_genome == record.reference_genome
            assert loaded.cohort_name == record.cohort_name
        finally:
            Path(filepath).unlink()

    def test_create_current(self):
        record = ProvenanceRecord.create_current(
            reference_genome="GRCh38",
            annotation_version="GENCODE v38",
        )

        assert record.reference_genome == "GRCh38"
        assert "numpy" in record.dependencies
        assert "scipy" in record.dependencies
        assert "sklearn" in record.dependencies
        assert isinstance(record.timestamp, datetime)


# =============================================================================
# Integration Tests
# =============================================================================

class TestValidationIntegration:
    """Integration tests for validation components."""

    def test_full_validation_workflow(self, sample_data, sample_confounds, clusterer):
        """Test complete validation workflow."""
        data, labels = sample_data

        # 1. Run confound analysis
        confound_analyzer = ConfoundAnalyzer()
        confound_report = confound_analyzer.test_cluster_confound_alignment(
            cluster_labels=labels,
            confounds=sample_confounds,
        )
        assert isinstance(confound_report, ConfoundReport)

        # 2. Run negative controls
        nc_config = NegativeControlConfig(n_permutations=5, random_state=42)
        nc_runner = NegativeControlRunner(nc_config)
        nc_report = nc_runner.run_full_negative_control(data, clusterer)
        assert isinstance(nc_report, NegativeControlReport)

        # 3. Create provenance record
        provenance = ProvenanceRecord.create_current()
        provenance.cohort_name = "Test Cohort"
        provenance.notes = f"Confound risk: {confound_report.overall_risk}"

        assert "Test Cohort" in provenance.get_summary()

    def test_provenance_reproducibility(self):
        """Test that provenance enables reproducibility checking."""
        # Create two records with same versions
        record1 = ProvenanceRecord(
            reference_genome="GRCh38",
            annotation_version="GENCODE v38",
            pathway_db_versions={"GO": "2023-01", "KEGG": "v95"},
            pipeline_version="1.0.0",
        )

        record2 = ProvenanceRecord(
            reference_genome="GRCh38",
            annotation_version="GENCODE v38",
            pathway_db_versions={"GO": "2023-01", "KEGG": "v95"},
            pipeline_version="1.0.0",
        )

        is_compatible, issues = record1.validate_compatibility(record2)
        assert is_compatible is True

        # Change one version
        record2.pathway_db_versions["GO"] = "2024-01"
        is_compatible, issues = record1.validate_compatibility(record2)
        assert is_compatible is False
        assert any("GO" in issue for issue in issues)
