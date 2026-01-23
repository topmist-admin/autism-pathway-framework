"""
Unit tests for Module 08: Subtype Clustering.

Tests clustering algorithms, stability analysis, and subtype characterization.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add module to path
module_dir = Path(__file__).parent.parent
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

from clustering import (
    ClusteringMethod,
    ClusteringConfig,
    ClusteringResult,
    SubtypeClusterer,
)
from stability import (
    StabilityConfig,
    StabilityResult,
    StabilityAnalyzer,
)
from characterization import (
    CharacterizationConfig,
    PathwaySignature,
    SubtypeProfile,
    SubtypeCharacterizer,
)


# ============================================================================
# Mock Data Structures
# ============================================================================


class MockPathwayScoreMatrix:
    """Mock PathwayScoreMatrix for testing."""

    def __init__(self, scores: np.ndarray, samples: list, pathways: list):
        self.scores = scores
        self.samples = samples
        self.pathways = pathways


class MockClusteringResult:
    """Mock ClusteringResult for characterization testing."""

    def __init__(self, labels: np.ndarray, sample_ids: list):
        self.labels = labels
        self.sample_ids = sample_ids
        self.n_clusters = len(np.unique(labels))


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_data():
    """Generate synthetic clustered data."""
    np.random.seed(42)
    n_samples_per_cluster = 20
    n_features = 10

    # Create 3 distinct clusters
    cluster1 = np.random.randn(n_samples_per_cluster, n_features) + np.array([0, 0, 0, 5, 5, 0, 0, 0, 0, 0])
    cluster2 = np.random.randn(n_samples_per_cluster, n_features) + np.array([5, 5, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster3 = np.random.randn(n_samples_per_cluster, n_features) + np.array([0, 0, 5, 0, 0, 5, 5, 0, 0, 0])

    data = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.array([0] * n_samples_per_cluster + [1] * n_samples_per_cluster + [2] * n_samples_per_cluster)
    sample_ids = [f"sample_{i}" for i in range(len(data))]

    return data, sample_ids, true_labels


@pytest.fixture
def mock_pathway_scores(sample_data):
    """Create mock PathwayScoreMatrix."""
    data, sample_ids, _ = sample_data
    pathway_ids = [f"pathway_{i}" for i in range(data.shape[1])]
    return MockPathwayScoreMatrix(data, sample_ids, pathway_ids)


@pytest.fixture
def pathway_ids():
    """Generate pathway identifiers."""
    return [f"pathway_{i}" for i in range(10)]


@pytest.fixture
def pathway_names(pathway_ids):
    """Generate pathway names."""
    return {pid: f"Pathway {i}" for i, pid in enumerate(pathway_ids)}


# ============================================================================
# SubtypeClusterer Tests
# ============================================================================


class TestSubtypeClusterer:
    """Tests for SubtypeClusterer class."""

    def test_initialization(self):
        """Test clusterer initialization."""
        config = ClusteringConfig(n_clusters=3)
        clusterer = SubtypeClusterer(config)

        assert clusterer.config.n_clusters == 3
        assert clusterer.config.method == ClusteringMethod.GMM

    def test_gmm_clustering(self, mock_pathway_scores):
        """Test GMM clustering method."""
        config = ClusteringConfig(
            method=ClusteringMethod.GMM,
            n_clusters=3,
            random_state=42,
        )
        clusterer = SubtypeClusterer(config)
        result = clusterer.fit(mock_pathway_scores, n_clusters=3)

        assert isinstance(result, ClusteringResult)
        assert result.n_clusters == 3
        assert len(result.labels) == len(mock_pathway_scores.scores)
        assert result.probabilities is not None
        assert result.probabilities.shape == (len(mock_pathway_scores.scores), 3)

    def test_spectral_clustering(self, mock_pathway_scores):
        """Test spectral clustering method."""
        config = ClusteringConfig(
            method=ClusteringMethod.SPECTRAL,
            n_clusters=3,
            random_state=42,
        )
        clusterer = SubtypeClusterer(config)
        result = clusterer.fit(mock_pathway_scores, n_clusters=3)

        assert result.n_clusters == 3
        assert len(result.labels) == len(mock_pathway_scores.scores)

    def test_hierarchical_clustering(self, mock_pathway_scores):
        """Test hierarchical clustering method."""
        config = ClusteringConfig(
            method=ClusteringMethod.HIERARCHICAL,
            n_clusters=3,
            hierarchical_linkage="ward",
        )
        clusterer = SubtypeClusterer(config)
        result = clusterer.fit(mock_pathway_scores, n_clusters=3)

        assert result.n_clusters == 3
        assert len(result.labels) == len(mock_pathway_scores.scores)

    def test_kmeans_clustering(self, mock_pathway_scores):
        """Test k-means clustering method."""
        config = ClusteringConfig(
            method=ClusteringMethod.KMEANS,
            n_clusters=3,
            random_state=42,
        )
        clusterer = SubtypeClusterer(config)
        result = clusterer.fit(mock_pathway_scores, n_clusters=3)

        assert result.n_clusters == 3
        assert len(result.labels) == len(mock_pathway_scores.scores)

    def test_clustering_metrics(self, mock_pathway_scores):
        """Test that clustering computes quality metrics."""
        config = ClusteringConfig(n_clusters=3, random_state=42)
        clusterer = SubtypeClusterer(config)
        result = clusterer.fit(mock_pathway_scores, n_clusters=3)

        assert "silhouette" in result.metrics
        assert -1 <= result.metrics["silhouette"] <= 1
        assert "calinski_harabasz" in result.metrics
        assert result.metrics["calinski_harabasz"] > 0
        assert "davies_bouldin" in result.metrics
        assert result.metrics["davies_bouldin"] >= 0

    def test_get_cluster_samples(self, mock_pathway_scores):
        """Test retrieval of samples in specific cluster."""
        config = ClusteringConfig(n_clusters=3, random_state=42)
        clusterer = SubtypeClusterer(config)
        result = clusterer.fit(mock_pathway_scores, n_clusters=3)

        for cluster_id in range(3):
            samples = result.get_cluster_samples(cluster_id)
            assert len(samples) > 0

    def test_auto_select_n_clusters(self, mock_pathway_scores):
        """Test automatic cluster number selection."""
        config = ClusteringConfig(
            n_clusters=None,
            min_clusters=2,
            max_clusters=5,
            random_state=42,
        )
        clusterer = SubtypeClusterer(config)
        result = clusterer.fit(mock_pathway_scores)

        assert 2 <= result.n_clusters <= 5

    def test_fit_multiple_methods(self, mock_pathway_scores):
        """Test fitting multiple methods."""
        config = ClusteringConfig(n_clusters=3, random_state=42)
        clusterer = SubtypeClusterer(config)

        methods = [ClusteringMethod.GMM, ClusteringMethod.KMEANS]
        results = clusterer.fit_multiple_methods(mock_pathway_scores, methods, n_clusters=3)

        assert len(results) == 2
        assert "gmm" in results
        assert "kmeans" in results

    def test_consensus_clustering(self, mock_pathway_scores):
        """Test consensus clustering."""
        config = ClusteringConfig(n_clusters=3, random_state=42)
        clusterer = SubtypeClusterer(config)

        result = clusterer.get_consensus_clustering(
            mock_pathway_scores,
            n_runs=5,
            n_clusters=3,
        )

        assert result.n_clusters == 3
        assert len(result.labels) == len(mock_pathway_scores.scores)
        assert result.method == "consensus"

    def test_predict(self, mock_pathway_scores):
        """Test prediction on new data."""
        config = ClusteringConfig(
            method=ClusteringMethod.GMM,
            n_clusters=3,
            random_state=42,
        )
        clusterer = SubtypeClusterer(config)
        clusterer.fit(mock_pathway_scores, n_clusters=3)

        # Predict on same data
        predictions = clusterer.predict(mock_pathway_scores)
        assert len(predictions) == len(mock_pathway_scores.scores)

    def test_predict_proba(self, mock_pathway_scores):
        """Test probability prediction."""
        config = ClusteringConfig(
            method=ClusteringMethod.GMM,
            n_clusters=3,
            random_state=42,
        )
        clusterer = SubtypeClusterer(config)
        clusterer.fit(mock_pathway_scores, n_clusters=3)

        proba = clusterer.predict_proba(mock_pathway_scores)
        assert proba.shape == (len(mock_pathway_scores.scores), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)


# ============================================================================
# StabilityAnalyzer Tests
# ============================================================================


class TestStabilityAnalyzer:
    """Tests for StabilityAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        config = StabilityConfig(n_bootstrap=50)
        analyzer = StabilityAnalyzer(config)

        assert analyzer.config.n_bootstrap == 50
        assert analyzer.config.sample_fraction == 0.8

    def test_analyze_stability(self, sample_data):
        """Test stability analysis."""
        data, sample_ids, true_labels = sample_data

        clusterer_config = ClusteringConfig(n_clusters=3, random_state=42)
        clusterer = SubtypeClusterer(clusterer_config)

        stability_config = StabilityConfig(n_bootstrap=20, random_state=42)
        analyzer = StabilityAnalyzer(stability_config)

        result = analyzer.analyze_stability(data, clusterer, sample_ids)

        assert isinstance(result, StabilityResult)
        assert 0 <= result.mean_ari <= 1
        assert 0 <= result.mean_nmi <= 1
        assert result.n_bootstrap == 20
        assert len(result.sample_stability) == len(data)
        assert result.co_clustering_matrix.shape == (len(data), len(data))

    def test_stability_metrics(self, sample_data):
        """Test that stability metrics are computed correctly."""
        data, sample_ids, true_labels = sample_data

        clusterer_config = ClusteringConfig(n_clusters=3, random_state=42)
        clusterer = SubtypeClusterer(clusterer_config)

        stability_config = StabilityConfig(n_bootstrap=20, random_state=42)
        analyzer = StabilityAnalyzer(stability_config)

        result = analyzer.analyze_stability(data, clusterer, sample_ids)

        assert result.ari_ci_low <= result.mean_ari <= result.ari_ci_high
        assert result.nmi_ci_low <= result.mean_nmi <= result.nmi_ci_high
        assert len(result.bootstrap_aris) <= 20
        assert len(result.bootstrap_nmis) <= 20

    def test_stability_rating(self, sample_data):
        """Test stability rating property."""
        data, sample_ids, true_labels = sample_data

        clusterer_config = ClusteringConfig(n_clusters=3, random_state=42)
        clusterer = SubtypeClusterer(clusterer_config)

        stability_config = StabilityConfig(n_bootstrap=20, random_state=42)
        analyzer = StabilityAnalyzer(stability_config)

        result = analyzer.analyze_stability(data, clusterer, sample_ids)

        assert result.stability_rating in ["excellent", "good", "moderate", "poor", "unstable"]

    def test_get_unstable_samples(self, sample_data):
        """Test identification of unstable samples."""
        data, sample_ids, true_labels = sample_data

        clusterer_config = ClusteringConfig(n_clusters=3, random_state=42)
        clusterer = SubtypeClusterer(clusterer_config)

        stability_config = StabilityConfig(n_bootstrap=20, random_state=42)
        analyzer = StabilityAnalyzer(stability_config)

        result = analyzer.analyze_stability(data, clusterer, sample_ids)
        unstable = result.get_unstable_samples(threshold=0.5)

        assert isinstance(unstable, list)
        assert all(isinstance(s, str) for s in unstable)

    def test_find_optimal_k_by_stability(self, sample_data):
        """Test finding optimal k using stability."""
        data, sample_ids, true_labels = sample_data

        stability_config = StabilityConfig(n_bootstrap=10, random_state=42)
        analyzer = StabilityAnalyzer(stability_config)

        result = analyzer.find_optimal_k(
            data,
            k_range=(2, 4),
            sample_ids=sample_ids,
        )

        assert "optimal_k" in result
        assert 2 <= result["optimal_k"] <= 4
        assert "stability_scores" in result
        assert len(result["stability_scores"]) == 3  # 2, 3, 4

    def test_compare_methods_stability(self, sample_data):
        """Test comparing methods by stability."""
        data, sample_ids, true_labels = sample_data

        stability_config = StabilityConfig(n_bootstrap=10, random_state=42)
        analyzer = StabilityAnalyzer(stability_config)

        results = analyzer.compare_methods(
            data,
            methods=[ClusteringMethod.GMM, ClusteringMethod.KMEANS],
            n_clusters=3,
            sample_ids=sample_ids,
        )

        assert len(results) == 2
        assert "gmm" in results
        assert "kmeans" in results
        assert all(isinstance(r, StabilityResult) for r in results.values())


# ============================================================================
# SubtypeCharacterizer Tests
# ============================================================================


class TestSubtypeCharacterizer:
    """Tests for SubtypeCharacterizer class."""

    def test_initialization(self):
        """Test characterizer initialization."""
        config = CharacterizationConfig(n_top_pathways=5)
        characterizer = SubtypeCharacterizer(config)

        assert characterizer.config.n_top_pathways == 5

    def test_characterize(self, sample_data, pathway_ids, pathway_names):
        """Test subtype characterization."""
        data, sample_ids, true_labels = sample_data

        # Create mock clustering result
        clustering_result = MockClusteringResult(true_labels, sample_ids)

        # Characterize subtypes
        characterizer = SubtypeCharacterizer()
        profiles = characterizer.characterize(
            clustering_result,
            data,
            pathway_ids,
            pathway_names,
        )

        assert len(profiles) == 3
        assert all(isinstance(p, SubtypeProfile) for p in profiles)

    def test_subtype_profile_properties(self, sample_data, pathway_ids, pathway_names):
        """Test SubtypeProfile properties."""
        data, sample_ids, true_labels = sample_data

        clustering_result = MockClusteringResult(true_labels, sample_ids)

        characterizer = SubtypeCharacterizer()
        profiles = characterizer.characterize(
            clustering_result,
            data,
            pathway_ids,
            pathway_names,
        )

        for profile in profiles:
            assert profile.n_samples > 0
            assert len(profile.sample_ids) == profile.n_samples
            assert len(profile.centroid) == len(pathway_ids)
            assert isinstance(profile.summary, str)

    def test_pathway_signatures(self, sample_data, pathway_ids, pathway_names):
        """Test pathway signature computation."""
        data, sample_ids, true_labels = sample_data

        clustering_result = MockClusteringResult(true_labels, sample_ids)

        config = CharacterizationConfig(
            n_top_pathways=5,
            p_value_threshold=0.1,  # Relaxed for test
            min_effect_size=0.3,  # Relaxed for test
        )
        characterizer = SubtypeCharacterizer(config)
        profiles = characterizer.characterize(
            clustering_result,
            data,
            pathway_ids,
            pathway_names,
        )

        # At least some profiles should have signatures
        all_signatures = []
        for profile in profiles:
            all_signatures.extend(profile.top_pathways)

        # Check signature properties if any exist
        for sig in all_signatures:
            assert isinstance(sig, PathwaySignature)
            assert sig.direction in ["up", "down"]
            assert sig.p_value >= 0
            assert sig.p_value <= 1

    def test_compare_subtypes(self, sample_data, pathway_ids, pathway_names):
        """Test subtype comparison."""
        data, sample_ids, true_labels = sample_data

        clustering_result = MockClusteringResult(true_labels, sample_ids)

        characterizer = SubtypeCharacterizer()
        profiles = characterizer.characterize(
            clustering_result,
            data,
            pathway_ids,
            pathway_names,
        )

        comparison = characterizer.compare_subtypes(profiles, pathway_ids)

        assert "comparison_matrix" in comparison
        assert "discriminating_pathways" in comparison
        assert "pairwise_distances" in comparison
        assert comparison["n_subtypes"] == 3

    def test_generate_report(self, sample_data, pathway_ids, pathway_names):
        """Test report generation."""
        data, sample_ids, true_labels = sample_data

        clustering_result = MockClusteringResult(true_labels, sample_ids)

        characterizer = SubtypeCharacterizer()
        profiles = characterizer.characterize(
            clustering_result,
            data,
            pathway_ids,
            pathway_names,
        )

        report = characterizer.generate_report(profiles)

        assert isinstance(report, str)
        assert "SUBTYPE 0" in report
        assert "SUBTYPE 1" in report
        assert "SUBTYPE 2" in report
        assert "Silhouette Score" in report


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the complete clustering pipeline."""

    def test_full_pipeline(self, mock_pathway_scores, pathway_ids, pathway_names):
        """Test complete subtype identification pipeline."""
        # Step 1: Cluster samples
        clusterer_config = ClusteringConfig(n_clusters=3, random_state=42)
        clusterer = SubtypeClusterer(clusterer_config)
        clustering_result = clusterer.fit(mock_pathway_scores, n_clusters=3)

        assert clustering_result.n_clusters == 3

        # Step 2: Assess stability
        stability_config = StabilityConfig(n_bootstrap=15, random_state=42)
        analyzer = StabilityAnalyzer(stability_config)

        # Create mock result for characterizer
        mock_result = MockClusteringResult(
            clustering_result.labels,
            mock_pathway_scores.samples
        )

        # Step 3: Characterize subtypes
        characterizer = SubtypeCharacterizer()
        profiles = characterizer.characterize(
            mock_result,
            mock_pathway_scores.scores,
            pathway_ids,
            pathway_names,
        )

        assert len(profiles) == 3

        # Step 4: Compare subtypes
        comparison = characterizer.compare_subtypes(profiles, pathway_ids)

        assert "discriminating_pathways" in comparison

        # Step 5: Generate report
        report = characterizer.generate_report(profiles)

        assert len(report) > 0

    def test_pipeline_with_auto_k(self, mock_pathway_scores):
        """Test pipeline with automatic k selection."""
        # Cluster with auto k selection
        config = ClusteringConfig(
            n_clusters=None,
            min_clusters=2,
            max_clusters=5,
            random_state=42,
        )
        clusterer = SubtypeClusterer(config)
        clustering_result = clusterer.fit(mock_pathway_scores)

        assert 2 <= clustering_result.n_clusters <= 5

    def test_handling_edge_cases(self):
        """Test handling of edge cases."""
        np.random.seed(42)

        # Small dataset
        small_data = np.random.randn(10, 5)
        small_ids = [f"s_{i}" for i in range(10)]
        small_pathways = [f"p_{i}" for i in range(5)]
        small_scores = MockPathwayScoreMatrix(small_data, small_ids, small_pathways)

        config = ClusteringConfig(n_clusters=2, random_state=42)
        clusterer = SubtypeClusterer(config)
        result = clusterer.fit(small_scores, n_clusters=2)

        assert result.n_clusters == 2
        assert len(result.labels) == 10

    def test_reproducibility(self, mock_pathway_scores):
        """Test that results are reproducible with same random state."""
        config = ClusteringConfig(
            method=ClusteringMethod.GMM,
            n_clusters=3,
            random_state=42,
        )

        clusterer1 = SubtypeClusterer(config)
        result1 = clusterer1.fit(mock_pathway_scores, n_clusters=3)

        clusterer2 = SubtypeClusterer(config)
        result2 = clusterer2.fit(mock_pathway_scores, n_clusters=3)

        # Labels should be identical
        assert np.array_equal(result1.labels, result2.labels)


# ============================================================================
# ClusteringResult Tests
# ============================================================================


class TestClusteringResult:
    """Tests for ClusteringResult dataclass."""

    def test_cluster_sizes(self, mock_pathway_scores):
        """Test cluster_sizes property."""
        config = ClusteringConfig(n_clusters=3, random_state=42)
        clusterer = SubtypeClusterer(config)
        result = clusterer.fit(mock_pathway_scores, n_clusters=3)

        sizes = result.cluster_sizes

        assert len(sizes) == 3
        assert sum(sizes.values()) == len(mock_pathway_scores.scores)

    def test_get_cluster_probability(self, mock_pathway_scores):
        """Test get_cluster_probability method."""
        config = ClusteringConfig(
            method=ClusteringMethod.GMM,
            n_clusters=3,
            random_state=42,
        )
        clusterer = SubtypeClusterer(config)
        result = clusterer.fit(mock_pathway_scores, n_clusters=3)

        probs = result.get_cluster_probability(0)
        assert len(probs) == 3
        assert abs(sum(probs.values()) - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
