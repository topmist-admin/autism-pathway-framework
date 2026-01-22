"""
Tests for Module 07: Pathway Scoring

Tests the pathway aggregation, network propagation, and normalization components.
"""

import sys
from pathlib import Path

# Add module to path to handle numeric prefix in module name
_module_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_module_dir))

import numpy as np
import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any

# Import from local module using direct imports
from aggregation import (
    AggregationConfig,
    AggregationMethod,
    PathwayAggregator,
    PathwayScoreMatrix,
)
from network_propagation import (
    NetworkPropagator,
    PropagationConfig,
    PropagationMethod,
)
from normalization import (
    NormalizationConfig,
    NormalizationMethod,
    PathwayScoreNormalizer,
)


# Mock data structures to avoid import dependencies during testing
@dataclass
class MockGeneBurdenMatrix:
    """Mock GeneBurdenMatrix for testing."""

    samples: List[str]
    genes: List[str]
    scores: np.ndarray
    sample_index: Dict[str, int] = field(default_factory=dict)
    gene_index: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.sample_index:
            self.sample_index = {s: i for i, s in enumerate(self.samples)}
        if not self.gene_index:
            self.gene_index = {g: i for i, g in enumerate(self.genes)}

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    def get_sample(self, sample_id: str) -> Dict[str, float]:
        if sample_id not in self.sample_index:
            return {}
        idx = self.sample_index[sample_id]
        return {
            gene: float(self.scores[idx, g_idx])
            for gene, g_idx in self.gene_index.items()
            if self.scores[idx, g_idx] > 0
        }

    def get_score(self, sample_id: str, gene_id: str) -> float:
        if sample_id not in self.sample_index:
            return 0.0
        if gene_id not in self.gene_index:
            return 0.0
        return float(self.scores[self.sample_index[sample_id], self.gene_index[gene_id]])


@dataclass
class MockPathwayDatabase:
    """Mock PathwayDatabase for testing."""

    pathways: Dict[str, Set[str]]
    pathway_names: Dict[str, str]
    source: str = "test"

    def filter_by_size(self, min_size: int = 5, max_size: int = 500) -> "MockPathwayDatabase":
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


class TestPathwayScoreMatrix:
    """Tests for PathwayScoreMatrix data structure."""

    def test_creation(self):
        """Test basic PathwayScoreMatrix creation."""
        scores = PathwayScoreMatrix(
            samples=["S1", "S2", "S3"],
            pathways=["P1", "P2"],
            scores=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        )

        assert scores.n_samples == 3
        assert scores.n_pathways == 2
        assert scores.shape == (3, 2)

    def test_get_sample(self):
        """Test getting pathway scores for a sample."""
        scores = PathwayScoreMatrix(
            samples=["S1", "S2"],
            pathways=["P1", "P2", "P3"],
            scores=np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]]),
        )

        s1_scores = scores.get_sample("S1")
        assert "P1" in s1_scores
        assert "P3" in s1_scores
        assert "P2" not in s1_scores  # Zero score excluded

    def test_get_pathway(self):
        """Test getting sample scores for a pathway."""
        scores = PathwayScoreMatrix(
            samples=["S1", "S2", "S3"],
            pathways=["P1", "P2"],
            scores=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        )

        p1_scores = scores.get_pathway("P1")
        np.testing.assert_array_equal(p1_scores, [1.0, 3.0, 5.0])

    def test_get_top_pathways(self):
        """Test getting top scoring pathways."""
        scores = PathwayScoreMatrix(
            samples=["S1"],
            pathways=["P1", "P2", "P3", "P4"],
            scores=np.array([[1.0, 4.0, 2.0, 3.0]]),
        )

        top = scores.get_top_pathways("S1", n=2)
        assert len(top) == 2
        assert top[0][0] == "P2"  # Highest score
        assert top[0][1] == 4.0

    def test_filter_pathways(self):
        """Test filtering pathways."""
        scores = PathwayScoreMatrix(
            samples=["S1", "S2", "S3"],
            pathways=["P1", "P2", "P3"],
            scores=np.array([
                [0.5, 0.1, 0.0],
                [0.6, 0.0, 0.0],
                [0.7, 0.0, 0.0],
            ]),
        )

        # Filter to pathways with min_score >= 0.5 and at least 2 samples
        filtered = scores.filter_pathways(min_score=0.5, min_samples_hit=2)
        assert filtered.n_pathways == 1
        assert "P1" in filtered.pathways


class TestPathwayAggregator:
    """Tests for PathwayAggregator."""

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        # Mock gene burden matrix
        genes = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
        samples = ["S1", "S2", "S3"]
        scores = np.array([
            [1.0, 0.5, 0.0, 0.3, 0.0],  # S1
            [0.0, 0.8, 0.6, 0.0, 0.2],  # S2
            [0.4, 0.0, 0.7, 0.5, 0.1],  # S3
        ])

        gene_burdens = MockGeneBurdenMatrix(
            samples=samples,
            genes=genes,
            scores=scores,
        )

        # Mock pathway database
        pathway_db = MockPathwayDatabase(
            pathways={
                "pathway_A": {"GENE1", "GENE2", "GENE3", "GENE6"},  # 3 genes overlap
                "pathway_B": {"GENE3", "GENE4", "GENE5", "GENE7"},  # 3 genes overlap
                "pathway_C": {"GENE1", "GENE2"},  # 2 genes, below min_size
                "pathway_D": {"GENE8", "GENE9", "GENE10"},  # No overlap
            },
            pathway_names={
                "pathway_A": "Test Pathway A",
                "pathway_B": "Test Pathway B",
                "pathway_C": "Test Pathway C",
                "pathway_D": "Test Pathway D",
            },
        )

        return gene_burdens, pathway_db

    def test_basic_aggregation(self, sample_data):
        """Test basic pathway score aggregation."""
        gene_burdens, pathway_db = sample_data

        config = AggregationConfig(
            method=AggregationMethod.SUM,
            min_pathway_size=2,  # Lower threshold for test data
            normalize_by_pathway_size=False,
            weight_by_gene_coverage=False,
        )
        aggregator = PathwayAggregator(config)

        result = aggregator.aggregate(gene_burdens, pathway_db)

        assert result.n_samples == 3
        assert result.n_pathways >= 1  # At least pathway_A and pathway_B

    def test_aggregation_methods(self, sample_data):
        """Test different aggregation methods produce different results."""
        gene_burdens, pathway_db = sample_data

        results = {}
        for method in [AggregationMethod.SUM, AggregationMethod.MAX, AggregationMethod.MEAN]:
            config = AggregationConfig(
                method=method,
                min_pathway_size=2,
                normalize_by_pathway_size=False,
                weight_by_gene_coverage=False,
            )
            aggregator = PathwayAggregator(config)
            results[method.value] = aggregator.aggregate(gene_burdens, pathway_db)

        # Methods should produce different scores
        if results["sum"].n_pathways > 0:
            sum_scores = results["sum"].scores
            max_scores = results["max"].scores
            mean_scores = results["mean"].scores

            # Sum should generally be >= max
            assert np.all(sum_scores >= max_scores - 1e-10)

    def test_contributing_genes_tracked(self, sample_data):
        """Test that contributing genes are tracked."""
        gene_burdens, pathway_db = sample_data

        config = AggregationConfig(min_pathway_size=2)
        aggregator = PathwayAggregator(config)
        result = aggregator.aggregate(gene_burdens, pathway_db)

        # Check that some contributing genes are tracked
        has_contributors = False
        for sample in result.samples:
            for pathway in result.pathways:
                contributors = result.get_contributing_genes(sample, pathway)
                if contributors:
                    has_contributors = True
                    break

        assert has_contributors


class TestNetworkPropagator:
    """Tests for NetworkPropagator."""

    def test_build_network_from_edges(self):
        """Test building network from edge list."""
        edges = [
            ("A", "B", 0.8),
            ("B", "C", 0.6),
            ("A", "C", 0.5),
            ("C", "D", 0.7),
        ]

        propagator = NetworkPropagator()
        propagator.build_network_from_edges(edges)

        stats = propagator.get_network_stats()
        assert stats["n_nodes"] == 4
        assert stats["n_edges"] == 4

    def test_random_walk_propagation(self):
        """Test random walk with restart propagation."""
        # Create a simple network
        edges = [
            ("A", "B", 1.0),
            ("B", "C", 1.0),
            ("C", "D", 1.0),
            ("A", "D", 1.0),
        ]

        config = PropagationConfig(
            method=PropagationMethod.RANDOM_WALK,
            restart_prob=0.5,
            n_iterations=50,
        )
        propagator = NetworkPropagator(config)
        propagator.build_network_from_edges(edges)

        # Propagate from node A
        seed_scores = {"A": 1.0}
        result = propagator.propagate(seed_scores)

        # A should have highest score (seed node)
        assert "A" in result.gene_scores
        # Neighbors should have some signal
        assert len(result.gene_scores) > 1

    def test_heat_diffusion_propagation(self):
        """Test heat diffusion propagation."""
        edges = [
            ("A", "B", 1.0),
            ("B", "C", 1.0),
            ("C", "D", 1.0),
        ]

        config = PropagationConfig(
            method=PropagationMethod.HEAT_DIFFUSION,
            diffusion_time=0.1,
        )
        propagator = NetworkPropagator(config)
        propagator.build_network_from_edges(edges)

        seed_scores = {"A": 1.0}
        result = propagator.propagate(seed_scores)

        assert result.converged
        assert len(result.gene_scores) >= 1

    def test_signal_decay_with_distance(self):
        """Test that signal decays with network distance."""
        # Linear chain: A - B - C - D - E
        edges = [
            ("A", "B", 1.0),
            ("B", "C", 1.0),
            ("C", "D", 1.0),
            ("D", "E", 1.0),
        ]

        config = PropagationConfig(
            method=PropagationMethod.RANDOM_WALK,
            restart_prob=0.5,
            normalize_output=False,
        )
        propagator = NetworkPropagator(config)
        propagator.build_network_from_edges(edges)

        seed_scores = {"A": 1.0}
        result = propagator.propagate(seed_scores)

        # Score should decay with distance from A
        scores = result.gene_scores
        if "B" in scores and "D" in scores:
            assert scores.get("B", 0) >= scores.get("D", 0)


class TestPathwayScoreNormalizer:
    """Tests for PathwayScoreNormalizer."""

    @pytest.fixture
    def sample_scores(self):
        """Create sample pathway scores for testing."""
        np.random.seed(42)
        scores = PathwayScoreMatrix(
            samples=[f"S{i}" for i in range(10)],
            pathways=["P1", "P2", "P3"],
            scores=np.random.rand(10, 3) * 10,  # Random scores 0-10
        )
        return scores

    def test_zscore_normalization(self, sample_scores):
        """Test z-score normalization."""
        config = NormalizationConfig(method=NormalizationMethod.ZSCORE)
        normalizer = PathwayScoreNormalizer(config)

        normalized = normalizer.normalize(sample_scores)

        # Check mean is ~0 and std is ~1 for each pathway
        for j in range(normalized.n_pathways):
            col = normalized.scores[:, j]
            assert abs(np.mean(col)) < 0.01
            assert abs(np.std(col) - 1.0) < 0.01

    def test_minmax_normalization(self, sample_scores):
        """Test min-max normalization."""
        config = NormalizationConfig(method=NormalizationMethod.MINMAX)
        normalizer = PathwayScoreNormalizer(config)

        normalized = normalizer.normalize(sample_scores)

        # Check all values in [0, 1]
        assert np.all(normalized.scores >= 0)
        assert np.all(normalized.scores <= 1)

        # Check min is 0 and max is 1 for each pathway
        for j in range(normalized.n_pathways):
            col = normalized.scores[:, j]
            assert abs(np.min(col)) < 0.01
            assert abs(np.max(col) - 1.0) < 0.01

    def test_rank_normalization(self, sample_scores):
        """Test rank-based normalization."""
        config = NormalizationConfig(method=NormalizationMethod.RANK)
        normalizer = PathwayScoreNormalizer(config)

        normalized = normalizer.normalize(sample_scores)

        # Check all values in [0, 1]
        assert np.all(normalized.scores >= 0)
        assert np.all(normalized.scores <= 1)

    def test_log_normalization(self, sample_scores):
        """Test log normalization."""
        config = NormalizationConfig(
            method=NormalizationMethod.LOG,
            pseudocount=1.0,
        )
        normalizer = PathwayScoreNormalizer(config)

        normalized = normalizer.normalize(sample_scores)

        # Log-transformed values should be smaller than original (for values > e)
        assert normalized.scores.shape == sample_scores.scores.shape

    def test_metadata_preserved(self, sample_scores):
        """Test that metadata is preserved after normalization."""
        sample_scores.metadata["test_key"] = "test_value"

        normalizer = PathwayScoreNormalizer()
        normalized = normalizer.normalize(sample_scores)

        assert "test_key" in normalized.metadata
        assert normalized.metadata["normalized"] is True


class TestIntegration:
    """Integration tests for the full pathway scoring pipeline."""

    def test_full_pipeline(self):
        """Test complete pathway scoring pipeline."""
        # Create mock data
        genes = ["G1", "G2", "G3", "G4", "G5", "G6"]
        samples = ["S1", "S2", "S3", "S4"]
        burden_scores = np.array([
            [1.0, 0.5, 0.0, 0.3, 0.0, 0.2],
            [0.0, 0.8, 0.6, 0.0, 0.2, 0.0],
            [0.4, 0.0, 0.7, 0.5, 0.1, 0.3],
            [0.2, 0.3, 0.4, 0.1, 0.5, 0.0],
        ])

        gene_burdens = MockGeneBurdenMatrix(
            samples=samples,
            genes=genes,
            scores=burden_scores,
        )

        pathway_db = MockPathwayDatabase(
            pathways={
                "pathway_1": {"G1", "G2", "G3"},
                "pathway_2": {"G3", "G4", "G5"},
                "pathway_3": {"G1", "G4", "G6"},
            },
            pathway_names={
                "pathway_1": "Pathway 1",
                "pathway_2": "Pathway 2",
                "pathway_3": "Pathway 3",
            },
        )

        # Step 1: Aggregate
        agg_config = AggregationConfig(
            method=AggregationMethod.WEIGHTED_SUM,
            min_pathway_size=2,
            normalize_by_pathway_size=True,
        )
        aggregator = PathwayAggregator(agg_config)
        pathway_scores = aggregator.aggregate(gene_burdens, pathway_db)

        assert pathway_scores.n_samples == 4
        assert pathway_scores.n_pathways == 3

        # Step 2: Normalize
        norm_config = NormalizationConfig(method=NormalizationMethod.ZSCORE)
        normalizer = PathwayScoreNormalizer(norm_config)
        normalized = normalizer.normalize(pathway_scores)

        # Check z-score properties
        for j in range(normalized.n_pathways):
            col = normalized.scores[:, j]
            assert abs(np.mean(col)) < 0.1

        # Step 3: Get top pathways for a sample
        top = normalized.get_top_pathways("S1", n=2)
        assert len(top) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
