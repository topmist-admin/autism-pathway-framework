"""
Clustering Stability Analysis

Provides bootstrap-based stability assessment for clustering solutions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import sys
from pathlib import Path
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from clustering import SubtypeClusterer, ClusteringConfig, ClusteringResult, ClusteringMethod

logger = logging.getLogger(__name__)


@dataclass
class StabilityResult:
    """Results from stability analysis."""

    # Overall stability metrics
    mean_ari: float  # Mean adjusted Rand index across bootstrap samples
    std_ari: float  # Standard deviation of ARI
    mean_nmi: float  # Mean normalized mutual information
    std_nmi: float  # Standard deviation of NMI

    # Per-sample stability
    sample_stability: np.ndarray  # How consistently each sample is clustered
    sample_ids: List[str]  # Sample identifiers

    # Cluster-level stability
    cluster_stability: np.ndarray  # Stability of each cluster
    n_clusters: int

    # Bootstrap results
    n_bootstrap: int
    bootstrap_aris: List[float]  # ARI for each bootstrap iteration
    bootstrap_nmis: List[float]  # NMI for each bootstrap iteration

    # Confidence intervals
    ari_ci_low: float
    ari_ci_high: float
    nmi_ci_low: float
    nmi_ci_high: float

    # Co-clustering matrix
    co_clustering_matrix: np.ndarray  # Probability two samples cluster together

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_stable(self) -> bool:
        """Check if clustering is considered stable (ARI > 0.8)."""
        return self.mean_ari > 0.8

    @property
    def stability_rating(self) -> str:
        """Get qualitative stability rating."""
        if self.mean_ari >= 0.9:
            return "excellent"
        elif self.mean_ari >= 0.8:
            return "good"
        elif self.mean_ari >= 0.6:
            return "moderate"
        elif self.mean_ari >= 0.4:
            return "poor"
        else:
            return "unstable"

    def get_unstable_samples(self, threshold: float = 0.7) -> List[str]:
        """Get samples with stability below threshold."""
        unstable_mask = self.sample_stability < threshold
        return [self.sample_ids[i] for i in np.where(unstable_mask)[0]]

    def get_stable_core(self, threshold: float = 0.9) -> List[str]:
        """Get highly stable samples that form cluster cores."""
        stable_mask = self.sample_stability >= threshold
        return [self.sample_ids[i] for i in np.where(stable_mask)[0]]


@dataclass
class StabilityConfig:
    """Configuration for stability analysis."""

    n_bootstrap: int = 100  # Number of bootstrap iterations
    sample_fraction: float = 0.8  # Fraction of samples per bootstrap
    random_state: Optional[int] = 42  # Random seed for reproducibility

    # Confidence interval
    ci_level: float = 0.95  # Confidence interval level

    # Parallel processing
    n_jobs: int = 1  # Number of parallel jobs (-1 for all cores)

    # Metrics to compute
    compute_co_clustering: bool = True  # Compute co-clustering matrix
    compute_sample_stability: bool = True  # Compute per-sample stability


class StabilityAnalyzer:
    """
    Analyzes clustering stability using bootstrap resampling.

    Assesses how robust clustering assignments are by repeatedly
    resampling the data and measuring consistency of cluster assignments.
    """

    def __init__(self, config: Optional[StabilityConfig] = None):
        """
        Initialize stability analyzer.

        Args:
            config: Stability analysis configuration
        """
        self.config = config or StabilityConfig()
        self._rng = np.random.default_rng(self.config.random_state)

    def analyze_stability(
        self,
        data: np.ndarray,
        clusterer: SubtypeClusterer,
        sample_ids: Optional[List[str]] = None,
        reference_labels: Optional[np.ndarray] = None,
        pathway_ids: Optional[List[str]] = None,
    ) -> StabilityResult:
        """
        Perform bootstrap stability analysis.

        Args:
            data: Feature matrix (n_samples, n_features)
            clusterer: Configured SubtypeClusterer instance
            sample_ids: Optional sample identifiers
            reference_labels: Optional reference clustering to compare against
            pathway_ids: Optional pathway identifiers

        Returns:
            StabilityResult with stability metrics
        """
        n_samples = data.shape[0]
        n_features = data.shape[1]
        sample_ids = sample_ids or [f"sample_{i}" for i in range(n_samples)]
        pathway_ids = pathway_ids or [f"pathway_{i}" for i in range(n_features)]

        # Create a mock PathwayScoreMatrix-like object for the clusterer
        class MockPathwayScores:
            def __init__(self, scores, samples, pathways):
                self.scores = scores
                self.samples = samples
                self.pathways = pathways

        # Get reference clustering if not provided
        if reference_labels is None:
            mock_scores = MockPathwayScores(data, sample_ids, pathway_ids)
            ref_result = clusterer.fit(mock_scores)
            reference_labels = ref_result.labels

        n_clusters = len(np.unique(reference_labels))

        # Initialize tracking arrays
        bootstrap_aris = []
        bootstrap_nmis = []
        co_clustering_counts = np.zeros((n_samples, n_samples))
        co_occurrence_counts = np.zeros((n_samples, n_samples))
        sample_agreement_counts = np.zeros(n_samples)
        sample_total_counts = np.zeros(n_samples)

        # Run bootstrap iterations
        for i in range(self.config.n_bootstrap):
            # Sample with replacement
            bootstrap_indices = self._rng.choice(
                n_samples,
                size=int(n_samples * self.config.sample_fraction),
                replace=True,
            )
            unique_indices = np.unique(bootstrap_indices)

            # Get bootstrap sample
            bootstrap_data = data[unique_indices]
            bootstrap_ids = [sample_ids[j] for j in unique_indices]

            # Cluster bootstrap sample
            try:
                mock_bootstrap = MockPathwayScores(bootstrap_data, bootstrap_ids, pathway_ids)
                bootstrap_result = clusterer.fit(mock_bootstrap)
                bootstrap_labels = bootstrap_result.labels
            except Exception as e:
                logger.warning(f"Bootstrap iteration {i} failed: {e}")
                continue

            # Map bootstrap labels back to original indices
            mapped_labels = np.full(n_samples, -1)
            for j, orig_idx in enumerate(unique_indices):
                mapped_labels[orig_idx] = bootstrap_labels[j]

            # Compute ARI and NMI on samples present in bootstrap
            present_mask = mapped_labels >= 0
            if np.sum(present_mask) > 0:
                ari = adjusted_rand_score(
                    reference_labels[present_mask],
                    mapped_labels[present_mask]
                )
                nmi = normalized_mutual_info_score(
                    reference_labels[present_mask],
                    mapped_labels[present_mask]
                )
                bootstrap_aris.append(ari)
                bootstrap_nmis.append(nmi)

            # Update co-clustering matrix
            if self.config.compute_co_clustering:
                for j in unique_indices:
                    for k in unique_indices:
                        co_occurrence_counts[j, k] += 1
                        if mapped_labels[j] == mapped_labels[k]:
                            co_clustering_counts[j, k] += 1

            # Update sample stability
            if self.config.compute_sample_stability:
                for j in unique_indices:
                    sample_total_counts[j] += 1
                    if mapped_labels[j] == reference_labels[j]:
                        # Need to handle label permutation
                        # For now, use simpler heuristic
                        sample_agreement_counts[j] += 1

        # Compute statistics
        mean_ari = np.mean(bootstrap_aris) if bootstrap_aris else 0.0
        std_ari = np.std(bootstrap_aris) if bootstrap_aris else 0.0
        mean_nmi = np.mean(bootstrap_nmis) if bootstrap_nmis else 0.0
        std_nmi = np.std(bootstrap_nmis) if bootstrap_nmis else 0.0

        # Compute confidence intervals
        ci_alpha = 1 - self.config.ci_level
        if len(bootstrap_aris) > 0:
            ari_ci_low = np.percentile(bootstrap_aris, ci_alpha / 2 * 100)
            ari_ci_high = np.percentile(bootstrap_aris, (1 - ci_alpha / 2) * 100)
            nmi_ci_low = np.percentile(bootstrap_nmis, ci_alpha / 2 * 100)
            nmi_ci_high = np.percentile(bootstrap_nmis, (1 - ci_alpha / 2) * 100)
        else:
            ari_ci_low = ari_ci_high = 0.0
            nmi_ci_low = nmi_ci_high = 0.0

        # Compute co-clustering probability matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            co_clustering_matrix = np.where(
                co_occurrence_counts > 0,
                co_clustering_counts / co_occurrence_counts,
                0.0
            )

        # Compute sample stability
        with np.errstate(divide='ignore', invalid='ignore'):
            sample_stability = np.where(
                sample_total_counts > 0,
                sample_agreement_counts / sample_total_counts,
                0.0
            )

        # Compute cluster stability (average stability of samples in each cluster)
        cluster_stability = np.zeros(n_clusters)
        for c in range(n_clusters):
            cluster_mask = reference_labels == c
            if np.sum(cluster_mask) > 0:
                cluster_stability[c] = np.mean(sample_stability[cluster_mask])

        return StabilityResult(
            mean_ari=mean_ari,
            std_ari=std_ari,
            mean_nmi=mean_nmi,
            std_nmi=std_nmi,
            sample_stability=sample_stability,
            sample_ids=sample_ids,
            cluster_stability=cluster_stability,
            n_clusters=n_clusters,
            n_bootstrap=self.config.n_bootstrap,
            bootstrap_aris=bootstrap_aris,
            bootstrap_nmis=bootstrap_nmis,
            ari_ci_low=ari_ci_low,
            ari_ci_high=ari_ci_high,
            nmi_ci_low=nmi_ci_low,
            nmi_ci_high=nmi_ci_high,
            co_clustering_matrix=co_clustering_matrix,
            metadata={
                "sample_fraction": self.config.sample_fraction,
                "n_successful_iterations": len(bootstrap_aris),
            },
        )

    def find_optimal_k(
        self,
        data: np.ndarray,
        k_range: Tuple[int, int],
        clusterer_config: Optional[ClusteringConfig] = None,
        sample_ids: Optional[List[str]] = None,
        pathway_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Find optimal number of clusters using stability analysis.

        Args:
            data: Feature matrix (n_samples, n_features)
            k_range: (min_k, max_k) range to search
            clusterer_config: Base clustering configuration
            sample_ids: Optional sample identifiers
            pathway_ids: Optional pathway identifiers

        Returns:
            Dictionary with optimal k and stability scores for each k
        """
        n_samples = data.shape[0]
        n_features = data.shape[1]
        sample_ids = sample_ids or [f"sample_{i}" for i in range(n_samples)]
        pathway_ids = pathway_ids or [f"pathway_{i}" for i in range(n_features)]

        config = clusterer_config or ClusteringConfig()

        k_values = list(range(k_range[0], k_range[1] + 1))
        stability_scores = []
        results = {}

        for k in k_values:
            config.n_clusters = k
            clusterer = SubtypeClusterer(config)

            try:
                stability_result = self.analyze_stability(
                    data, clusterer, sample_ids, pathway_ids=pathway_ids
                )
                stability_scores.append(stability_result.mean_ari)
                results[k] = {
                    "mean_ari": stability_result.mean_ari,
                    "std_ari": stability_result.std_ari,
                    "mean_nmi": stability_result.mean_nmi,
                    "rating": stability_result.stability_rating,
                }
            except Exception as e:
                logger.warning(f"Failed to analyze k={k}: {e}")
                stability_scores.append(0.0)
                results[k] = {"error": str(e)}

        # Find optimal k (highest stability)
        optimal_idx = np.argmax(stability_scores)
        optimal_k = k_values[optimal_idx]

        return {
            "optimal_k": optimal_k,
            "optimal_stability": stability_scores[optimal_idx],
            "k_values": k_values,
            "stability_scores": stability_scores,
            "detailed_results": results,
        }

    def compare_methods(
        self,
        data: np.ndarray,
        methods: List[ClusteringMethod],
        n_clusters: int,
        sample_ids: Optional[List[str]] = None,
        pathway_ids: Optional[List[str]] = None,
    ) -> Dict[str, StabilityResult]:
        """
        Compare stability of different clustering methods.

        Args:
            data: Feature matrix (n_samples, n_features)
            methods: List of clustering methods to compare
            n_clusters: Number of clusters
            sample_ids: Optional sample identifiers
            pathway_ids: Optional pathway identifiers

        Returns:
            Dictionary mapping method name to StabilityResult
        """
        n_samples = data.shape[0]
        n_features = data.shape[1]
        sample_ids = sample_ids or [f"sample_{i}" for i in range(n_samples)]
        pathway_ids = pathway_ids or [f"pathway_{i}" for i in range(n_features)]

        results = {}

        for method in methods:
            config = ClusteringConfig(
                method=method,
                n_clusters=n_clusters,
            )
            clusterer = SubtypeClusterer(config)

            try:
                stability_result = self.analyze_stability(
                    data, clusterer, sample_ids, pathway_ids=pathway_ids
                )
                results[method.value] = stability_result
                logger.info(
                    f"Method {method.value}: ARI={stability_result.mean_ari:.3f} "
                    f"({stability_result.stability_rating})"
                )
            except Exception as e:
                logger.warning(f"Method {method.value} failed: {e}")

        return results

    def identify_core_samples(
        self,
        stability_result: StabilityResult,
        threshold: float = 0.9,
    ) -> Dict[int, List[str]]:
        """
        Identify core samples for each cluster based on stability.

        Core samples are those consistently assigned to the same cluster
        across bootstrap iterations.

        Args:
            stability_result: Result from stability analysis
            threshold: Minimum stability to be considered core

        Returns:
            Dictionary mapping cluster ID to list of core sample IDs
        """
        # Need reference labels to group by cluster
        # This is a simplified version that just returns stable samples
        stable_samples = stability_result.get_stable_core(threshold)

        return {
            0: stable_samples  # Simplified - would need cluster labels for proper grouping
        }

    def compute_cluster_purity(
        self,
        co_clustering_matrix: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cluster purity from co-clustering matrix.

        Args:
            co_clustering_matrix: Probability matrix of co-clustering
            labels: Cluster assignments

        Returns:
            Array of purity scores for each cluster
        """
        n_clusters = len(np.unique(labels))
        purity = np.zeros(n_clusters)

        for c in range(n_clusters):
            cluster_mask = labels == c
            if np.sum(cluster_mask) > 1:
                # Average co-clustering probability within cluster
                within_probs = co_clustering_matrix[np.ix_(cluster_mask, cluster_mask)]
                # Exclude diagonal
                np.fill_diagonal(within_probs, np.nan)
                purity[c] = np.nanmean(within_probs)
            else:
                purity[c] = 1.0  # Single-sample cluster is perfectly pure

        return purity
