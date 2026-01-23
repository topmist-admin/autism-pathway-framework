"""
Subtype Clustering

Implements clustering algorithms for identifying ASD subtypes based on
pathway-level disruption patterns.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

logger = logging.getLogger(__name__)


class ClusteringMethod(Enum):
    """Available clustering methods."""

    GMM = "gmm"  # Gaussian Mixture Model
    SPECTRAL = "spectral"  # Spectral clustering
    HIERARCHICAL = "hierarchical"  # Agglomerative hierarchical
    KMEANS = "kmeans"  # K-means clustering


@dataclass
class ClusteringConfig:
    """Configuration for subtype clustering."""

    method: ClusteringMethod = ClusteringMethod.GMM

    # Number of clusters
    n_clusters: Optional[int] = None  # If None, auto-detect
    min_clusters: int = 2
    max_clusters: int = 10

    # GMM-specific parameters
    gmm_covariance_type: str = "full"  # full, tied, diag, spherical
    gmm_n_init: int = 10
    gmm_max_iter: int = 200

    # Spectral clustering parameters
    spectral_affinity: str = "rbf"  # rbf, nearest_neighbors
    spectral_n_neighbors: int = 10

    # Hierarchical clustering parameters
    hierarchical_linkage: str = "ward"  # ward, complete, average, single
    hierarchical_distance: str = "euclidean"

    # Preprocessing
    scale_features: bool = True
    remove_zero_variance: bool = True

    # Model selection
    use_bic_for_gmm: bool = True  # Use BIC for GMM cluster selection
    use_silhouette: bool = True  # Use silhouette score for other methods

    # Random state for reproducibility
    random_state: int = 42


@dataclass
class ClusteringResult:
    """Result of clustering analysis."""

    labels: np.ndarray  # Cluster assignments (n_samples,)
    probabilities: np.ndarray  # Soft membership probabilities (n_samples, n_clusters)
    n_clusters: int
    model: Any  # Fitted model for prediction
    method: str
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def cluster_sizes(self) -> Dict[int, int]:
        """Get size of each cluster."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.astype(int), counts.astype(int)))

    def get_cluster_samples(self, cluster_id: int) -> np.ndarray:
        """Get indices of samples in a cluster."""
        return np.where(self.labels == cluster_id)[0]

    def get_cluster_probability(self, sample_idx: int) -> Dict[int, float]:
        """Get cluster membership probabilities for a sample."""
        if self.probabilities is None:
            return {int(self.labels[sample_idx]): 1.0}
        return {i: float(p) for i, p in enumerate(self.probabilities[sample_idx])}


class SubtypeClusterer:
    """
    Clusters samples into subtypes based on pathway disruption patterns.

    Supports multiple clustering algorithms:
    - Gaussian Mixture Model (GMM): Soft clustering with probabilistic membership
    - Spectral Clustering: Graph-based clustering for non-convex clusters
    - Hierarchical Clustering: Agglomerative clustering with dendrogram
    - K-means: Fast centroid-based clustering
    """

    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Initialize subtype clusterer.

        Args:
            config: Clustering configuration
        """
        self.config = config or ClusteringConfig()
        self._scaler: Optional[StandardScaler] = None
        self._fitted_model: Optional[Any] = None
        self._feature_mask: Optional[np.ndarray] = None

    def fit(
        self,
        pathway_scores: Any,  # PathwayScoreMatrix from Module 07
        n_clusters: Optional[int] = None,
    ) -> ClusteringResult:
        """
        Fit clustering model to pathway scores.

        Args:
            pathway_scores: PathwayScoreMatrix with pathway-level scores
            n_clusters: Number of clusters (overrides config if provided)

        Returns:
            ClusteringResult with cluster assignments
        """
        # Extract scores matrix
        X = pathway_scores.scores.copy()
        n_clusters = n_clusters or self.config.n_clusters

        logger.info(
            f"Clustering {X.shape[0]} samples with {X.shape[1]} pathway features"
        )

        # Preprocess data
        X_processed = self._preprocess(X, fit=True)

        # Determine number of clusters if not specified
        if n_clusters is None:
            n_clusters = self._select_n_clusters(X_processed)
            logger.info(f"Auto-selected {n_clusters} clusters")

        # Fit clustering model
        method = self.config.method

        if method == ClusteringMethod.GMM:
            result = self._fit_gmm(X_processed, n_clusters)
        elif method == ClusteringMethod.SPECTRAL:
            result = self._fit_spectral(X_processed, n_clusters)
        elif method == ClusteringMethod.HIERARCHICAL:
            result = self._fit_hierarchical(X_processed, n_clusters)
        elif method == ClusteringMethod.KMEANS:
            result = self._fit_kmeans(X_processed, n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Compute evaluation metrics
        result.metrics = self._compute_metrics(X_processed, result.labels)

        # Store metadata
        result.metadata = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_features_used": X_processed.shape[1],
            "samples": pathway_scores.samples,
            "pathways": pathway_scores.pathways,
        }

        self._fitted_model = result.model

        logger.info(
            f"Clustering complete: {n_clusters} clusters, "
            f"silhouette={result.metrics.get('silhouette', 'N/A'):.3f}"
        )

        return result

    def predict(
        self,
        pathway_scores: Any,  # PathwayScoreMatrix
    ) -> np.ndarray:
        """
        Predict cluster labels for new samples.

        Args:
            pathway_scores: PathwayScoreMatrix with pathway-level scores

        Returns:
            Array of cluster labels
        """
        if self._fitted_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = pathway_scores.scores.copy()
        X_processed = self._preprocess(X, fit=False)

        if hasattr(self._fitted_model, "predict"):
            return self._fitted_model.predict(X_processed)
        else:
            raise RuntimeError("Fitted model does not support prediction")

    def predict_proba(
        self,
        pathway_scores: Any,  # PathwayScoreMatrix
    ) -> np.ndarray:
        """
        Predict cluster membership probabilities for new samples.

        Args:
            pathway_scores: PathwayScoreMatrix with pathway-level scores

        Returns:
            Array of shape (n_samples, n_clusters) with probabilities
        """
        if self._fitted_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = pathway_scores.scores.copy()
        X_processed = self._preprocess(X, fit=False)

        if hasattr(self._fitted_model, "predict_proba"):
            return self._fitted_model.predict_proba(X_processed)
        else:
            # Return hard assignments as one-hot
            labels = self.predict(pathway_scores)
            n_clusters = len(np.unique(labels))
            proba = np.zeros((len(labels), n_clusters))
            proba[np.arange(len(labels)), labels] = 1.0
            return proba

    def _preprocess(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Preprocess data for clustering."""
        X_processed = X.copy()

        # Remove zero-variance features
        if self.config.remove_zero_variance:
            if fit:
                variances = np.var(X_processed, axis=0)
                self._feature_mask = variances > 1e-10
            if self._feature_mask is not None:
                X_processed = X_processed[:, self._feature_mask]

        # Scale features
        if self.config.scale_features:
            if fit:
                self._scaler = StandardScaler()
                X_processed = self._scaler.fit_transform(X_processed)
            elif self._scaler is not None:
                X_processed = self._scaler.transform(X_processed)

        return X_processed

    def _select_n_clusters(self, X: np.ndarray) -> int:
        """Auto-select number of clusters using BIC or silhouette."""
        min_k = self.config.min_clusters
        max_k = min(self.config.max_clusters, X.shape[0] - 1)

        if self.config.method == ClusteringMethod.GMM and self.config.use_bic_for_gmm:
            # Use BIC for GMM
            bic_scores = []
            for k in range(min_k, max_k + 1):
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=self.config.gmm_covariance_type,
                    n_init=self.config.gmm_n_init,
                    random_state=self.config.random_state,
                )
                gmm.fit(X)
                bic_scores.append(gmm.bic(X))

            # Select k with lowest BIC
            best_k = min_k + np.argmin(bic_scores)
            return best_k

        elif self.config.use_silhouette:
            # Use silhouette score
            silhouette_scores = []
            for k in range(min_k, max_k + 1):
                if self.config.method == ClusteringMethod.KMEANS:
                    model = KMeans(n_clusters=k, random_state=self.config.random_state)
                else:
                    model = KMeans(n_clusters=k, random_state=self.config.random_state)

                labels = model.fit_predict(X)
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X, labels)
                else:
                    score = -1
                silhouette_scores.append(score)

            # Select k with highest silhouette
            best_k = min_k + np.argmax(silhouette_scores)
            return best_k

        else:
            # Default to middle of range
            return (min_k + max_k) // 2

    def _fit_gmm(self, X: np.ndarray, n_clusters: int) -> ClusteringResult:
        """Fit Gaussian Mixture Model."""
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=self.config.gmm_covariance_type,
            n_init=self.config.gmm_n_init,
            max_iter=self.config.gmm_max_iter,
            random_state=self.config.random_state,
        )

        labels = gmm.fit_predict(X)
        probabilities = gmm.predict_proba(X)

        return ClusteringResult(
            labels=labels,
            probabilities=probabilities,
            n_clusters=n_clusters,
            model=gmm,
            method="gmm",
            metadata={"bic": gmm.bic(X), "aic": gmm.aic(X)},
        )

    def _fit_spectral(self, X: np.ndarray, n_clusters: int) -> ClusteringResult:
        """Fit Spectral Clustering."""
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity=self.config.spectral_affinity,
            n_neighbors=self.config.spectral_n_neighbors,
            random_state=self.config.random_state,
        )

        labels = spectral.fit_predict(X)

        # Spectral clustering doesn't provide probabilities
        # Use distance to cluster centers as proxy
        probabilities = self._compute_distance_probabilities(X, labels, n_clusters)

        return ClusteringResult(
            labels=labels,
            probabilities=probabilities,
            n_clusters=n_clusters,
            model=spectral,
            method="spectral",
        )

    def _fit_hierarchical(self, X: np.ndarray, n_clusters: int) -> ClusteringResult:
        """Fit Hierarchical Clustering."""
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.config.hierarchical_linkage,
        )

        labels = hierarchical.fit_predict(X)

        # Compute distance-based probabilities
        probabilities = self._compute_distance_probabilities(X, labels, n_clusters)

        # Also compute linkage for dendrogram
        linkage_matrix = linkage(
            X,
            method=self.config.hierarchical_linkage,
            metric=self.config.hierarchical_distance,
        )

        return ClusteringResult(
            labels=labels,
            probabilities=probabilities,
            n_clusters=n_clusters,
            model=hierarchical,
            method="hierarchical",
            metadata={"linkage_matrix": linkage_matrix},
        )

    def _fit_kmeans(self, X: np.ndarray, n_clusters: int) -> ClusteringResult:
        """Fit K-means Clustering."""
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=self.config.random_state,
        )

        labels = kmeans.fit_predict(X)

        # Compute soft assignments based on distance to centroids
        distances = kmeans.transform(X)  # Distance to each centroid
        # Convert distances to probabilities (inverse distance, normalized)
        inv_distances = 1.0 / (distances + 1e-10)
        probabilities = inv_distances / inv_distances.sum(axis=1, keepdims=True)

        return ClusteringResult(
            labels=labels,
            probabilities=probabilities,
            n_clusters=n_clusters,
            model=kmeans,
            method="kmeans",
            metadata={"inertia": kmeans.inertia_},
        )

    def _compute_distance_probabilities(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
    ) -> np.ndarray:
        """Compute probability-like scores based on distance to cluster centers."""
        # Compute cluster centers
        centers = np.zeros((n_clusters, X.shape[1]))
        for k in range(n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                centers[k] = X[mask].mean(axis=0)

        # Compute distances to centers
        distances = np.zeros((X.shape[0], n_clusters))
        for k in range(n_clusters):
            distances[:, k] = np.sqrt(np.sum((X - centers[k]) ** 2, axis=1))

        # Convert to probabilities (inverse distance, normalized)
        inv_distances = 1.0 / (distances + 1e-10)
        probabilities = inv_distances / inv_distances.sum(axis=1, keepdims=True)

        return probabilities

    def _compute_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Compute clustering evaluation metrics."""
        metrics = {}

        n_clusters = len(np.unique(labels))
        if n_clusters < 2 or n_clusters >= X.shape[0]:
            return metrics

        try:
            metrics["silhouette"] = silhouette_score(X, labels)
        except Exception:
            pass

        try:
            metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
        except Exception:
            pass

        try:
            metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
        except Exception:
            pass

        return metrics

    def fit_multiple_methods(
        self,
        pathway_scores: Any,
        methods: Optional[List[ClusteringMethod]] = None,
        n_clusters: Optional[int] = None,
    ) -> Dict[str, ClusteringResult]:
        """
        Fit multiple clustering methods for comparison.

        Args:
            pathway_scores: PathwayScoreMatrix
            methods: List of methods to try (default: all)
            n_clusters: Number of clusters

        Returns:
            Dictionary mapping method name to ClusteringResult
        """
        if methods is None:
            methods = list(ClusteringMethod)

        results = {}
        original_method = self.config.method

        for method in methods:
            self.config.method = method
            try:
                results[method.value] = self.fit(pathway_scores, n_clusters)
            except Exception as e:
                logger.warning(f"Method {method.value} failed: {e}")

        self.config.method = original_method
        return results

    def get_consensus_clustering(
        self,
        pathway_scores: Any,
        n_runs: int = 10,
        n_clusters: Optional[int] = None,
    ) -> ClusteringResult:
        """
        Perform consensus clustering by aggregating multiple runs.

        Args:
            pathway_scores: PathwayScoreMatrix
            n_runs: Number of clustering runs
            n_clusters: Number of clusters

        Returns:
            Consensus ClusteringResult
        """
        n_samples = pathway_scores.scores.shape[0]
        co_occurrence = np.zeros((n_samples, n_samples))

        # Run clustering multiple times with different random states
        original_state = self.config.random_state

        for i in range(n_runs):
            self.config.random_state = original_state + i
            result = self.fit(pathway_scores, n_clusters)

            # Update co-occurrence matrix
            for j in range(n_samples):
                for k in range(j + 1, n_samples):
                    if result.labels[j] == result.labels[k]:
                        co_occurrence[j, k] += 1
                        co_occurrence[k, j] += 1

        self.config.random_state = original_state

        # Normalize co-occurrence
        co_occurrence /= n_runs

        # Cluster co-occurrence matrix using hierarchical clustering
        # Convert similarity to distance
        distance_matrix = 1 - co_occurrence
        np.fill_diagonal(distance_matrix, 0)

        condensed = pdist(distance_matrix)
        Z = linkage(condensed, method="average")

        if n_clusters is None:
            n_clusters = self._select_n_clusters(pathway_scores.scores)

        consensus_labels = fcluster(Z, n_clusters, criterion="maxclust") - 1

        # Compute probabilities from co-occurrence
        probabilities = self._compute_distance_probabilities(
            pathway_scores.scores, consensus_labels, n_clusters
        )

        return ClusteringResult(
            labels=consensus_labels,
            probabilities=probabilities,
            n_clusters=n_clusters,
            model=None,
            method="consensus",
            metadata={
                "n_runs": n_runs,
                "co_occurrence_matrix": co_occurrence,
            },
        )
