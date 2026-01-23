"""
Subtype Characterization

Generates biological profiles for identified autism subtypes based on
pathway disruption patterns and gene involvement.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

import numpy as np
from scipy import stats

import sys
from pathlib import Path
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from clustering import ClusteringResult

logger = logging.getLogger(__name__)


@dataclass
class PathwaySignature:
    """Signature pathways for a subtype."""

    pathway_id: str
    pathway_name: str
    mean_score: float  # Mean score in this subtype
    std_score: float  # Standard deviation
    fold_change: float  # Fold change vs other subtypes
    p_value: float  # Statistical significance
    effect_size: float  # Cohen's d effect size
    direction: str  # "up" or "down" regulated


@dataclass
class SubtypeProfile:
    """Biological profile for an identified subtype."""

    subtype_id: int
    n_samples: int
    sample_ids: List[str]

    # Pathway signatures
    top_pathways: List[PathwaySignature]
    pathway_scores_mean: Dict[str, float]  # Mean score per pathway
    pathway_scores_std: Dict[str, float]  # Std per pathway

    # Gene involvement
    top_genes: List[Tuple[str, float]]  # (gene_id, importance)
    gene_frequency: Dict[str, float]  # How often each gene is hit

    # Cluster statistics
    centroid: np.ndarray  # Cluster centroid in pathway space
    within_cluster_variance: float
    silhouette_score: float

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Generate text summary of subtype."""
        top_pw = ", ".join([p.pathway_name for p in self.top_pathways[:3]])
        return (
            f"Subtype {self.subtype_id}: {self.n_samples} samples, "
            f"characterized by {top_pw}"
        )


@dataclass
class CharacterizationConfig:
    """Configuration for subtype characterization."""

    # Pathway analysis
    n_top_pathways: int = 10  # Number of top pathways per subtype
    min_fold_change: float = 1.5  # Minimum fold change for significance
    p_value_threshold: float = 0.05  # P-value cutoff
    use_fdr_correction: bool = True  # Apply FDR correction

    # Gene analysis
    n_top_genes: int = 20  # Number of top genes per subtype
    min_gene_frequency: float = 0.1  # Min frequency to include gene

    # Effect size
    min_effect_size: float = 0.5  # Minimum Cohen's d


class SubtypeCharacterizer:
    """
    Characterizes identified subtypes by their pathway and gene signatures.

    Generates biological profiles that can be used for interpretation
    and downstream analysis.
    """

    def __init__(self, config: Optional[CharacterizationConfig] = None):
        """
        Initialize characterizer.

        Args:
            config: Characterization configuration
        """
        self.config = config or CharacterizationConfig()

    def characterize(
        self,
        clustering_result: ClusteringResult,
        pathway_scores: np.ndarray,
        pathway_ids: List[str],
        pathway_names: Optional[Dict[str, str]] = None,
        gene_contributions: Optional[Dict[str, Dict[str, List[str]]]] = None,
    ) -> List[SubtypeProfile]:
        """
        Generate profiles for all subtypes.

        Args:
            clustering_result: Clustering result with labels
            pathway_scores: Pathway score matrix (n_samples, n_pathways)
            pathway_ids: List of pathway identifiers
            pathway_names: Optional mapping of pathway ID to name
            gene_contributions: Optional dict mapping (sample, pathway) to genes

        Returns:
            List of SubtypeProfile for each subtype
        """
        pathway_names = pathway_names or {p: p for p in pathway_ids}
        labels = clustering_result.labels
        sample_ids = clustering_result.sample_ids
        n_clusters = clustering_result.n_clusters

        profiles = []

        for cluster_id in range(n_clusters):
            # Get samples in this cluster
            cluster_mask = labels == cluster_id
            cluster_sample_ids = [
                sample_ids[i] for i in range(len(sample_ids)) if cluster_mask[i]
            ]
            cluster_scores = pathway_scores[cluster_mask]
            other_scores = pathway_scores[~cluster_mask]

            # Compute pathway signatures
            signatures = self._compute_pathway_signatures(
                cluster_scores,
                other_scores,
                pathway_ids,
                pathway_names,
            )

            # Compute gene involvement
            top_genes = []
            gene_frequency = {}
            if gene_contributions:
                gene_frequency = self._compute_gene_frequency(
                    cluster_sample_ids, pathway_ids, gene_contributions
                )
                top_genes = sorted(
                    gene_frequency.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:self.config.n_top_genes]

            # Compute cluster statistics
            centroid = np.mean(cluster_scores, axis=0)
            within_var = np.mean(np.var(cluster_scores, axis=0))

            # Compute silhouette score for this cluster
            silhouette = self._compute_cluster_silhouette(
                pathway_scores, labels, cluster_id
            )

            # Mean and std per pathway
            pathway_means = {
                pathway_ids[j]: float(np.mean(cluster_scores[:, j]))
                for j in range(len(pathway_ids))
            }
            pathway_stds = {
                pathway_ids[j]: float(np.std(cluster_scores[:, j]))
                for j in range(len(pathway_ids))
            }

            profile = SubtypeProfile(
                subtype_id=cluster_id,
                n_samples=int(np.sum(cluster_mask)),
                sample_ids=cluster_sample_ids,
                top_pathways=signatures[:self.config.n_top_pathways],
                pathway_scores_mean=pathway_means,
                pathway_scores_std=pathway_stds,
                top_genes=top_genes,
                gene_frequency=gene_frequency,
                centroid=centroid,
                within_cluster_variance=within_var,
                silhouette_score=silhouette,
            )
            profiles.append(profile)

        return profiles

    def _compute_pathway_signatures(
        self,
        cluster_scores: np.ndarray,
        other_scores: np.ndarray,
        pathway_ids: List[str],
        pathway_names: Dict[str, str],
    ) -> List[PathwaySignature]:
        """Compute pathway signatures for a cluster."""
        signatures = []
        n_pathways = len(pathway_ids)

        p_values = []

        for j in range(n_pathways):
            cluster_vals = cluster_scores[:, j]
            other_vals = other_scores[:, j]

            # Skip if no variation
            if np.std(cluster_vals) == 0 and np.std(other_vals) == 0:
                continue

            # Compute statistics
            cluster_mean = np.mean(cluster_vals)
            cluster_std = np.std(cluster_vals)
            other_mean = np.mean(other_vals)
            other_std = np.std(other_vals)

            # Fold change (handle near-zero values)
            if abs(other_mean) > 1e-10:
                fold_change = cluster_mean / other_mean
            else:
                fold_change = np.inf if cluster_mean > 0 else 0.0

            # Statistical test (Welch's t-test)
            if len(cluster_vals) > 1 and len(other_vals) > 1:
                _, p_value = stats.ttest_ind(
                    cluster_vals, other_vals, equal_var=False
                )
            else:
                p_value = 1.0

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                ((len(cluster_vals) - 1) * cluster_std**2 +
                 (len(other_vals) - 1) * other_std**2) /
                (len(cluster_vals) + len(other_vals) - 2)
            )
            if pooled_std > 0:
                effect_size = (cluster_mean - other_mean) / pooled_std
            else:
                effect_size = 0.0

            # Direction
            direction = "up" if cluster_mean > other_mean else "down"

            signature = PathwaySignature(
                pathway_id=pathway_ids[j],
                pathway_name=pathway_names.get(pathway_ids[j], pathway_ids[j]),
                mean_score=cluster_mean,
                std_score=cluster_std,
                fold_change=fold_change,
                p_value=p_value,
                effect_size=effect_size,
                direction=direction,
            )
            signatures.append(signature)
            p_values.append(p_value)

        # Apply FDR correction if configured
        if self.config.use_fdr_correction and len(p_values) > 0:
            adjusted_p = self._fdr_correction(p_values)
            for i, sig in enumerate(signatures):
                sig.p_value = adjusted_p[i]

        # Filter by significance criteria
        significant = [
            s for s in signatures
            if (s.p_value < self.config.p_value_threshold and
                abs(s.effect_size) >= self.config.min_effect_size)
        ]

        # Sort by effect size (absolute value)
        significant.sort(key=lambda x: abs(x.effect_size), reverse=True)

        return significant

    def _fdr_correction(self, p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg FDR correction."""
        n = len(p_values)
        if n == 0:
            return []

        # Sort p-values and track original indices
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        # Compute adjusted p-values
        adjusted = np.zeros(n)
        for i in range(n):
            rank = i + 1
            adjusted[sorted_indices[i]] = sorted_p[i] * n / rank

        # Ensure monotonicity (adjusted p-values should not decrease)
        for i in range(n - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1] if i + 1 < n else 1.0)

        # Clip to [0, 1]
        adjusted = np.clip(adjusted, 0, 1)

        return adjusted.tolist()

    def _compute_gene_frequency(
        self,
        sample_ids: List[str],
        pathway_ids: List[str],
        gene_contributions: Dict[str, Dict[str, List[str]]],
    ) -> Dict[str, float]:
        """Compute how often each gene contributes across samples in cluster."""
        gene_counts: Dict[str, int] = {}
        n_samples = len(sample_ids)

        for sample_id in sample_ids:
            sample_genes: Set[str] = set()

            for pathway_id in pathway_ids:
                key = f"{sample_id}:{pathway_id}"
                if key in gene_contributions:
                    sample_genes.update(gene_contributions[key])

            for gene in sample_genes:
                gene_counts[gene] = gene_counts.get(gene, 0) + 1

        # Convert to frequency
        gene_frequency = {
            gene: count / n_samples
            for gene, count in gene_counts.items()
            if count / n_samples >= self.config.min_gene_frequency
        }

        return gene_frequency

    def _compute_cluster_silhouette(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        cluster_id: int,
    ) -> float:
        """Compute silhouette score for a specific cluster."""
        cluster_mask = labels == cluster_id
        n_in_cluster = np.sum(cluster_mask)

        if n_in_cluster <= 1:
            return 0.0

        cluster_points = data[cluster_mask]
        other_points = data[~cluster_mask]
        other_labels = labels[~cluster_mask]

        silhouettes = []

        for i in range(n_in_cluster):
            point = cluster_points[i]

            # Mean distance to other points in same cluster (a)
            other_cluster_points = np.delete(cluster_points, i, axis=0)
            if len(other_cluster_points) > 0:
                a = np.mean(np.linalg.norm(other_cluster_points - point, axis=1))
            else:
                a = 0.0

            # Mean distance to nearest other cluster (b)
            b = np.inf
            for other_cluster_id in np.unique(labels):
                if other_cluster_id == cluster_id:
                    continue
                other_cluster_mask = other_labels == other_cluster_id
                if np.sum(other_cluster_mask) == 0:
                    continue
                other_cluster_points = other_points[other_cluster_mask]
                mean_dist = np.mean(np.linalg.norm(other_cluster_points - point, axis=1))
                b = min(b, mean_dist)

            if b == np.inf:
                b = 0.0

            # Silhouette
            if max(a, b) > 0:
                s = (b - a) / max(a, b)
            else:
                s = 0.0

            silhouettes.append(s)

        return float(np.mean(silhouettes))

    def compare_subtypes(
        self,
        profiles: List[SubtypeProfile],
        pathway_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Compare subtypes to identify distinguishing features.

        Args:
            profiles: List of subtype profiles
            pathway_ids: List of pathway identifiers

        Returns:
            Dictionary with comparison results
        """
        n_subtypes = len(profiles)

        # Build comparison matrix
        comparison_matrix = np.zeros((n_subtypes, len(pathway_ids)))
        for i, profile in enumerate(profiles):
            for j, pathway_id in enumerate(pathway_ids):
                comparison_matrix[i, j] = profile.pathway_scores_mean.get(pathway_id, 0.0)

        # Find pathways that discriminate between subtypes
        discriminating_pathways = []
        for j, pathway_id in enumerate(pathway_ids):
            pathway_scores = comparison_matrix[:, j]
            variance = np.var(pathway_scores)
            if variance > 0:
                discriminating_pathways.append((pathway_id, variance))

        discriminating_pathways.sort(key=lambda x: x[1], reverse=True)

        # Compute pairwise subtype distances
        pairwise_distances = np.zeros((n_subtypes, n_subtypes))
        for i in range(n_subtypes):
            for j in range(n_subtypes):
                if i != j:
                    dist = np.linalg.norm(
                        profiles[i].centroid - profiles[j].centroid
                    )
                    pairwise_distances[i, j] = dist

        return {
            "comparison_matrix": comparison_matrix,
            "discriminating_pathways": discriminating_pathways[:20],
            "pairwise_distances": pairwise_distances,
            "n_subtypes": n_subtypes,
        }

    def generate_report(
        self,
        profiles: List[SubtypeProfile],
    ) -> str:
        """
        Generate text report summarizing subtype profiles.

        Args:
            profiles: List of subtype profiles

        Returns:
            Formatted text report
        """
        lines = ["=" * 60]
        lines.append("SUBTYPE CHARACTERIZATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        for profile in profiles:
            lines.append(f"SUBTYPE {profile.subtype_id}")
            lines.append("-" * 40)
            lines.append(f"  Samples: {profile.n_samples}")
            lines.append(f"  Silhouette Score: {profile.silhouette_score:.3f}")
            lines.append(f"  Within-cluster Variance: {profile.within_cluster_variance:.3f}")
            lines.append("")

            lines.append("  Top Disrupted Pathways:")
            for i, pathway in enumerate(profile.top_pathways[:5], 1):
                lines.append(
                    f"    {i}. {pathway.pathway_name} "
                    f"(effect={pathway.effect_size:.2f}, p={pathway.p_value:.4f}, {pathway.direction})"
                )
            lines.append("")

            if profile.top_genes:
                lines.append("  Top Contributing Genes:")
                for gene, freq in profile.top_genes[:5]:
                    lines.append(f"    - {gene}: {freq:.1%} of samples")
                lines.append("")

            lines.append("")

        return "\n".join(lines)
