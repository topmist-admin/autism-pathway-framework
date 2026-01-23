"""
Research Integrity Validation Components

Provides confound testing, negative controls, and provenance tracking
for robust and reproducible subtype clustering analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, kruskal, f_oneway
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import sys
from pathlib import Path
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from clustering import SubtypeClusterer, ClusteringConfig, ClusteringResult

logger = logging.getLogger(__name__)


# =============================================================================
# Confound Analysis
# =============================================================================

class ConfoundType(Enum):
    """Types of confounding variables."""
    BATCH = "batch"
    SITE = "site"
    ANCESTRY = "ancestry"
    SEX = "sex"
    AGE = "age"
    SEQUENCING_DEPTH = "sequencing_depth"
    PHENOTYPING_DEPTH = "phenotyping_depth"
    CUSTOM = "custom"


@dataclass
class ConfoundTestResult:
    """Result of a single confound test."""
    confound_name: str
    confound_type: ConfoundType
    test_statistic: float
    p_value: float
    effect_size: float  # Cramer's V for categorical, eta-squared for continuous
    test_method: str  # chi2, kruskal, anova, etc.
    is_significant: bool
    interpretation: str


@dataclass
class ConfoundReport:
    """Comprehensive confound analysis report."""
    test_results: List[ConfoundTestResult]
    overall_risk: str  # "low", "moderate", "high"
    problematic_confounds: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_significant_confounds(self) -> bool:
        """Check if any confound is significantly associated with clusters."""
        return len(self.problematic_confounds) > 0

    def get_summary(self) -> str:
        """Get human-readable summary of confound analysis."""
        lines = [
            f"Confound Analysis Report",
            f"========================",
            f"Overall Risk: {self.overall_risk.upper()}",
            f"Tests Performed: {len(self.test_results)}",
            f"Significant Confounds: {len(self.problematic_confounds)}",
        ]
        if self.problematic_confounds:
            lines.append(f"  - {', '.join(self.problematic_confounds)}")
        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
        return "\n".join(lines)


@dataclass
class ConfoundAnalyzerConfig:
    """Configuration for confound analysis."""
    significance_threshold: float = 0.05
    effect_size_threshold: float = 0.3  # Cramer's V or eta-squared
    apply_bonferroni: bool = True
    min_samples_per_group: int = 5


class ConfoundAnalyzer:
    """
    Tests if clusters align with known confounds.

    Implements confound-first modeling approach from research integrity
    guidelines to detect whether clustering solutions reflect technical
    artifacts rather than biological signal.
    """

    def __init__(self, config: Optional[ConfoundAnalyzerConfig] = None):
        """
        Initialize confound analyzer.

        Args:
            config: Configuration for analysis parameters
        """
        self.config = config or ConfoundAnalyzerConfig()

    def test_cluster_confound_alignment(
        self,
        cluster_labels: np.ndarray,
        confounds: Dict[str, np.ndarray],
        confound_types: Optional[Dict[str, ConfoundType]] = None,
    ) -> ConfoundReport:
        """
        Test if clusters align with known confounds.

        Args:
            cluster_labels: Cluster assignments (n_samples,)
            confounds: Dictionary mapping confound names to values
            confound_types: Optional mapping of confound names to types
                           (inferred if not provided)

        Returns:
            ConfoundReport with test results and recommendations
        """
        confound_types = confound_types or {}
        test_results = []
        n_tests = len(confounds)

        # Adjust significance threshold for multiple testing
        alpha = self.config.significance_threshold
        if self.config.apply_bonferroni and n_tests > 1:
            alpha = alpha / n_tests

        for name, values in confounds.items():
            # Infer confound type if not provided
            c_type = confound_types.get(name, self._infer_confound_type(name, values))

            # Test association
            result = self._test_single_confound(
                cluster_labels, values, name, c_type, alpha
            )
            test_results.append(result)

        # Compile report
        problematic = [r.confound_name for r in test_results if r.is_significant]

        # Determine overall risk
        if len(problematic) == 0:
            risk = "low"
        elif len(problematic) <= len(confounds) * 0.3:
            risk = "moderate"
        else:
            risk = "high"

        # Generate recommendations
        recommendations = self._generate_recommendations(test_results, risk)

        return ConfoundReport(
            test_results=test_results,
            overall_risk=risk,
            problematic_confounds=problematic,
            recommendations=recommendations,
            metadata={
                "n_samples": len(cluster_labels),
                "n_clusters": len(np.unique(cluster_labels)),
                "alpha": alpha,
                "bonferroni_applied": self.config.apply_bonferroni,
            },
        )

    def compute_confound_association(
        self,
        cluster_labels: np.ndarray,
        confound_values: np.ndarray,
        is_categorical: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Compute association between clusters and a confound.

        Args:
            cluster_labels: Cluster assignments
            confound_values: Confound variable values
            is_categorical: Whether confound is categorical

        Returns:
            Tuple of (test_statistic, p_value, effect_size)
        """
        if is_categorical:
            return self._test_categorical_confound(cluster_labels, confound_values)
        else:
            return self._test_continuous_confound(cluster_labels, confound_values)

    def _test_single_confound(
        self,
        cluster_labels: np.ndarray,
        values: np.ndarray,
        name: str,
        c_type: ConfoundType,
        alpha: float,
    ) -> ConfoundTestResult:
        """Test association with a single confound."""
        # Determine if categorical or continuous
        is_categorical = self._is_categorical(values)

        if is_categorical:
            stat, p_value, effect = self._test_categorical_confound(
                cluster_labels, values
            )
            method = "chi-squared"
        else:
            stat, p_value, effect = self._test_continuous_confound(
                cluster_labels, values
            )
            method = "Kruskal-Wallis"

        is_significant = (
            p_value < alpha and
            effect >= self.config.effect_size_threshold
        )

        interpretation = self._interpret_result(
            name, p_value, effect, is_significant
        )

        return ConfoundTestResult(
            confound_name=name,
            confound_type=c_type,
            test_statistic=stat,
            p_value=p_value,
            effect_size=effect,
            test_method=method,
            is_significant=is_significant,
            interpretation=interpretation,
        )

    def _test_categorical_confound(
        self,
        cluster_labels: np.ndarray,
        confound_values: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Chi-squared test for categorical confound."""
        # Create contingency table
        unique_clusters = np.unique(cluster_labels)
        unique_confounds = np.unique(confound_values)

        contingency = np.zeros((len(unique_clusters), len(unique_confounds)))
        for i, c in enumerate(unique_clusters):
            for j, v in enumerate(unique_confounds):
                contingency[i, j] = np.sum(
                    (cluster_labels == c) & (confound_values == v)
                )

        # Chi-squared test
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            # Cramer's V effect size
            n = contingency.sum()
            min_dim = min(contingency.shape) - 1
            if min_dim > 0 and n > 0:
                cramers_v = np.sqrt(chi2 / (n * min_dim))
            else:
                cramers_v = 0.0
        except Exception:
            chi2, p_value, cramers_v = 0.0, 1.0, 0.0

        return chi2, p_value, cramers_v

    def _test_continuous_confound(
        self,
        cluster_labels: np.ndarray,
        confound_values: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Kruskal-Wallis test for continuous confound."""
        unique_clusters = np.unique(cluster_labels)

        # Group values by cluster
        groups = [
            confound_values[cluster_labels == c]
            for c in unique_clusters
        ]

        # Filter out empty groups
        groups = [g for g in groups if len(g) >= self.config.min_samples_per_group]

        if len(groups) < 2:
            return 0.0, 1.0, 0.0

        try:
            stat, p_value = kruskal(*groups)
            # Eta-squared effect size
            n = len(confound_values)
            k = len(groups)
            eta_squared = (stat - k + 1) / (n - k) if n > k else 0.0
            eta_squared = max(0.0, min(1.0, eta_squared))
        except Exception:
            stat, p_value, eta_squared = 0.0, 1.0, 0.0

        return stat, p_value, eta_squared

    def _infer_confound_type(self, name: str, values: np.ndarray) -> ConfoundType:
        """Infer confound type from name and values."""
        name_lower = name.lower()

        if "batch" in name_lower:
            return ConfoundType.BATCH
        elif "site" in name_lower:
            return ConfoundType.SITE
        elif "ancestry" in name_lower or "population" in name_lower:
            return ConfoundType.ANCESTRY
        elif "sex" in name_lower or "gender" in name_lower:
            return ConfoundType.SEX
        elif "age" in name_lower:
            return ConfoundType.AGE
        elif "depth" in name_lower or "coverage" in name_lower:
            return ConfoundType.SEQUENCING_DEPTH
        else:
            return ConfoundType.CUSTOM

    def _is_categorical(self, values: np.ndarray) -> bool:
        """Determine if values are categorical."""
        if values.dtype.kind in ['U', 'S', 'O']:  # String types
            return True
        unique_ratio = len(np.unique(values)) / len(values)
        return unique_ratio < 0.05  # Less than 5% unique = likely categorical

    def _interpret_result(
        self,
        name: str,
        p_value: float,
        effect_size: float,
        is_significant: bool,
    ) -> str:
        """Generate interpretation text."""
        if not is_significant:
            return f"No significant association between clusters and {name}."

        if effect_size >= 0.5:
            strength = "strong"
        elif effect_size >= 0.3:
            strength = "moderate"
        else:
            strength = "weak"

        return (
            f"Clusters show {strength} association with {name} "
            f"(p={p_value:.2e}, effect={effect_size:.3f}). "
            f"Consider stratified analysis or confound regression."
        )

    def _generate_recommendations(
        self,
        results: List[ConfoundTestResult],
        risk: str,
    ) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        if risk == "high":
            recommendations.append(
                "HIGH RISK: Clustering may primarily reflect technical artifacts. "
                "Consider re-clustering after batch correction or stratification."
            )

        for result in results:
            if result.is_significant:
                if result.confound_type == ConfoundType.BATCH:
                    recommendations.append(
                        f"Apply batch correction (e.g., ComBat) before clustering."
                    )
                elif result.confound_type == ConfoundType.ANCESTRY:
                    recommendations.append(
                        f"Perform ancestry-stratified analysis or use ancestry-aware normalization."
                    )
                elif result.confound_type == ConfoundType.SITE:
                    recommendations.append(
                        f"Consider site as a covariate or perform site-stratified validation."
                    )

        if not recommendations:
            recommendations.append(
                "No significant confound associations detected. "
                "Proceed with biological interpretation."
            )

        return list(set(recommendations))  # Remove duplicates


# =============================================================================
# Negative Control Framework
# =============================================================================

@dataclass
class PermutationResult:
    """Result of permutation testing."""
    observed_metric: float
    null_distribution: np.ndarray
    p_value: float
    metric_name: str
    n_permutations: int
    is_significant: bool

    @property
    def empirical_p_value(self) -> float:
        """Calculate empirical p-value from null distribution."""
        return (np.sum(self.null_distribution >= self.observed_metric) + 1) / (self.n_permutations + 1)

    @property
    def z_score(self) -> float:
        """Z-score of observed metric against null distribution."""
        null_mean = np.mean(self.null_distribution)
        null_std = np.std(self.null_distribution)
        if null_std > 0:
            return (self.observed_metric - null_mean) / null_std
        return 0.0


@dataclass
class NegativeControlReport:
    """Comprehensive negative control analysis report."""
    permutation_results: List[PermutationResult]
    random_baseline_results: Dict[str, float]
    passes_negative_control: bool
    confidence_level: float
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NegativeControlConfig:
    """Configuration for negative control analysis."""
    n_permutations: int = 1000
    significance_threshold: float = 0.05
    random_state: Optional[int] = 42
    metrics: List[str] = field(default_factory=lambda: ["silhouette", "stability"])


class NegativeControlRunner:
    """
    Validates that pipeline doesn't find structure in null data.

    Implements negative controls as mandatory CI from research integrity
    guidelines to ensure that clustering findings are not artifacts.
    """

    def __init__(self, config: Optional[NegativeControlConfig] = None):
        """
        Initialize negative control runner.

        Args:
            config: Configuration for negative control parameters
        """
        self.config = config or NegativeControlConfig()
        self._rng = np.random.default_rng(self.config.random_state)

    def permutation_test(
        self,
        data: np.ndarray,
        clusterer: SubtypeClusterer,
        observed_result: Optional[ClusteringResult] = None,
    ) -> PermutationResult:
        """
        Test if clustering structure is significant via permutation.

        Shuffles sample labels and re-clusters to build null distribution.

        Args:
            data: Feature matrix (n_samples, n_features)
            clusterer: Configured SubtypeClusterer instance
            observed_result: Pre-computed clustering result (optional)

        Returns:
            PermutationResult with significance assessment
        """
        n_samples, n_features = data.shape

        # Create mock object for clusterer interface
        class MockPathwayScores:
            def __init__(self, scores, samples, pathways):
                self.scores = scores
                self.samples = samples
                self.pathways = pathways

        sample_ids = [f"sample_{i}" for i in range(n_samples)]
        pathway_ids = [f"pathway_{i}" for i in range(n_features)]

        # Get observed clustering if not provided
        if observed_result is None:
            mock_scores = MockPathwayScores(data, sample_ids, pathway_ids)
            observed_result = clusterer.fit(mock_scores)

        # Get observed metric (silhouette score)
        observed_metric = getattr(observed_result, 'silhouette_score', 0.0)
        if observed_metric is None:
            observed_metric = self._compute_silhouette(data, observed_result.labels)

        # Build null distribution
        null_metrics = []
        for i in range(self.config.n_permutations):
            # Permute features independently for each sample
            permuted_data = data.copy()
            for j in range(n_samples):
                self._rng.shuffle(permuted_data[j, :])

            # Cluster permuted data
            try:
                mock_permuted = MockPathwayScores(permuted_data, sample_ids, pathway_ids)
                perm_result = clusterer.fit(mock_permuted)
                perm_metric = getattr(perm_result, 'silhouette_score', None)
                if perm_metric is None:
                    perm_metric = self._compute_silhouette(permuted_data, perm_result.labels)
                null_metrics.append(perm_metric)
            except Exception as e:
                logger.warning(f"Permutation {i} failed: {e}")
                null_metrics.append(0.0)

        null_distribution = np.array(null_metrics)

        # Calculate p-value
        p_value = (np.sum(null_distribution >= observed_metric) + 1) / (len(null_metrics) + 1)

        return PermutationResult(
            observed_metric=observed_metric,
            null_distribution=null_distribution,
            p_value=p_value,
            metric_name="silhouette_score",
            n_permutations=self.config.n_permutations,
            is_significant=p_value < self.config.significance_threshold,
        )

    def random_geneset_baseline(
        self,
        data: np.ndarray,
        clusterer: SubtypeClusterer,
        n_random_sets: int = 100,
        set_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compare clustering to random gene set baselines.

        Tests whether pathway-based features outperform random feature subsets.

        Args:
            data: Feature matrix (n_samples, n_features)
            clusterer: Configured SubtypeClusterer instance
            n_random_sets: Number of random feature sets to test
            set_size: Size of random feature sets (defaults to all features)

        Returns:
            Dictionary with baseline comparison results
        """
        n_samples, n_features = data.shape
        set_size = set_size or n_features

        # Create mock object for clusterer interface
        class MockPathwayScores:
            def __init__(self, scores, samples, pathways):
                self.scores = scores
                self.samples = samples
                self.pathways = pathways

        sample_ids = [f"sample_{i}" for i in range(n_samples)]
        pathway_ids = [f"pathway_{i}" for i in range(n_features)]

        # Get observed clustering
        mock_scores = MockPathwayScores(data, sample_ids, pathway_ids)
        observed_result = clusterer.fit(mock_scores)
        observed_metric = getattr(observed_result, 'silhouette_score', 0.0)
        if observed_metric is None:
            observed_metric = self._compute_silhouette(data, observed_result.labels)

        # Test random feature subsets
        random_metrics = []
        for i in range(n_random_sets):
            # Select random features
            random_features = self._rng.choice(
                n_features, size=min(set_size, n_features), replace=False
            )
            random_data = data[:, random_features]

            try:
                random_pathway_ids = [f"random_{j}" for j in range(len(random_features))]
                mock_random = MockPathwayScores(random_data, sample_ids, random_pathway_ids)
                random_result = clusterer.fit(mock_random)
                random_metric = getattr(random_result, 'silhouette_score', None)
                if random_metric is None:
                    random_metric = self._compute_silhouette(random_data, random_result.labels)
                random_metrics.append(random_metric)
            except Exception as e:
                logger.warning(f"Random baseline {i} failed: {e}")
                random_metrics.append(0.0)

        random_metrics = np.array(random_metrics)

        # Compute statistics
        p_value = (np.sum(random_metrics >= observed_metric) + 1) / (n_random_sets + 1)

        return {
            "observed_metric": observed_metric,
            "random_mean": np.mean(random_metrics),
            "random_std": np.std(random_metrics),
            "random_max": np.max(random_metrics),
            "p_value": p_value,
            "outperforms_random": observed_metric > np.mean(random_metrics),
            "significantly_better": p_value < self.config.significance_threshold,
            "n_random_sets": n_random_sets,
        }

    def label_shuffle_test(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        clusterer: SubtypeClusterer,
    ) -> PermutationResult:
        """
        Test if cluster assignments are better than random.

        Shuffles cluster labels and computes clustering metrics to
        establish null distribution.

        Args:
            data: Feature matrix (n_samples, n_features)
            labels: Original cluster labels
            clusterer: Configured SubtypeClusterer instance

        Returns:
            PermutationResult for label shuffle test
        """
        # Compute observed metric
        observed_metric = self._compute_silhouette(data, labels)

        # Build null distribution by shuffling labels
        null_metrics = []
        for i in range(self.config.n_permutations):
            shuffled_labels = labels.copy()
            self._rng.shuffle(shuffled_labels)
            null_metric = self._compute_silhouette(data, shuffled_labels)
            null_metrics.append(null_metric)

        null_distribution = np.array(null_metrics)
        p_value = (np.sum(null_distribution >= observed_metric) + 1) / (self.config.n_permutations + 1)

        return PermutationResult(
            observed_metric=observed_metric,
            null_distribution=null_distribution,
            p_value=p_value,
            metric_name="silhouette_score",
            n_permutations=self.config.n_permutations,
            is_significant=p_value < self.config.significance_threshold,
        )

    def run_full_negative_control(
        self,
        data: np.ndarray,
        clusterer: SubtypeClusterer,
    ) -> NegativeControlReport:
        """
        Run comprehensive negative control analysis.

        Args:
            data: Feature matrix (n_samples, n_features)
            clusterer: Configured SubtypeClusterer instance

        Returns:
            NegativeControlReport with all test results
        """
        n_samples, n_features = data.shape

        # Create mock object for clusterer interface
        class MockPathwayScores:
            def __init__(self, scores, samples, pathways):
                self.scores = scores
                self.samples = samples
                self.pathways = pathways

        sample_ids = [f"sample_{i}" for i in range(n_samples)]
        pathway_ids = [f"pathway_{i}" for i in range(n_features)]

        # Get observed clustering
        mock_scores = MockPathwayScores(data, sample_ids, pathway_ids)
        observed_result = clusterer.fit(mock_scores)

        # Run permutation test
        perm_result = self.permutation_test(data, clusterer, observed_result)

        # Run label shuffle test
        shuffle_result = self.label_shuffle_test(data, observed_result.labels, clusterer)

        # Run random baseline
        baseline_result = self.random_geneset_baseline(data, clusterer, n_random_sets=50)

        # Compile results
        permutation_results = [perm_result, shuffle_result]

        passes = (
            perm_result.is_significant and
            shuffle_result.is_significant and
            baseline_result["significantly_better"]
        )

        recommendations = []
        if not perm_result.is_significant:
            recommendations.append(
                "Clustering structure not significant vs. permuted null. "
                "Consider alternative feature selection or more samples."
            )
        if not shuffle_result.is_significant:
            recommendations.append(
                "Cluster assignments not better than random. "
                "Review clustering parameters or feature quality."
            )
        if not baseline_result["significantly_better"]:
            recommendations.append(
                "Pathway features don't outperform random gene sets. "
                "Consider revising pathway selection criteria."
            )

        if passes:
            recommendations.append(
                "All negative controls passed. Proceed with confidence."
            )

        return NegativeControlReport(
            permutation_results=permutation_results,
            random_baseline_results=baseline_result,
            passes_negative_control=passes,
            confidence_level=1 - max(r.p_value for r in permutation_results),
            recommendations=recommendations,
            metadata={
                "n_samples": n_samples,
                "n_features": n_features,
                "n_permutations": self.config.n_permutations,
            },
        )

    def _compute_silhouette(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score."""
        from sklearn.metrics import silhouette_score

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0

        try:
            return silhouette_score(data, labels)
        except Exception:
            return 0.0


# =============================================================================
# Provenance Tracking
# =============================================================================

@dataclass
class ProvenanceRecord:
    """
    Tracks versions for reproducibility.

    Implements version pinning and provenance tracking from research
    integrity guidelines to ensure results are reproducible and
    traceable to specific data/code versions.
    """
    # Reference and annotation versions
    reference_genome: str  # e.g., "GRCh38"
    annotation_version: str  # e.g., "GENCODE v38"

    # Database versions
    pathway_db_versions: Dict[str, str] = field(default_factory=dict)
    # e.g., {"GO": "2023-01", "KEGG": "release95", "Reactome": "v84"}

    # Gene ID mappings
    gene_id_mapping_version: str = ""  # e.g., "Ensembl 108"

    # Pipeline and code versions
    pipeline_version: str = ""
    framework_version: str = "0.1.0"

    # Software dependencies
    dependencies: Dict[str, str] = field(default_factory=dict)
    # e.g., {"numpy": "1.24.0", "scipy": "1.11.0", "sklearn": "1.3.0"}

    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)
    data_freeze_date: Optional[datetime] = None

    # Data sources
    cohort_name: str = ""
    sample_qc_version: str = ""
    variant_calling_pipeline: str = ""

    # Checksums for data integrity
    data_checksums: Dict[str, str] = field(default_factory=dict)

    # Notes and metadata
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "reference_genome": self.reference_genome,
            "annotation_version": self.annotation_version,
            "pathway_db_versions": self.pathway_db_versions,
            "gene_id_mapping_version": self.gene_id_mapping_version,
            "pipeline_version": self.pipeline_version,
            "framework_version": self.framework_version,
            "dependencies": self.dependencies,
            "timestamp": self.timestamp.isoformat(),
            "data_freeze_date": self.data_freeze_date.isoformat() if self.data_freeze_date else None,
            "cohort_name": self.cohort_name,
            "sample_qc_version": self.sample_qc_version,
            "variant_calling_pipeline": self.variant_calling_pipeline,
            "data_checksums": self.data_checksums,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceRecord":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        freeze_date = data.get("data_freeze_date")
        if isinstance(freeze_date, str):
            freeze_date = datetime.fromisoformat(freeze_date)

        return cls(
            reference_genome=data.get("reference_genome", ""),
            annotation_version=data.get("annotation_version", ""),
            pathway_db_versions=data.get("pathway_db_versions", {}),
            gene_id_mapping_version=data.get("gene_id_mapping_version", ""),
            pipeline_version=data.get("pipeline_version", ""),
            framework_version=data.get("framework_version", "0.1.0"),
            dependencies=data.get("dependencies", {}),
            timestamp=timestamp or datetime.now(),
            data_freeze_date=freeze_date,
            cohort_name=data.get("cohort_name", ""),
            sample_qc_version=data.get("sample_qc_version", ""),
            variant_calling_pipeline=data.get("variant_calling_pipeline", ""),
            data_checksums=data.get("data_checksums", {}),
            notes=data.get("notes", ""),
            metadata=data.get("metadata", {}),
        )

    def validate_compatibility(
        self,
        other: "ProvenanceRecord",
        strict: bool = False,
    ) -> Tuple[bool, List[str]]:
        """
        Check compatibility with another provenance record.

        Args:
            other: Another ProvenanceRecord to compare
            strict: If True, require exact matches on all fields

        Returns:
            Tuple of (is_compatible, list of incompatibility messages)
        """
        issues = []

        # Critical fields that must match
        if self.reference_genome != other.reference_genome:
            issues.append(
                f"Reference genome mismatch: {self.reference_genome} vs {other.reference_genome}"
            )

        if self.annotation_version != other.annotation_version:
            issues.append(
                f"Annotation version mismatch: {self.annotation_version} vs {other.annotation_version}"
            )

        # Check pathway database versions
        for db_name in set(self.pathway_db_versions) | set(other.pathway_db_versions):
            v1 = self.pathway_db_versions.get(db_name)
            v2 = other.pathway_db_versions.get(db_name)
            if v1 != v2:
                issues.append(
                    f"Pathway DB '{db_name}' version mismatch: {v1} vs {v2}"
                )

        # Check gene ID mapping
        if self.gene_id_mapping_version != other.gene_id_mapping_version:
            if strict or (self.gene_id_mapping_version and other.gene_id_mapping_version):
                issues.append(
                    f"Gene ID mapping mismatch: {self.gene_id_mapping_version} vs {other.gene_id_mapping_version}"
                )

        if strict:
            # Check dependencies in strict mode
            for pkg in set(self.dependencies) | set(other.dependencies):
                v1 = self.dependencies.get(pkg)
                v2 = other.dependencies.get(pkg)
                if v1 != v2:
                    issues.append(
                        f"Dependency '{pkg}' version mismatch: {v1} vs {v2}"
                    )

        is_compatible = len(issues) == 0
        return is_compatible, issues

    def get_summary(self) -> str:
        """Get human-readable summary of provenance."""
        lines = [
            "Provenance Record",
            "=================",
            f"Reference: {self.reference_genome} ({self.annotation_version})",
            f"Timestamp: {self.timestamp.isoformat()}",
        ]

        if self.cohort_name:
            lines.append(f"Cohort: {self.cohort_name}")

        if self.pathway_db_versions:
            lines.append("Pathway Databases:")
            for db, ver in self.pathway_db_versions.items():
                lines.append(f"  - {db}: {ver}")

        if self.pipeline_version:
            lines.append(f"Pipeline: {self.pipeline_version}")

        lines.append(f"Framework: {self.framework_version}")

        return "\n".join(lines)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save provenance record to JSON file."""
        import json
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved provenance record to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ProvenanceRecord":
        """Load provenance record from JSON file."""
        import json
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def create_current(
        cls,
        reference_genome: str = "GRCh38",
        annotation_version: str = "GENCODE v38",
    ) -> "ProvenanceRecord":
        """
        Create provenance record with current environment info.

        Args:
            reference_genome: Reference genome version
            annotation_version: Gene annotation version

        Returns:
            ProvenanceRecord with current dependency versions
        """
        import numpy as np
        import scipy
        import sklearn

        dependencies = {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        }

        try:
            import pandas as pd
            dependencies["pandas"] = pd.__version__
        except ImportError:
            pass

        return cls(
            reference_genome=reference_genome,
            annotation_version=annotation_version,
            dependencies=dependencies,
            timestamp=datetime.now(),
        )
