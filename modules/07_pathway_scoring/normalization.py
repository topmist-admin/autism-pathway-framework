"""
Pathway Score Normalization

Normalizes pathway scores to enable cross-sample and cross-pathway comparisons.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
from scipy import stats

import sys
from pathlib import Path
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from aggregation import PathwayScoreMatrix

logger = logging.getLogger(__name__)


class NormalizationMethod(Enum):
    """Methods for normalizing pathway scores."""

    ZSCORE = "zscore"  # Z-score normalization (per pathway)
    ZSCORE_ROBUST = "zscore_robust"  # Robust z-score using median and MAD
    MINMAX = "minmax"  # Min-max scaling to [0, 1]
    RANK = "rank"  # Rank-based normalization
    PERCENTILE = "percentile"  # Percentile ranks
    QUANTILE = "quantile"  # Quantile normalization (across pathways)
    LOG = "log"  # Log transformation
    SAMPLE_ZSCORE = "sample_zscore"  # Z-score across pathways per sample


@dataclass
class NormalizationConfig:
    """Configuration for pathway score normalization."""

    method: NormalizationMethod = NormalizationMethod.ZSCORE

    # Z-score parameters
    center: bool = True  # Subtract mean
    scale: bool = True  # Divide by std

    # Robust normalization
    use_median: bool = False  # Use median instead of mean
    use_mad: bool = False  # Use MAD instead of std

    # Log transform parameters
    log_base: float = 2.0  # Log base for LOG method
    pseudocount: float = 1.0  # Pseudocount for log(x + pseudocount)

    # Handling zeros and outliers
    handle_zeros: str = "keep"  # "keep", "nan", "small"
    clip_outliers: bool = False  # Clip extreme values
    outlier_std: float = 3.0  # Number of stds for outlier detection


class PathwayScoreNormalizer:
    """
    Normalizes pathway scores for downstream analysis.

    Supports multiple normalization strategies for different analysis needs.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Initialize normalizer.

        Args:
            config: Normalization configuration
        """
        self.config = config or NormalizationConfig()

    def normalize(
        self,
        scores: PathwayScoreMatrix,
        method: Optional[NormalizationMethod] = None,
    ) -> PathwayScoreMatrix:
        """
        Normalize pathway scores.

        Args:
            scores: PathwayScoreMatrix to normalize
            method: Optional override for normalization method

        Returns:
            Normalized PathwayScoreMatrix
        """
        method = method or self.config.method

        if method == NormalizationMethod.ZSCORE:
            normalized = self._zscore_normalize(scores.scores)
        elif method == NormalizationMethod.ZSCORE_ROBUST:
            normalized = self._robust_zscore_normalize(scores.scores)
        elif method == NormalizationMethod.MINMAX:
            normalized = self._minmax_normalize(scores.scores)
        elif method == NormalizationMethod.RANK:
            normalized = self._rank_normalize(scores.scores)
        elif method == NormalizationMethod.PERCENTILE:
            normalized = self._percentile_normalize(scores.scores)
        elif method == NormalizationMethod.QUANTILE:
            normalized = self._quantile_normalize(scores.scores)
        elif method == NormalizationMethod.LOG:
            normalized = self._log_normalize(scores.scores)
        elif method == NormalizationMethod.SAMPLE_ZSCORE:
            normalized = self._sample_zscore_normalize(scores.scores)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Handle outliers if configured
        if self.config.clip_outliers:
            normalized = self._clip_outliers(normalized)

        return PathwayScoreMatrix(
            samples=scores.samples.copy(),
            pathways=scores.pathways.copy(),
            scores=normalized,
            pathway_names=scores.pathway_names.copy(),
            contributing_genes=scores.contributing_genes.copy(),
            metadata={
                **scores.metadata,
                "normalized": True,
                "normalization_method": method.value,
            },
        )

    def _zscore_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Z-score normalize per pathway (column-wise).

        z = (x - mean) / std
        """
        normalized = np.zeros_like(scores, dtype=float)

        for j in range(scores.shape[1]):
            col = scores[:, j]

            if self.config.center:
                if self.config.use_median:
                    center = np.median(col)
                else:
                    center = np.mean(col)
            else:
                center = 0.0

            if self.config.scale:
                if self.config.use_mad:
                    scale = stats.median_abs_deviation(col)
                    scale = scale * 1.4826  # Scale to match std for normal data
                else:
                    scale = np.std(col)
                scale = max(scale, 1e-10)  # Avoid division by zero
            else:
                scale = 1.0

            normalized[:, j] = (col - center) / scale

        return normalized

    def _robust_zscore_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Robust z-score using median and MAD.

        z = (x - median) / MAD
        """
        normalized = np.zeros_like(scores, dtype=float)

        for j in range(scores.shape[1]):
            col = scores[:, j]
            median = np.median(col)
            mad = stats.median_abs_deviation(col)
            mad = max(mad * 1.4826, 1e-10)  # Scale MAD and avoid zero
            normalized[:, j] = (col - median) / mad

        return normalized

    def _minmax_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Min-max normalization to [0, 1] range.

        x' = (x - min) / (max - min)
        """
        normalized = np.zeros_like(scores, dtype=float)

        for j in range(scores.shape[1]):
            col = scores[:, j]
            min_val = np.min(col)
            max_val = np.max(col)
            range_val = max_val - min_val

            if range_val > 0:
                normalized[:, j] = (col - min_val) / range_val
            else:
                normalized[:, j] = 0.0

        return normalized

    def _rank_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Rank-based normalization.

        Converts scores to ranks, then scales to [0, 1].
        """
        normalized = np.zeros_like(scores, dtype=float)
        n_samples = scores.shape[0]

        for j in range(scores.shape[1]):
            col = scores[:, j]
            # Average rank for ties
            ranks = stats.rankdata(col, method="average")
            # Scale to [0, 1]
            normalized[:, j] = (ranks - 1) / max(n_samples - 1, 1)

        return normalized

    def _percentile_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Percentile rank normalization.

        Returns the percentile rank of each value.
        """
        normalized = np.zeros_like(scores, dtype=float)

        for j in range(scores.shape[1]):
            col = scores[:, j]
            for i in range(len(col)):
                # Percentile = fraction of values <= current value
                normalized[i, j] = np.sum(col <= col[i]) / len(col) * 100

        return normalized

    def _quantile_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Quantile normalization across all pathways.

        Makes the distribution of scores the same across all pathways.
        """
        # Sort each column
        sorted_scores = np.sort(scores, axis=0)

        # Calculate mean across rows for each rank position
        row_means = np.mean(sorted_scores, axis=1)

        # Replace each value with the row mean for its rank
        normalized = np.zeros_like(scores, dtype=float)

        for j in range(scores.shape[1]):
            col = scores[:, j]
            # Get rank of each value (0-indexed)
            ranks = stats.rankdata(col, method="ordinal") - 1
            # Replace with row mean
            normalized[:, j] = row_means[ranks.astype(int)]

        return normalized

    def _log_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Log transformation.

        x' = log(x + pseudocount)
        """
        base = self.config.log_base
        pseudocount = self.config.pseudocount

        # Handle negative values
        min_val = np.min(scores)
        if min_val < 0:
            # Shift to make all values positive
            shifted = scores - min_val + pseudocount
        else:
            shifted = scores + pseudocount

        if base == np.e:
            normalized = np.log(shifted)
        elif base == 2:
            normalized = np.log2(shifted)
        elif base == 10:
            normalized = np.log10(shifted)
        else:
            normalized = np.log(shifted) / np.log(base)

        return normalized

    def _sample_zscore_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Z-score normalize per sample (row-wise).

        Normalizes each sample's pathway scores to have mean 0 and std 1.
        """
        normalized = np.zeros_like(scores, dtype=float)

        for i in range(scores.shape[0]):
            row = scores[i, :]
            mean = np.mean(row)
            std = np.std(row)
            std = max(std, 1e-10)
            normalized[i, :] = (row - mean) / std

        return normalized

    def _clip_outliers(self, scores: np.ndarray) -> np.ndarray:
        """
        Clip outlier values.

        Values beyond outlier_std standard deviations from mean are clipped.
        """
        clipped = scores.copy()
        n_std = self.config.outlier_std

        for j in range(scores.shape[1]):
            col = clipped[:, j]
            mean = np.mean(col)
            std = np.std(col)

            lower = mean - n_std * std
            upper = mean + n_std * std

            clipped[:, j] = np.clip(col, lower, upper)

        return clipped

    def fit_transform(
        self,
        scores: PathwayScoreMatrix,
        reference_scores: Optional[PathwayScoreMatrix] = None,
    ) -> PathwayScoreMatrix:
        """
        Fit normalization parameters and transform scores.

        If reference_scores is provided, uses those to compute normalization
        parameters (e.g., for normalizing test data using training statistics).

        Args:
            scores: PathwayScoreMatrix to normalize
            reference_scores: Optional reference for computing parameters

        Returns:
            Normalized PathwayScoreMatrix
        """
        if reference_scores is None:
            return self.normalize(scores)

        # Compute statistics from reference
        ref = reference_scores.scores
        method = self.config.method

        if method == NormalizationMethod.ZSCORE:
            means = np.mean(ref, axis=0)
            stds = np.std(ref, axis=0)
            stds[stds == 0] = 1.0

            normalized = (scores.scores - means) / stds

        elif method == NormalizationMethod.MINMAX:
            mins = np.min(ref, axis=0)
            maxs = np.max(ref, axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0

            normalized = (scores.scores - mins) / ranges
            normalized = np.clip(normalized, 0, 1)

        else:
            # For other methods, just use standard normalize
            return self.normalize(scores)

        return PathwayScoreMatrix(
            samples=scores.samples.copy(),
            pathways=scores.pathways.copy(),
            scores=normalized,
            pathway_names=scores.pathway_names.copy(),
            contributing_genes=scores.contributing_genes.copy(),
            metadata={
                **scores.metadata,
                "normalized": True,
                "normalization_method": method.value,
                "reference_based": True,
            },
        )

    def compute_significance(
        self,
        scores: PathwayScoreMatrix,
        method: str = "zscore",
    ) -> PathwayScoreMatrix:
        """
        Compute significance (p-values) for pathway scores.

        Args:
            scores: Normalized PathwayScoreMatrix
            method: Method for computing p-values ("zscore", "permutation")

        Returns:
            PathwayScoreMatrix with -log10(p-values) as scores
        """
        if method == "zscore":
            # Assume z-score normalized, compute two-tailed p-value
            pvals = 2 * (1 - stats.norm.cdf(np.abs(scores.scores)))
        else:
            raise ValueError(f"Unknown significance method: {method}")

        # Convert to -log10(p)
        pvals[pvals == 0] = 1e-300  # Avoid log(0)
        neg_log_p = -np.log10(pvals)

        return PathwayScoreMatrix(
            samples=scores.samples.copy(),
            pathways=scores.pathways.copy(),
            scores=neg_log_p,
            pathway_names=scores.pathway_names.copy(),
            metadata={
                **scores.metadata,
                "score_type": "neg_log10_pval",
            },
        )

    @staticmethod
    def combine_normalization_methods(
        scores: PathwayScoreMatrix,
        methods: List[NormalizationMethod],
    ) -> Dict[str, PathwayScoreMatrix]:
        """
        Apply multiple normalization methods for comparison.

        Args:
            scores: PathwayScoreMatrix to normalize
            methods: List of normalization methods to apply

        Returns:
            Dictionary mapping method name to normalized PathwayScoreMatrix
        """
        results = {}
        normalizer = PathwayScoreNormalizer()

        for method in methods:
            normalizer.config.method = method
            results[method.value] = normalizer.normalize(scores)

        return results
