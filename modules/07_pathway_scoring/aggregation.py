"""
Pathway Aggregation

Aggregates gene-level burden scores to pathway-level scores using various methods.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Methods for aggregating gene scores to pathway scores."""

    WEIGHTED_SUM = "weighted_sum"  # Sum of gene scores weighted by pathway membership
    MEAN = "mean"  # Mean of gene scores in pathway
    MAX = "max"  # Maximum gene score in pathway
    SUM = "sum"  # Simple sum of gene scores
    MEDIAN = "median"  # Median of gene scores in pathway
    TOP_K_MEAN = "top_k_mean"  # Mean of top K gene scores
    SQRT_SUM = "sqrt_sum"  # Square root of sum (reduces impact of many small contributions)


@dataclass
class AggregationConfig:
    """Configuration for pathway score aggregation."""

    method: AggregationMethod = AggregationMethod.WEIGHTED_SUM

    # Pathway size constraints
    min_pathway_size: int = 5  # Minimum genes in pathway to score
    max_pathway_size: int = 500  # Maximum genes in pathway to score

    # Gene weighting options
    weight_by_gene_coverage: bool = True  # Weight by fraction of pathway genes with burden
    normalize_by_pathway_size: bool = True  # Normalize score by pathway size

    # Top-K parameters (for TOP_K_MEAN method)
    top_k: int = 10  # Number of top genes to consider

    # Minimum gene overlap
    min_genes_hit: int = 1  # Minimum genes with burden to score pathway

    # Gene constraint weighting
    use_constraint_weights: bool = False  # Weight genes by pLI/LOEUF scores

    # Sparse handling
    treat_missing_as_zero: bool = True  # Treat genes not in burden matrix as zero


@dataclass
class PathwayScoreMatrix:
    """
    Pathway scores for all samples.

    Attributes:
        samples: List of sample IDs
        pathways: List of pathway IDs
        scores: 2D array of shape (n_samples, n_pathways)
        pathway_names: Mapping from pathway ID to human-readable name
        contributing_genes: Dict mapping (sample, pathway) to genes that contributed
        sample_index: Mapping from sample ID to index
        pathway_index: Mapping from pathway ID to index
        metadata: Additional metadata
    """

    samples: List[str]
    pathways: List[str]
    scores: np.ndarray  # shape: (n_samples, n_pathways)
    pathway_names: Dict[str, str] = field(default_factory=dict)
    contributing_genes: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)
    sample_index: Dict[str, int] = field(default_factory=dict)
    pathway_index: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Build index mappings if not provided."""
        if not self.sample_index:
            self.sample_index = {s: i for i, s in enumerate(self.samples)}
        if not self.pathway_index:
            self.pathway_index = {p: i for i, p in enumerate(self.pathways)}

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_pathways(self) -> int:
        return len(self.pathways)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.scores.shape

    def get_sample(self, sample_id: str) -> Dict[str, float]:
        """
        Get pathway scores for a specific sample.

        Args:
            sample_id: Sample identifier

        Returns:
            Dictionary mapping pathway to score
        """
        if sample_id not in self.sample_index:
            raise KeyError(f"Sample not found: {sample_id}")

        sample_idx = self.sample_index[sample_id]
        return {
            pathway: float(self.scores[sample_idx, pathway_idx])
            for pathway, pathway_idx in self.pathway_index.items()
            if self.scores[sample_idx, pathway_idx] != 0
        }

    def get_pathway(self, pathway_id: str) -> np.ndarray:
        """
        Get scores for a pathway across all samples.

        Args:
            pathway_id: Pathway identifier

        Returns:
            Array of scores
        """
        if pathway_id not in self.pathway_index:
            raise KeyError(f"Pathway not found: {pathway_id}")

        pathway_idx = self.pathway_index[pathway_id]
        return self.scores[:, pathway_idx]

    def get_score(self, sample_id: str, pathway_id: str) -> float:
        """Get score for specific sample and pathway."""
        if sample_id not in self.sample_index:
            return 0.0
        if pathway_id not in self.pathway_index:
            return 0.0

        sample_idx = self.sample_index[sample_id]
        pathway_idx = self.pathway_index[pathway_id]
        return float(self.scores[sample_idx, pathway_idx])

    def get_contributing_genes(
        self, sample_id: str, pathway_id: str
    ) -> List[str]:
        """Get genes that contributed to a sample's pathway score."""
        return self.contributing_genes.get((sample_id, pathway_id), [])

    def get_top_pathways(
        self,
        sample_id: str,
        n: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Get top scoring pathways for a sample.

        Args:
            sample_id: Sample identifier
            n: Number of top pathways to return

        Returns:
            List of (pathway_id, score) tuples, sorted by score descending
        """
        if sample_id not in self.sample_index:
            return []

        sample_scores = self.get_sample(sample_id)
        sorted_pathways = sorted(
            sample_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_pathways[:n]

    def to_dataframe(self):
        """
        Convert to pandas DataFrame.

        Returns:
            DataFrame with samples as rows, pathways as columns
        """
        import pandas as pd
        return pd.DataFrame(
            self.scores,
            index=self.samples,
            columns=self.pathways,
        )

    def filter_pathways(
        self,
        min_score: float = 0.0,
        min_samples_hit: int = 1,
    ) -> "PathwayScoreMatrix":
        """
        Filter to pathways meeting criteria.

        Args:
            min_score: Minimum score in any sample
            min_samples_hit: Minimum samples with nonzero score

        Returns:
            New PathwayScoreMatrix with filtered pathways
        """
        keep_indices = []
        for i, pathway in enumerate(self.pathways):
            col = self.scores[:, i]
            if np.max(col) >= min_score and np.sum(col > 0) >= min_samples_hit:
                keep_indices.append(i)

        keep_pathways = [self.pathways[i] for i in keep_indices]

        return PathwayScoreMatrix(
            samples=self.samples.copy(),
            pathways=keep_pathways,
            scores=self.scores[:, keep_indices].copy(),
            pathway_names={p: self.pathway_names.get(p, p) for p in keep_pathways},
            metadata={**self.metadata, "filtered": True},
        )

    def copy(self) -> "PathwayScoreMatrix":
        """Create a deep copy."""
        return PathwayScoreMatrix(
            samples=self.samples.copy(),
            pathways=self.pathways.copy(),
            scores=self.scores.copy(),
            pathway_names=self.pathway_names.copy(),
            contributing_genes=self.contributing_genes.copy(),
            metadata=self.metadata.copy(),
        )


class PathwayAggregator:
    """
    Aggregates gene-level burden scores to pathway-level scores.

    Supports multiple aggregation methods and optional gene weighting.
    """

    def __init__(self, config: Optional[AggregationConfig] = None):
        """
        Initialize pathway aggregator.

        Args:
            config: Aggregation configuration
        """
        self.config = config or AggregationConfig()

    def aggregate(
        self,
        gene_burdens: Any,  # GeneBurdenMatrix from Module 02
        pathway_db: Any,  # PathwayDatabase from Module 01
        gene_weights: Optional[Dict[str, float]] = None,
    ) -> PathwayScoreMatrix:
        """
        Aggregate gene burden scores to pathway scores.

        Args:
            gene_burdens: GeneBurdenMatrix with gene-level scores
            pathway_db: PathwayDatabase with pathway definitions
            gene_weights: Optional per-gene weights (e.g., pLI scores)

        Returns:
            PathwayScoreMatrix with pathway-level scores
        """
        # Filter pathways by size
        filtered_db = pathway_db.filter_by_size(
            min_size=self.config.min_pathway_size,
            max_size=self.config.max_pathway_size,
        )

        # Get available genes from burden matrix
        burden_genes = set(gene_burdens.genes)

        # Build pathway list (only pathways with gene overlap)
        pathways = []
        pathway_gene_sets = {}

        for pathway_id, pathway_genes in filtered_db.pathways.items():
            # Find overlap with genes in burden matrix
            overlap = pathway_genes & burden_genes
            if len(overlap) >= self.config.min_genes_hit:
                pathways.append(pathway_id)
                pathway_gene_sets[pathway_id] = overlap

        logger.info(
            f"Scoring {len(pathways)} pathways "
            f"(filtered from {len(filtered_db.pathways)} by gene overlap)"
        )

        # Initialize score matrix
        n_samples = gene_burdens.n_samples
        n_pathways = len(pathways)
        scores = np.zeros((n_samples, n_pathways))

        # Track contributing genes
        contributing_genes: Dict[Tuple[str, str], List[str]] = {}

        # Compute scores for each sample and pathway
        for p_idx, pathway_id in enumerate(pathways):
            pathway_genes = pathway_gene_sets[pathway_id]
            pathway_size = len(filtered_db.pathways[pathway_id])

            for s_idx, sample_id in enumerate(gene_burdens.samples):
                # Get gene scores for this sample
                gene_scores = []
                contributing = []

                for gene in pathway_genes:
                    score = gene_burdens.get_score(sample_id, gene)

                    # Apply gene weight if provided
                    if gene_weights and gene in gene_weights:
                        score *= gene_weights[gene]

                    if score > 0:
                        gene_scores.append(score)
                        contributing.append(gene)

                if not gene_scores:
                    continue

                # Aggregate scores using configured method
                pathway_score = self._aggregate_scores(
                    gene_scores,
                    pathway_size=pathway_size,
                    n_genes_with_burden=len(contributing),
                )

                scores[s_idx, p_idx] = pathway_score

                if contributing:
                    contributing_genes[(sample_id, pathway_id)] = contributing

        # Get pathway names
        pathway_names = {
            p: filtered_db.pathway_names.get(p, p)
            for p in pathways
        }

        logger.info(
            f"Computed pathway scores: {n_samples} samples × {n_pathways} pathways"
        )

        return PathwayScoreMatrix(
            samples=gene_burdens.samples.copy(),
            pathways=pathways,
            scores=scores,
            pathway_names=pathway_names,
            contributing_genes=contributing_genes,
            metadata={
                "method": self.config.method.value,
                "config": str(self.config),
                "source_db": pathway_db.source,
            },
        )

    def _aggregate_scores(
        self,
        gene_scores: List[float],
        pathway_size: int,
        n_genes_with_burden: int,
    ) -> float:
        """
        Aggregate gene scores using configured method.

        Args:
            gene_scores: List of gene burden scores
            pathway_size: Total size of pathway
            n_genes_with_burden: Number of genes with burden

        Returns:
            Aggregated pathway score
        """
        if not gene_scores:
            return 0.0

        arr = np.array(gene_scores)
        method = self.config.method

        if method == AggregationMethod.WEIGHTED_SUM:
            score = np.sum(arr)
        elif method == AggregationMethod.MEAN:
            score = np.mean(arr)
        elif method == AggregationMethod.MAX:
            score = np.max(arr)
        elif method == AggregationMethod.SUM:
            score = np.sum(arr)
        elif method == AggregationMethod.MEDIAN:
            score = np.median(arr)
        elif method == AggregationMethod.TOP_K_MEAN:
            k = min(self.config.top_k, len(arr))
            top_k = np.sort(arr)[-k:]
            score = np.mean(top_k)
        elif method == AggregationMethod.SQRT_SUM:
            score = np.sqrt(np.sum(arr))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # Apply optional normalization
        if self.config.normalize_by_pathway_size and pathway_size > 0:
            if method in (AggregationMethod.SUM, AggregationMethod.WEIGHTED_SUM):
                # Normalize by pathway size for sum-based methods
                score = score / np.sqrt(pathway_size)

        # Apply coverage weighting
        if self.config.weight_by_gene_coverage and pathway_size > 0:
            coverage = n_genes_with_burden / pathway_size
            score = score * np.sqrt(coverage)

        return float(score)

    def aggregate_with_multiple_methods(
        self,
        gene_burdens: Any,
        pathway_db: Any,
        methods: Optional[List[AggregationMethod]] = None,
    ) -> Dict[str, PathwayScoreMatrix]:
        """
        Aggregate using multiple methods for comparison.

        Args:
            gene_burdens: GeneBurdenMatrix
            pathway_db: PathwayDatabase
            methods: List of methods to use (default: all)

        Returns:
            Dictionary mapping method name to PathwayScoreMatrix
        """
        if methods is None:
            methods = list(AggregationMethod)

        results = {}
        original_method = self.config.method

        for method in methods:
            self.config.method = method
            results[method.value] = self.aggregate(gene_burdens, pathway_db)

        # Restore original method
        self.config.method = original_method

        return results

    def aggregate_by_burden_type(
        self,
        burden_matrices: Dict[str, Any],  # Dict of burden type -> GeneBurdenMatrix
        pathway_db: Any,
        combine_method: str = "sum",
    ) -> PathwayScoreMatrix:
        """
        Aggregate multiple burden types (e.g., LoF + missense) into pathway scores.

        Args:
            burden_matrices: Dictionary mapping burden type to GeneBurdenMatrix
            pathway_db: PathwayDatabase
            combine_method: How to combine different burden types ("sum", "max", "weighted")

        Returns:
            Combined PathwayScoreMatrix
        """
        if not burden_matrices:
            raise ValueError("No burden matrices provided")

        # Aggregate each burden type
        pathway_scores = {}
        for burden_type, burden_matrix in burden_matrices.items():
            pathway_scores[burden_type] = self.aggregate(burden_matrix, pathway_db)

        # Get common structure
        first_scores = list(pathway_scores.values())[0]
        samples = first_scores.samples
        pathways = first_scores.pathways

        # Combine scores
        combined = np.zeros((len(samples), len(pathways)))

        for burden_type, ps in pathway_scores.items():
            if combine_method == "sum":
                combined += ps.scores
            elif combine_method == "max":
                combined = np.maximum(combined, ps.scores)
            elif combine_method == "weighted":
                # Weight LoF higher than missense
                weight = 1.5 if "lof" in burden_type.lower() else 1.0
                combined += weight * ps.scores

        return PathwayScoreMatrix(
            samples=samples,
            pathways=pathways,
            scores=combined,
            pathway_names=first_scores.pathway_names,
            metadata={
                "combined_from": list(burden_matrices.keys()),
                "combine_method": combine_method,
            },
        )
