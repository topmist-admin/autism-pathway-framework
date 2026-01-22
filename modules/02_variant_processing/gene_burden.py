"""
Gene Burden Calculator

Aggregates variant-level data to gene-level burden scores.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import logging

import numpy as np

# Import from local module
import sys
from pathlib import Path
_module_root = Path(__file__).parent
sys.path.insert(0, str(_module_root))

try:
    from .annotation import (
        AnnotatedVariant,
        VariantConsequence,
        ImpactLevel,
        LOF_CONSEQUENCES,
    )
except ImportError:
    from annotation import (
        AnnotatedVariant,
        VariantConsequence,
        ImpactLevel,
        LOF_CONSEQUENCES,
    )

logger = logging.getLogger(__name__)


@dataclass
class WeightConfig:
    """Configuration for variant weighting in burden calculation."""

    # Consequence-based weights
    consequence_weights: Dict[str, float] = field(default_factory=lambda: {
        # Loss-of-function (highest weight)
        "frameshift_variant": 1.0,
        "stop_gained": 1.0,
        "splice_acceptor_variant": 1.0,
        "splice_donor_variant": 1.0,
        "start_lost": 1.0,

        # Moderate impact
        "missense_variant": 0.5,
        "inframe_insertion": 0.3,
        "inframe_deletion": 0.3,
        "protein_altering_variant": 0.5,

        # Low impact
        "splice_region_variant": 0.2,
        "synonymous_variant": 0.0,

        # Modifier (usually excluded)
        "intron_variant": 0.0,
        "5_prime_UTR_variant": 0.1,
        "3_prime_UTR_variant": 0.1,
    })

    # CADD score thresholds and weighting
    use_cadd_weighting: bool = True
    cadd_threshold: float = 20.0  # Minimum CADD phred to include
    cadd_weight_scale: float = 0.05  # Weight = CADD * scale

    # REVEL score threshold for missense
    use_revel_weighting: bool = True
    revel_threshold: float = 0.5

    # Allele frequency weighting
    use_af_weighting: bool = False
    af_weight_beta: float = 1.0  # Weight = (1 - AF)^beta

    # Filter settings
    include_synonymous: bool = False
    include_modifier: bool = False
    min_impact: str = "MODERATE"  # MODIFIER, LOW, MODERATE, HIGH

    # Aggregation method
    aggregation: str = "weighted_sum"  # weighted_sum, max, count


@dataclass
class GeneBurdenMatrix:
    """
    Gene burden scores for all samples.

    Attributes:
        samples: List of sample IDs
        genes: List of gene IDs
        scores: 2D array of shape (n_samples, n_genes)
        sample_index: Mapping from sample ID to index
        gene_index: Mapping from gene ID to index
        contributing_variants: Dict mapping (sample, gene) to variants that contributed
    """

    samples: List[str]
    genes: List[str]
    scores: np.ndarray  # shape: (n_samples, n_genes)
    sample_index: Dict[str, int] = field(default_factory=dict)
    gene_index: Dict[str, int] = field(default_factory=dict)
    contributing_variants: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Build index mappings if not provided."""
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

    @property
    def shape(self) -> Tuple[int, int]:
        return self.scores.shape

    def get_sample(self, sample_id: str) -> Dict[str, float]:
        """
        Get burden scores for a specific sample.

        Args:
            sample_id: Sample identifier

        Returns:
            Dictionary mapping gene to burden score
        """
        if sample_id not in self.sample_index:
            raise KeyError(f"Sample not found: {sample_id}")

        sample_idx = self.sample_index[sample_id]
        return {
            gene: self.scores[sample_idx, gene_idx]
            for gene, gene_idx in self.gene_index.items()
            if self.scores[sample_idx, gene_idx] > 0
        }

    def get_gene(self, gene_id: str) -> np.ndarray:
        """
        Get burden scores for a gene across all samples.

        Args:
            gene_id: Gene identifier

        Returns:
            Array of burden scores
        """
        if gene_id not in self.gene_index:
            raise KeyError(f"Gene not found: {gene_id}")

        gene_idx = self.gene_index[gene_id]
        return self.scores[:, gene_idx]

    def get_score(self, sample_id: str, gene_id: str) -> float:
        """Get burden score for specific sample and gene."""
        if sample_id not in self.sample_index:
            return 0.0
        if gene_id not in self.gene_index:
            return 0.0

        sample_idx = self.sample_index[sample_id]
        gene_idx = self.gene_index[gene_id]
        return float(self.scores[sample_idx, gene_idx])

    def to_sparse(self):
        """
        Convert to sparse matrix format.

        Returns:
            scipy.sparse.csr_matrix
        """
        from scipy.sparse import csr_matrix
        return csr_matrix(self.scores)

    def to_dataframe(self):
        """
        Convert to pandas DataFrame.

        Returns:
            DataFrame with samples as rows, genes as columns
        """
        import pandas as pd
        return pd.DataFrame(
            self.scores,
            index=self.samples,
            columns=self.genes,
        )

    def get_nonzero_genes(self) -> Set[str]:
        """Get genes with nonzero burden in at least one sample."""
        gene_sums = np.sum(self.scores, axis=0)
        nonzero_idx = np.where(gene_sums > 0)[0]
        return {self.genes[i] for i in nonzero_idx}

    def filter_genes(self, genes: Set[str]) -> "GeneBurdenMatrix":
        """
        Filter to subset of genes.

        Args:
            genes: Set of genes to keep

        Returns:
            New GeneBurdenMatrix with filtered genes
        """
        keep_indices = [i for i, g in enumerate(self.genes) if g in genes]
        keep_genes = [self.genes[i] for i in keep_indices]

        return GeneBurdenMatrix(
            samples=self.samples.copy(),
            genes=keep_genes,
            scores=self.scores[:, keep_indices].copy(),
            metadata={**self.metadata, "filtered": True},
        )

    def filter_samples(self, samples: Set[str]) -> "GeneBurdenMatrix":
        """
        Filter to subset of samples.

        Args:
            samples: Set of samples to keep

        Returns:
            New GeneBurdenMatrix with filtered samples
        """
        keep_indices = [i for i, s in enumerate(self.samples) if s in samples]
        keep_samples = [self.samples[i] for i in keep_indices]

        return GeneBurdenMatrix(
            samples=keep_samples,
            genes=self.genes.copy(),
            scores=self.scores[keep_indices, :].copy(),
            metadata={**self.metadata, "filtered": True},
        )

    def normalize(self, method: str = "zscore") -> "GeneBurdenMatrix":
        """
        Normalize burden scores.

        Args:
            method: Normalization method (zscore, minmax, rank)

        Returns:
            New GeneBurdenMatrix with normalized scores
        """
        if method == "zscore":
            mean = np.mean(self.scores, axis=0)
            std = np.std(self.scores, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            normalized = (self.scores - mean) / std

        elif method == "minmax":
            min_val = np.min(self.scores, axis=0)
            max_val = np.max(self.scores, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            normalized = (self.scores - min_val) / range_val

        elif method == "rank":
            from scipy.stats import rankdata
            normalized = np.zeros_like(self.scores)
            for j in range(self.scores.shape[1]):
                normalized[:, j] = rankdata(self.scores[:, j]) / self.scores.shape[0]

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return GeneBurdenMatrix(
            samples=self.samples.copy(),
            genes=self.genes.copy(),
            scores=normalized,
            metadata={**self.metadata, "normalized": method},
        )


class GeneBurdenCalculator:
    """
    Calculates gene-level burden scores from annotated variants.

    Supports multiple weighting schemes:
    - Consequence-based weighting
    - CADD/REVEL score weighting
    - Allele frequency weighting
    """

    def __init__(self, config: Optional[WeightConfig] = None):
        """
        Initialize gene burden calculator.

        Args:
            config: Weighting configuration
        """
        self.config = config or WeightConfig()

    def compute(
        self,
        variants: List[AnnotatedVariant],
        samples: Optional[List[str]] = None,
    ) -> GeneBurdenMatrix:
        """
        Compute gene burden scores from annotated variants.

        Args:
            variants: List of annotated variants
            samples: Optional list of samples (inferred from variants if not provided)

        Returns:
            GeneBurdenMatrix with burden scores
        """
        # Filter variants based on config
        filtered = self._filter_variants(variants)

        # Get unique samples and genes
        if samples is None:
            samples = sorted(set(v.variant.sample_id for v in filtered))
        genes = sorted(set(v.gene_id for v in filtered if v.gene_id is not None))

        # Build index mappings
        sample_idx = {s: i for i, s in enumerate(samples)}
        gene_idx = {g: i for i, g in enumerate(genes)}

        # Initialize score matrix
        scores = np.zeros((len(samples), len(genes)))

        # Track contributing variants
        contributing: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        # Compute weights and aggregate
        for variant in filtered:
            if variant.gene_id is None:
                continue

            sample_id = variant.variant.sample_id
            gene_id = variant.gene_id

            if sample_id not in sample_idx or gene_id not in gene_idx:
                continue

            weight = self._compute_weight(variant)

            s_idx = sample_idx[sample_id]
            g_idx = gene_idx[gene_id]

            if self.config.aggregation == "weighted_sum":
                scores[s_idx, g_idx] += weight
            elif self.config.aggregation == "max":
                scores[s_idx, g_idx] = max(scores[s_idx, g_idx], weight)
            elif self.config.aggregation == "count":
                scores[s_idx, g_idx] += 1

            # Track contributing variant
            variant_key = f"{variant.variant.chrom}:{variant.variant.pos}:{variant.variant.ref}>{variant.variant.alt}"
            contributing[(sample_id, gene_id)].append(variant_key)

        logger.info(
            f"Computed gene burden: {len(samples)} samples, {len(genes)} genes, "
            f"{len(filtered)} variants"
        )

        return GeneBurdenMatrix(
            samples=samples,
            genes=genes,
            scores=scores,
            contributing_variants=dict(contributing),
            metadata={
                "config": str(self.config),
                "n_input_variants": len(variants),
                "n_filtered_variants": len(filtered),
            },
        )

    def _filter_variants(
        self, variants: List[AnnotatedVariant]
    ) -> List[AnnotatedVariant]:
        """Filter variants based on configuration."""
        filtered = []

        # Map string impact to enum
        impact_order = ["MODIFIER", "LOW", "MODERATE", "HIGH"]
        min_idx = impact_order.index(self.config.min_impact)

        for v in variants:
            # Check impact level
            impact_idx = impact_order.index(v.impact.value)
            if impact_idx < min_idx:
                continue

            # Skip synonymous if configured
            if not self.config.include_synonymous:
                if v.consequence == VariantConsequence.SYNONYMOUS:
                    continue

            # Skip modifier impact if configured
            if not self.config.include_modifier:
                if v.impact == ImpactLevel.MODIFIER:
                    continue

            # Check CADD threshold for missense
            if v.consequence == VariantConsequence.MISSENSE:
                if self.config.use_cadd_weighting:
                    if v.cadd_phred is not None and v.cadd_phred < self.config.cadd_threshold:
                        continue

            filtered.append(v)

        return filtered

    def _compute_weight(self, variant: AnnotatedVariant) -> float:
        """Compute weight for a single variant."""
        weight = 0.0

        # Base consequence weight
        consequence_str = variant.consequence.value
        if consequence_str in self.config.consequence_weights:
            weight = self.config.consequence_weights[consequence_str]
        elif variant.impact == ImpactLevel.HIGH:
            weight = 1.0
        elif variant.impact == ImpactLevel.MODERATE:
            weight = 0.5
        else:
            weight = 0.1

        # CADD weighting for missense
        if self.config.use_cadd_weighting and variant.consequence == VariantConsequence.MISSENSE:
            if variant.cadd_phred is not None:
                weight = variant.cadd_phred * self.config.cadd_weight_scale

        # REVEL weighting for missense
        if self.config.use_revel_weighting and variant.consequence == VariantConsequence.MISSENSE:
            if variant.revel_score is not None:
                if variant.revel_score >= self.config.revel_threshold:
                    weight = max(weight, variant.revel_score)

        # Allele frequency weighting
        if self.config.use_af_weighting:
            if variant.gnomad_af is not None:
                af_weight = (1 - variant.gnomad_af) ** self.config.af_weight_beta
                weight *= af_weight

        return max(0.0, weight)

    def compute_lof_burden(
        self, variants: List[AnnotatedVariant]
    ) -> GeneBurdenMatrix:
        """
        Compute burden using only loss-of-function variants.

        Args:
            variants: List of annotated variants

        Returns:
            GeneBurdenMatrix with LoF burden
        """
        # Filter to LoF only
        lof_variants = [v for v in variants if v.is_lof]

        # Use simple counting
        config = WeightConfig(
            consequence_weights={c.value: 1.0 for c in LOF_CONSEQUENCES},
            use_cadd_weighting=False,
            use_revel_weighting=False,
            include_synonymous=False,
            min_impact="HIGH",
            aggregation="count",
        )

        calculator = GeneBurdenCalculator(config)
        return calculator.compute(lof_variants)

    def compute_missense_burden(
        self,
        variants: List[AnnotatedVariant],
        cadd_threshold: float = 20.0,
        revel_threshold: float = 0.5,
    ) -> GeneBurdenMatrix:
        """
        Compute burden using only damaging missense variants.

        Args:
            variants: List of annotated variants
            cadd_threshold: Minimum CADD score
            revel_threshold: Minimum REVEL score

        Returns:
            GeneBurdenMatrix with missense burden
        """
        # Filter to missense only
        missense = [v for v in variants if v.consequence == VariantConsequence.MISSENSE]

        config = WeightConfig(
            consequence_weights={"missense_variant": 1.0},
            use_cadd_weighting=True,
            cadd_threshold=cadd_threshold,
            use_revel_weighting=True,
            revel_threshold=revel_threshold,
            min_impact="MODERATE",
            aggregation="weighted_sum",
        )

        calculator = GeneBurdenCalculator(config)
        return calculator.compute(missense)

    @staticmethod
    def combine_burdens(
        burdens: List[GeneBurdenMatrix],
        weights: Optional[List[float]] = None,
    ) -> GeneBurdenMatrix:
        """
        Combine multiple burden matrices.

        Args:
            burdens: List of GeneBurdenMatrix objects
            weights: Optional weights for each burden type

        Returns:
            Combined GeneBurdenMatrix
        """
        if not burdens:
            raise ValueError("No burden matrices to combine")

        if len(burdens) == 1:
            return burdens[0]

        if weights is None:
            weights = [1.0] * len(burdens)

        # Ensure same samples and genes
        samples = burdens[0].samples
        genes = burdens[0].genes

        for b in burdens[1:]:
            if b.samples != samples or b.genes != genes:
                raise ValueError("All burden matrices must have same samples and genes")

        # Combine scores
        combined_scores = np.zeros_like(burdens[0].scores)
        for b, w in zip(burdens, weights):
            combined_scores += w * b.scores

        return GeneBurdenMatrix(
            samples=samples,
            genes=genes,
            scores=combined_scores,
            metadata={"combined": True, "n_sources": len(burdens)},
        )
