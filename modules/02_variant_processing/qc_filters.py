"""
QC Filters

Quality control filtering for genetic variants and samples.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import logging

# Import from Module 01
import sys
from pathlib import Path
_module_root = Path(__file__).parent.parent
sys.path.insert(0, str(_module_root / "01_data_loaders"))
from vcf_loader import Variant, VariantDataset

logger = logging.getLogger(__name__)


@dataclass
class QCConfig:
    """Configuration for quality control filtering."""

    # Variant-level filters
    min_quality: float = 20.0
    min_depth: Optional[int] = 10
    min_genotype_quality: Optional[int] = 20
    filter_pass_only: bool = True

    # Allele frequency filters
    min_allele_freq: float = 0.0
    max_allele_freq: float = 0.01  # 1% - rare variants

    # Sample-level filters
    max_missing_rate: float = 0.1  # 10% missing
    min_variants_per_sample: int = 0
    max_variants_per_sample: Optional[int] = None

    # Variant-level missingness
    max_variant_missing_rate: float = 0.1

    # Additional filters
    exclude_chromosomes: Set[str] = field(default_factory=lambda: {"chrM", "M", "MT"})
    include_variant_types: Optional[Set[str]] = None  # None = all types


@dataclass
class QCReport:
    """Report from QC filtering."""

    # Input counts
    input_variants: int
    input_samples: int

    # Output counts
    output_variants: int
    output_samples: int

    # Filtering statistics
    variants_removed: Dict[str, int]
    samples_removed: Dict[str, int]

    # Quality metrics
    quality_distribution: Dict[str, float]
    allele_freq_distribution: Dict[str, float]

    def __str__(self) -> str:
        lines = [
            "QC Report",
            "=" * 40,
            f"Input: {self.input_variants} variants, {self.input_samples} samples",
            f"Output: {self.output_variants} variants, {self.output_samples} samples",
            "",
            "Variants removed by filter:",
        ]
        for reason, count in self.variants_removed.items():
            lines.append(f"  {reason}: {count}")

        if self.samples_removed:
            lines.append("")
            lines.append("Samples removed by filter:")
            for reason, count in self.samples_removed.items():
                lines.append(f"  {reason}: {count}")

        return "\n".join(lines)

    @property
    def variant_retention_rate(self) -> float:
        """Fraction of variants retained."""
        if self.input_variants == 0:
            return 0.0
        return self.output_variants / self.input_variants

    @property
    def sample_retention_rate(self) -> float:
        """Fraction of samples retained."""
        if self.input_samples == 0:
            return 0.0
        return self.output_samples / self.input_samples


class QCFilter:
    """
    Quality control filter for genetic variants and samples.

    Applies configurable filters for:
    - Variant quality scores
    - Read depth
    - Allele frequency
    - Sample call rate
    - Variant call rate
    """

    def __init__(self):
        """Initialize QC filter."""
        self._last_report: Optional[QCReport] = None
        self._variants_removed: Dict[str, int] = defaultdict(int)
        self._samples_removed: Dict[str, int] = defaultdict(int)

    def filter_variants(
        self, dataset: VariantDataset, config: QCConfig
    ) -> VariantDataset:
        """
        Filter variants based on QC criteria.

        Args:
            dataset: Input variant dataset
            config: QC configuration

        Returns:
            Filtered VariantDataset
        """
        self._variants_removed = defaultdict(int)
        input_count = len(dataset.variants)

        filtered_variants = []

        for variant in dataset.variants:
            # Apply filters
            passed, reason = self._check_variant(variant, config)
            if passed:
                filtered_variants.append(variant)
            else:
                self._variants_removed[reason] += 1

        # Create new dataset
        filtered_dataset = VariantDataset(
            variants=filtered_variants,
            samples=dataset.samples,
            metadata={
                **dataset.metadata,
                "qc_filtered": True,
                "qc_config": str(config),
            },
        )

        logger.info(
            f"Variant QC: {input_count} -> {len(filtered_variants)} "
            f"({100 * len(filtered_variants) / max(input_count, 1):.1f}% retained)"
        )

        return filtered_dataset

    def _check_variant(
        self, variant: Variant, config: QCConfig
    ) -> Tuple[bool, str]:
        """
        Check if a variant passes all filters.

        Returns:
            Tuple of (passed, reason_if_failed)
        """
        # Quality filter
        if variant.quality < config.min_quality:
            return False, "low_quality"

        # Filter status
        if config.filter_pass_only and variant.filter_status != "PASS":
            return False, "non_pass_filter"

        # Chromosome filter
        if variant.chrom in config.exclude_chromosomes:
            return False, "excluded_chromosome"

        # Variant type filter
        if config.include_variant_types is not None:
            if variant.variant_type not in config.include_variant_types:
                return False, "excluded_variant_type"

        # Depth filter (if available in INFO)
        if config.min_depth is not None:
            depth = variant.info.get("DP")
            if depth is not None and depth < config.min_depth:
                return False, "low_depth"

        # Allele frequency filter (if available in INFO)
        af = self._get_allele_frequency(variant)
        if af is not None:
            if af < config.min_allele_freq:
                return False, "low_af"
            if af > config.max_allele_freq:
                return False, "high_af"

        return True, ""

    def _get_allele_frequency(self, variant: Variant) -> Optional[float]:
        """Extract allele frequency from variant INFO."""
        # Try common AF field names
        for key in ["AF", "MAF", "gnomAD_AF", "ExAC_AF"]:
            if key in variant.info:
                val = variant.info[key]
                if isinstance(val, list):
                    val = val[0]
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return None

    def filter_samples(
        self, dataset: VariantDataset, config: QCConfig
    ) -> VariantDataset:
        """
        Filter samples based on QC criteria.

        Args:
            dataset: Input variant dataset
            config: QC configuration

        Returns:
            Filtered VariantDataset with only passing samples
        """
        self._samples_removed = defaultdict(int)

        # Calculate per-sample statistics
        sample_stats = self._calculate_sample_stats(dataset)

        # Determine which samples pass
        passing_samples = set()
        for sample_id in dataset.samples:
            stats = sample_stats.get(sample_id, {})
            passed, reason = self._check_sample(stats, config)
            if passed:
                passing_samples.add(sample_id)
            else:
                self._samples_removed[reason] += 1

        # Filter variants to only include passing samples
        filtered_variants = [
            v for v in dataset.variants if v.sample_id in passing_samples
        ]

        filtered_dataset = VariantDataset(
            variants=filtered_variants,
            samples=[s for s in dataset.samples if s in passing_samples],
            metadata={
                **dataset.metadata,
                "sample_qc_filtered": True,
            },
        )

        logger.info(
            f"Sample QC: {len(dataset.samples)} -> {len(passing_samples)} "
            f"({100 * len(passing_samples) / max(len(dataset.samples), 1):.1f}% retained)"
        )

        return filtered_dataset

    def _calculate_sample_stats(
        self, dataset: VariantDataset
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for each sample."""
        stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"n_variants": 0, "total_quality": 0.0}
        )

        for variant in dataset.variants:
            sample_id = variant.sample_id
            stats[sample_id]["n_variants"] += 1
            stats[sample_id]["total_quality"] += variant.quality

        # Calculate derived stats
        for sample_id, sample_stats in stats.items():
            n = sample_stats["n_variants"]
            if n > 0:
                sample_stats["mean_quality"] = sample_stats["total_quality"] / n
            else:
                sample_stats["mean_quality"] = 0.0

        return dict(stats)

    def _check_sample(
        self, stats: Dict[str, Any], config: QCConfig
    ) -> Tuple[bool, str]:
        """
        Check if a sample passes all filters.

        Returns:
            Tuple of (passed, reason_if_failed)
        """
        n_variants = stats.get("n_variants", 0)

        # Minimum variants filter
        if n_variants < config.min_variants_per_sample:
            return False, "too_few_variants"

        # Maximum variants filter
        if config.max_variants_per_sample is not None:
            if n_variants > config.max_variants_per_sample:
                return False, "too_many_variants"

        return True, ""

    def filter_by_call_rate(
        self,
        dataset: VariantDataset,
        min_call_rate: float = 0.9,
    ) -> VariantDataset:
        """
        Filter variants by call rate across samples.

        Args:
            dataset: Input variant dataset
            min_call_rate: Minimum fraction of samples with calls

        Returns:
            Filtered VariantDataset
        """
        # Group variants by position
        position_samples: Dict[Tuple[str, int, str, str], Set[str]] = defaultdict(set)

        for variant in dataset.variants:
            key = (variant.chrom, variant.pos, variant.ref, variant.alt)
            position_samples[key].add(variant.sample_id)

        n_samples = len(dataset.samples)
        min_samples = int(min_call_rate * n_samples)

        # Find positions with sufficient call rate
        passing_positions = {
            pos for pos, samples in position_samples.items()
            if len(samples) >= min_samples
        }

        # Filter variants
        filtered_variants = [
            v for v in dataset.variants
            if (v.chrom, v.pos, v.ref, v.alt) in passing_positions
        ]

        return VariantDataset(
            variants=filtered_variants,
            samples=dataset.samples,
            metadata={
                **dataset.metadata,
                "call_rate_filtered": True,
                "min_call_rate": min_call_rate,
            },
        )

    def get_qc_report(self) -> QCReport:
        """
        Get QC report from last filtering operation.

        Returns:
            QCReport object
        """
        if self._last_report is not None:
            return self._last_report

        # Create report from current state
        return QCReport(
            input_variants=0,
            input_samples=0,
            output_variants=0,
            output_samples=0,
            variants_removed=dict(self._variants_removed),
            samples_removed=dict(self._samples_removed),
            quality_distribution={},
            allele_freq_distribution={},
        )

    def run_full_qc(
        self, dataset: VariantDataset, config: QCConfig
    ) -> Tuple[VariantDataset, QCReport]:
        """
        Run full QC pipeline on dataset.

        Args:
            dataset: Input variant dataset
            config: QC configuration

        Returns:
            Tuple of (filtered_dataset, qc_report)
        """
        input_variants = len(dataset.variants)
        input_samples = len(dataset.samples)

        # Reset counters
        self._variants_removed = defaultdict(int)
        self._samples_removed = defaultdict(int)

        # Filter variants
        filtered = self.filter_variants(dataset, config)

        # Filter samples
        filtered = self.filter_samples(filtered, config)

        # Calculate quality distribution
        quality_dist = self._calculate_quality_distribution(filtered)
        af_dist = self._calculate_af_distribution(filtered)

        # Create report
        report = QCReport(
            input_variants=input_variants,
            input_samples=input_samples,
            output_variants=len(filtered.variants),
            output_samples=len(filtered.samples),
            variants_removed=dict(self._variants_removed),
            samples_removed=dict(self._samples_removed),
            quality_distribution=quality_dist,
            allele_freq_distribution=af_dist,
        )

        self._last_report = report

        return filtered, report

    def _calculate_quality_distribution(
        self, dataset: VariantDataset
    ) -> Dict[str, float]:
        """Calculate quality score distribution."""
        if not dataset.variants:
            return {}

        qualities = [v.quality for v in dataset.variants]
        qualities.sort()
        n = len(qualities)

        return {
            "min": qualities[0],
            "q25": qualities[n // 4],
            "median": qualities[n // 2],
            "q75": qualities[3 * n // 4],
            "max": qualities[-1],
            "mean": sum(qualities) / n,
        }

    def _calculate_af_distribution(
        self, dataset: VariantDataset
    ) -> Dict[str, float]:
        """Calculate allele frequency distribution."""
        afs = []
        for v in dataset.variants:
            af = self._get_allele_frequency(v)
            if af is not None:
                afs.append(af)

        if not afs:
            return {}

        afs.sort()
        n = len(afs)

        return {
            "min": afs[0],
            "median": afs[n // 2],
            "max": afs[-1],
            "mean": sum(afs) / n,
            "n_with_af": n,
        }
