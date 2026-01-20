"""
VCF Loader

Handles loading and parsing of Variant Call Format (VCF) files.
Supports both uncompressed (.vcf) and compressed (.vcf.gz) files.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterator
from pathlib import Path
import gzip
import logging

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    """Represents a single genetic variant."""

    chrom: str
    pos: int
    ref: str
    alt: str
    sample_id: str
    genotype: str
    quality: float
    info: Dict[str, Any] = field(default_factory=dict)
    filter_status: str = "PASS"
    variant_id: Optional[str] = None

    @property
    def is_snv(self) -> bool:
        """Check if variant is a single nucleotide variant."""
        return len(self.ref) == 1 and len(self.alt) == 1

    @property
    def is_indel(self) -> bool:
        """Check if variant is an insertion or deletion."""
        return len(self.ref) != len(self.alt)

    @property
    def is_insertion(self) -> bool:
        """Check if variant is an insertion."""
        return len(self.alt) > len(self.ref)

    @property
    def is_deletion(self) -> bool:
        """Check if variant is a deletion."""
        return len(self.ref) > len(self.alt)

    @property
    def variant_type(self) -> str:
        """Get the type of variant."""
        if self.is_snv:
            return "SNV"
        elif self.is_insertion:
            return "INS"
        elif self.is_deletion:
            return "DEL"
        else:
            return "COMPLEX"

    def __hash__(self) -> int:
        return hash((self.chrom, self.pos, self.ref, self.alt, self.sample_id))


@dataclass
class VariantDataset:
    """Collection of variants with associated metadata."""

    variants: List[Variant]
    samples: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.variants)

    def __iter__(self) -> Iterator[Variant]:
        return iter(self.variants)

    def get_variants_by_sample(self, sample_id: str) -> List[Variant]:
        """Get all variants for a specific sample."""
        return [v for v in self.variants if v.sample_id == sample_id]

    def get_variants_by_chrom(self, chrom: str) -> List[Variant]:
        """Get all variants on a specific chromosome."""
        return [v for v in self.variants if v.chrom == chrom]

    def get_variants_by_region(
        self, chrom: str, start: int, end: int
    ) -> List[Variant]:
        """Get variants within a genomic region."""
        return [
            v for v in self.variants
            if v.chrom == chrom and start <= v.pos <= end
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "n_variants": len(self.variants),
            "n_samples": len(self.samples),
            "samples": self.samples,
            "metadata": self.metadata,
        }


@dataclass
class ValidationReport:
    """Report from validating a variant dataset."""

    is_valid: bool
    n_variants: int
    n_samples: int
    n_chromosomes: int
    variant_types: Dict[str, int]
    quality_stats: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [
            f"Validation Report: {status}",
            f"  Variants: {self.n_variants}",
            f"  Samples: {self.n_samples}",
            f"  Chromosomes: {self.n_chromosomes}",
            f"  Variant types: {self.variant_types}",
        ]
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
        return "\n".join(lines)


class VCFLoader:
    """
    Loader for VCF (Variant Call Format) files.

    Supports:
    - Uncompressed VCF (.vcf)
    - Gzip compressed VCF (.vcf.gz)
    - Multi-sample VCFs
    - Standard VCF 4.x format
    """

    def __init__(
        self,
        min_quality: float = 0.0,
        filter_pass_only: bool = False,
        include_info: bool = True,
    ):
        """
        Initialize VCF loader.

        Args:
            min_quality: Minimum quality score to include variant
            filter_pass_only: Only include variants with PASS filter
            include_info: Parse and include INFO field data
        """
        self.min_quality = min_quality
        self.filter_pass_only = filter_pass_only
        self.include_info = include_info

    def load(self, vcf_path: str) -> VariantDataset:
        """
        Load variants from a VCF file.

        Args:
            vcf_path: Path to VCF file (.vcf or .vcf.gz)

        Returns:
            VariantDataset containing all variants
        """
        path = Path(vcf_path)
        if not path.exists():
            raise FileNotFoundError(f"VCF file not found: {vcf_path}")

        variants = []
        samples = []
        metadata = {
            "source_file": str(path),
            "format_version": None,
            "contigs": [],
            "info_fields": {},
            "format_fields": {},
        }

        # Determine if file is gzipped
        open_func = gzip.open if str(path).endswith(".gz") else open
        mode = "rt" if str(path).endswith(".gz") else "r"

        with open_func(path, mode) as f:
            for line in f:
                line = line.strip()

                # Parse header lines
                if line.startswith("##"):
                    self._parse_header_line(line, metadata)
                    continue

                # Parse column header line
                if line.startswith("#CHROM"):
                    columns = line.split("\t")
                    if len(columns) > 9:
                        samples = columns[9:]
                    continue

                # Parse variant line
                if not line:
                    continue

                parsed_variants = self._parse_variant_line(line, samples)
                variants.extend(parsed_variants)

        logger.info(f"Loaded {len(variants)} variants from {len(samples)} samples")

        return VariantDataset(
            variants=variants,
            samples=samples,
            metadata=metadata,
        )

    def _parse_header_line(self, line: str, metadata: Dict[str, Any]) -> None:
        """Parse a VCF header line (starting with ##)."""
        if line.startswith("##fileformat="):
            metadata["format_version"] = line.split("=")[1]
        elif line.startswith("##contig="):
            # Extract contig ID
            if "ID=" in line:
                start = line.index("ID=") + 3
                end = line.index(",", start) if "," in line[start:] else line.index(">", start)
                metadata["contigs"].append(line[start:end])
        elif line.startswith("##INFO="):
            self._parse_field_definition(line, metadata["info_fields"])
        elif line.startswith("##FORMAT="):
            self._parse_field_definition(line, metadata["format_fields"])

    def _parse_field_definition(
        self, line: str, field_dict: Dict[str, Dict[str, str]]
    ) -> None:
        """Parse INFO or FORMAT field definitions."""
        # Extract ID
        if "ID=" not in line:
            return

        start = line.index("ID=") + 3
        end = line.index(",", start)
        field_id = line[start:end]

        field_dict[field_id] = {"raw": line}

    def _parse_variant_line(
        self, line: str, samples: List[str]
    ) -> List[Variant]:
        """Parse a variant data line."""
        fields = line.split("\t")
        if len(fields) < 8:
            logger.warning(f"Skipping malformed line: {line[:50]}...")
            return []

        chrom = fields[0]
        pos = int(fields[1])
        variant_id = fields[2] if fields[2] != "." else None
        ref = fields[3]
        alt_str = fields[4]
        quality = float(fields[5]) if fields[5] != "." else 0.0
        filter_status = fields[6]

        # Apply filters
        if quality < self.min_quality:
            return []
        if self.filter_pass_only and filter_status != "PASS":
            return []

        # Parse INFO field
        info = {}
        if self.include_info and len(fields) > 7:
            info = self._parse_info_field(fields[7])

        # Handle multiple alternate alleles
        alts = alt_str.split(",")

        variants = []

        # Parse genotypes for each sample
        if len(fields) > 9 and samples:
            format_field = fields[8].split(":")
            gt_index = format_field.index("GT") if "GT" in format_field else 0

            for i, sample_data in enumerate(fields[9:]):
                if i >= len(samples):
                    break

                sample_id = samples[i]
                sample_fields = sample_data.split(":")

                if len(sample_fields) <= gt_index:
                    continue

                genotype = sample_fields[gt_index]

                # Skip reference-only genotypes
                if genotype in ("0/0", "0|0", "./.", ".|."):
                    continue

                # Determine which alt allele(s) are present
                gt_alleles = genotype.replace("|", "/").split("/")
                for gt_allele in gt_alleles:
                    if gt_allele == "." or gt_allele == "0":
                        continue
                    try:
                        allele_idx = int(gt_allele) - 1
                        if 0 <= allele_idx < len(alts):
                            variant = Variant(
                                chrom=chrom,
                                pos=pos,
                                ref=ref,
                                alt=alts[allele_idx],
                                sample_id=sample_id,
                                genotype=genotype,
                                quality=quality,
                                info=info.copy(),
                                filter_status=filter_status,
                                variant_id=variant_id,
                            )
                            variants.append(variant)
                    except ValueError:
                        continue
        else:
            # Single-sample or no genotype data - create one variant per alt
            for alt in alts:
                variant = Variant(
                    chrom=chrom,
                    pos=pos,
                    ref=ref,
                    alt=alt,
                    sample_id="UNKNOWN",
                    genotype="./.",
                    quality=quality,
                    info=info.copy(),
                    filter_status=filter_status,
                    variant_id=variant_id,
                )
                variants.append(variant)

        return variants

    def _parse_info_field(self, info_str: str) -> Dict[str, Any]:
        """Parse the INFO field of a VCF line."""
        info = {}
        if info_str == ".":
            return info

        for item in info_str.split(";"):
            if "=" in item:
                key, value = item.split("=", 1)
                # Try to convert to appropriate type
                if "," in value:
                    info[key] = value.split(",")
                else:
                    try:
                        info[key] = int(value)
                    except ValueError:
                        try:
                            info[key] = float(value)
                        except ValueError:
                            info[key] = value
            else:
                # Flag field (presence indicates True)
                info[item] = True

        return info

    def validate(self, dataset: VariantDataset) -> ValidationReport:
        """
        Validate a variant dataset.

        Args:
            dataset: VariantDataset to validate

        Returns:
            ValidationReport with validation results
        """
        warnings = []
        errors = []

        # Count variant types
        variant_types: Dict[str, int] = {}
        qualities = []
        chromosomes = set()

        for variant in dataset.variants:
            vtype = variant.variant_type
            variant_types[vtype] = variant_types.get(vtype, 0) + 1
            qualities.append(variant.quality)
            chromosomes.add(variant.chrom)

        # Compute quality statistics
        quality_stats = {}
        if qualities:
            qualities_sorted = sorted(qualities)
            quality_stats = {
                "min": min(qualities),
                "max": max(qualities),
                "mean": sum(qualities) / len(qualities),
                "median": qualities_sorted[len(qualities) // 2],
            }

        # Check for common issues
        if len(dataset.variants) == 0:
            warnings.append("No variants loaded")

        if len(dataset.samples) == 0:
            warnings.append("No samples found in VCF")

        # Check for unusual chromosome names
        standard_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}
        standard_chroms |= {str(i) for i in range(1, 23)} | {"X", "Y", "M", "MT"}
        unusual_chroms = chromosomes - standard_chroms
        if unusual_chroms:
            warnings.append(f"Unusual chromosome names: {unusual_chroms}")

        is_valid = len(errors) == 0

        return ValidationReport(
            is_valid=is_valid,
            n_variants=len(dataset.variants),
            n_samples=len(dataset.samples),
            n_chromosomes=len(chromosomes),
            variant_types=variant_types,
            quality_stats=quality_stats,
            warnings=warnings,
            errors=errors,
        )

    def load_region(
        self, vcf_path: str, chrom: str, start: int, end: int
    ) -> VariantDataset:
        """
        Load variants from a specific genomic region.

        Note: For large VCF files, consider using tabix-indexed files
        with pysam for efficient region queries.

        Args:
            vcf_path: Path to VCF file
            chrom: Chromosome name
            start: Start position (1-based)
            end: End position (1-based)

        Returns:
            VariantDataset with variants in the region
        """
        dataset = self.load(vcf_path)
        region_variants = dataset.get_variants_by_region(chrom, start, end)

        return VariantDataset(
            variants=region_variants,
            samples=dataset.samples,
            metadata={**dataset.metadata, "region": f"{chrom}:{start}-{end}"},
        )
