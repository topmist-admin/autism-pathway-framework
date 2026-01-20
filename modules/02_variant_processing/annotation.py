"""
Variant Annotation

Functional annotation of genetic variants with consequence prediction
and impact classification.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

# Import from Module 01
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.01_data_loaders import Variant, VariantDataset, GeneAnnotationDB

logger = logging.getLogger(__name__)


class VariantConsequence(Enum):
    """Standard variant consequence types (based on Sequence Ontology)."""

    # High impact
    FRAMESHIFT = "frameshift_variant"
    NONSENSE = "stop_gained"
    SPLICE_ACCEPTOR = "splice_acceptor_variant"
    SPLICE_DONOR = "splice_donor_variant"
    START_LOST = "start_lost"
    STOP_LOST = "stop_lost"

    # Moderate impact
    MISSENSE = "missense_variant"
    INFRAME_INSERTION = "inframe_insertion"
    INFRAME_DELETION = "inframe_deletion"
    PROTEIN_ALTERING = "protein_altering_variant"

    # Low impact
    SPLICE_REGION = "splice_region_variant"
    SYNONYMOUS = "synonymous_variant"
    START_RETAINED = "start_retained_variant"
    STOP_RETAINED = "stop_retained_variant"

    # Modifier
    INTRON = "intron_variant"
    UTR_5 = "5_prime_UTR_variant"
    UTR_3 = "3_prime_UTR_variant"
    UPSTREAM = "upstream_gene_variant"
    DOWNSTREAM = "downstream_gene_variant"
    INTERGENIC = "intergenic_variant"
    NON_CODING = "non_coding_transcript_variant"

    UNKNOWN = "unknown"


class ImpactLevel(Enum):
    """Variant impact levels."""

    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    MODIFIER = "MODIFIER"


# Mapping from consequence to impact
CONSEQUENCE_IMPACT = {
    VariantConsequence.FRAMESHIFT: ImpactLevel.HIGH,
    VariantConsequence.NONSENSE: ImpactLevel.HIGH,
    VariantConsequence.SPLICE_ACCEPTOR: ImpactLevel.HIGH,
    VariantConsequence.SPLICE_DONOR: ImpactLevel.HIGH,
    VariantConsequence.START_LOST: ImpactLevel.HIGH,
    VariantConsequence.STOP_LOST: ImpactLevel.HIGH,
    VariantConsequence.MISSENSE: ImpactLevel.MODERATE,
    VariantConsequence.INFRAME_INSERTION: ImpactLevel.MODERATE,
    VariantConsequence.INFRAME_DELETION: ImpactLevel.MODERATE,
    VariantConsequence.PROTEIN_ALTERING: ImpactLevel.MODERATE,
    VariantConsequence.SPLICE_REGION: ImpactLevel.LOW,
    VariantConsequence.SYNONYMOUS: ImpactLevel.LOW,
    VariantConsequence.START_RETAINED: ImpactLevel.LOW,
    VariantConsequence.STOP_RETAINED: ImpactLevel.LOW,
    VariantConsequence.INTRON: ImpactLevel.MODIFIER,
    VariantConsequence.UTR_5: ImpactLevel.MODIFIER,
    VariantConsequence.UTR_3: ImpactLevel.MODIFIER,
    VariantConsequence.UPSTREAM: ImpactLevel.MODIFIER,
    VariantConsequence.DOWNSTREAM: ImpactLevel.MODIFIER,
    VariantConsequence.INTERGENIC: ImpactLevel.MODIFIER,
    VariantConsequence.NON_CODING: ImpactLevel.MODIFIER,
    VariantConsequence.UNKNOWN: ImpactLevel.MODIFIER,
}

# Loss-of-function consequences
LOF_CONSEQUENCES = {
    VariantConsequence.FRAMESHIFT,
    VariantConsequence.NONSENSE,
    VariantConsequence.SPLICE_ACCEPTOR,
    VariantConsequence.SPLICE_DONOR,
    VariantConsequence.START_LOST,
}


@dataclass
class AnnotatedVariant:
    """A variant with functional annotations."""

    variant: Variant
    gene_id: Optional[str] = None
    gene_name: Optional[str] = None
    transcript_id: Optional[str] = None
    consequence: VariantConsequence = VariantConsequence.UNKNOWN
    impact: ImpactLevel = ImpactLevel.MODIFIER
    hgvsc: Optional[str] = None  # cDNA change
    hgvsp: Optional[str] = None  # Protein change

    # Pathogenicity scores
    cadd_score: Optional[float] = None
    cadd_phred: Optional[float] = None
    revel_score: Optional[float] = None
    polyphen_score: Optional[float] = None
    sift_score: Optional[float] = None

    # Population frequencies
    gnomad_af: Optional[float] = None
    gnomad_af_popmax: Optional[float] = None

    # Additional annotations
    is_canonical: bool = True
    is_lof: bool = False
    lof_confidence: Optional[str] = None  # HC, LC for LOFTEE

    @property
    def is_damaging(self) -> bool:
        """Check if variant is predicted damaging."""
        # High impact is always damaging
        if self.impact == ImpactLevel.HIGH:
            return True

        # Moderate impact with high CADD or REVEL
        if self.impact == ImpactLevel.MODERATE:
            if self.cadd_phred is not None and self.cadd_phred >= 20:
                return True
            if self.revel_score is not None and self.revel_score >= 0.5:
                return True

        return False

    @property
    def is_rare(self) -> bool:
        """Check if variant is rare (AF < 1%)."""
        if self.gnomad_af is None:
            return True  # Assume rare if no frequency data
        return self.gnomad_af < 0.01

    @property
    def is_ultra_rare(self) -> bool:
        """Check if variant is ultra-rare (AF < 0.01%)."""
        if self.gnomad_af is None:
            return True
        return self.gnomad_af < 0.0001

    def __hash__(self) -> int:
        return hash((
            self.variant.chrom, self.variant.pos,
            self.variant.ref, self.variant.alt,
            self.variant.sample_id, self.gene_id
        ))


class VariantAnnotator:
    """
    Annotates variants with functional consequences and pathogenicity scores.

    Can parse annotations from:
    - VEP (Variant Effect Predictor) output
    - ANNOVAR output
    - Pre-annotated VCF INFO fields
    """

    # Consequence string to enum mapping
    CONSEQUENCE_MAP = {
        "frameshift_variant": VariantConsequence.FRAMESHIFT,
        "frameshift": VariantConsequence.FRAMESHIFT,
        "stop_gained": VariantConsequence.NONSENSE,
        "nonsense": VariantConsequence.NONSENSE,
        "splice_acceptor_variant": VariantConsequence.SPLICE_ACCEPTOR,
        "splice_donor_variant": VariantConsequence.SPLICE_DONOR,
        "splicing": VariantConsequence.SPLICE_DONOR,
        "start_lost": VariantConsequence.START_LOST,
        "stop_lost": VariantConsequence.STOP_LOST,
        "missense_variant": VariantConsequence.MISSENSE,
        "missense": VariantConsequence.MISSENSE,
        "nonsynonymous": VariantConsequence.MISSENSE,
        "inframe_insertion": VariantConsequence.INFRAME_INSERTION,
        "inframe_deletion": VariantConsequence.INFRAME_DELETION,
        "protein_altering_variant": VariantConsequence.PROTEIN_ALTERING,
        "splice_region_variant": VariantConsequence.SPLICE_REGION,
        "synonymous_variant": VariantConsequence.SYNONYMOUS,
        "synonymous": VariantConsequence.SYNONYMOUS,
        "intron_variant": VariantConsequence.INTRON,
        "intronic": VariantConsequence.INTRON,
        "5_prime_UTR_variant": VariantConsequence.UTR_5,
        "UTR5": VariantConsequence.UTR_5,
        "3_prime_UTR_variant": VariantConsequence.UTR_3,
        "UTR3": VariantConsequence.UTR_3,
        "upstream_gene_variant": VariantConsequence.UPSTREAM,
        "upstream": VariantConsequence.UPSTREAM,
        "downstream_gene_variant": VariantConsequence.DOWNSTREAM,
        "downstream": VariantConsequence.DOWNSTREAM,
        "intergenic_variant": VariantConsequence.INTERGENIC,
        "intergenic": VariantConsequence.INTERGENIC,
    }

    def __init__(self):
        """Initialize variant annotator."""
        self._gene_db: Optional[GeneAnnotationDB] = None

    def set_gene_database(self, gene_db: GeneAnnotationDB) -> None:
        """Set gene annotation database for coordinate-based lookup."""
        self._gene_db = gene_db

    def annotate(
        self,
        variants: List[Variant],
        gene_db: Optional[GeneAnnotationDB] = None,
    ) -> List[AnnotatedVariant]:
        """
        Annotate a list of variants.

        Annotations are extracted from variant INFO fields if available,
        otherwise minimal annotations are created.

        Args:
            variants: List of variants to annotate
            gene_db: Optional gene annotation database

        Returns:
            List of annotated variants
        """
        if gene_db is not None:
            self._gene_db = gene_db

        annotated = []
        for variant in variants:
            ann = self._annotate_single(variant)
            annotated.append(ann)

        logger.info(f"Annotated {len(annotated)} variants")

        return annotated

    def _annotate_single(self, variant: Variant) -> AnnotatedVariant:
        """Annotate a single variant."""
        # Try to extract annotations from INFO field
        info = variant.info

        # Parse gene
        gene_id = self._extract_gene(info)

        # Parse consequence
        consequence = self._extract_consequence(info)
        impact = CONSEQUENCE_IMPACT.get(consequence, ImpactLevel.MODIFIER)

        # Parse pathogenicity scores
        cadd_phred = self._extract_float(info, ["CADD_PHRED", "CADD_phred", "CADD"])
        revel = self._extract_float(info, ["REVEL", "REVEL_score"])
        sift = self._extract_float(info, ["SIFT", "SIFT_score"])
        polyphen = self._extract_float(info, ["PolyPhen", "Polyphen2_HDIV_score"])

        # Parse population frequencies
        gnomad_af = self._extract_float(info, ["gnomAD_AF", "AF_gnomAD", "AF"])
        gnomad_popmax = self._extract_float(info, ["gnomAD_AF_popmax", "AF_popmax"])

        # Parse transcript info
        transcript = self._extract_string(info, ["Feature", "transcript", "TRANSCRIPT"])
        hgvsc = self._extract_string(info, ["HGVSc", "cDNA_change"])
        hgvsp = self._extract_string(info, ["HGVSp", "Protein_change", "AAChange"])

        # Determine if LoF
        is_lof = consequence in LOF_CONSEQUENCES
        lof_conf = self._extract_string(info, ["LoF", "LOFTEE", "LoF_filter"])

        return AnnotatedVariant(
            variant=variant,
            gene_id=gene_id,
            transcript_id=transcript,
            consequence=consequence,
            impact=impact,
            hgvsc=hgvsc,
            hgvsp=hgvsp,
            cadd_phred=cadd_phred,
            revel_score=revel,
            sift_score=sift,
            polyphen_score=polyphen,
            gnomad_af=gnomad_af,
            gnomad_af_popmax=gnomad_popmax,
            is_lof=is_lof,
            lof_confidence=lof_conf,
        )

    def _extract_gene(self, info: Dict[str, Any]) -> Optional[str]:
        """Extract gene symbol from INFO."""
        for key in ["SYMBOL", "Gene", "gene", "Gene.refGene", "GENE"]:
            if key in info:
                val = info[key]
                if isinstance(val, list):
                    val = val[0]
                if val and val not in (".", "NA", ""):
                    return str(val)
        return None

    def _extract_consequence(self, info: Dict[str, Any]) -> VariantConsequence:
        """Extract and map variant consequence."""
        for key in ["Consequence", "Func.refGene", "ExonicFunc.refGene", "effect"]:
            if key in info:
                val = info[key]
                if isinstance(val, list):
                    val = val[0]
                if val:
                    # Try to map
                    val_lower = str(val).lower()
                    for pattern, consequence in self.CONSEQUENCE_MAP.items():
                        if pattern.lower() in val_lower:
                            return consequence

        return VariantConsequence.UNKNOWN

    def _extract_float(
        self, info: Dict[str, Any], keys: List[str]
    ) -> Optional[float]:
        """Extract float value from INFO."""
        for key in keys:
            if key in info:
                val = info[key]
                if isinstance(val, list):
                    val = val[0]
                try:
                    if val and val not in (".", "NA", ""):
                        return float(val)
                except (ValueError, TypeError):
                    pass
        return None

    def _extract_string(
        self, info: Dict[str, Any], keys: List[str]
    ) -> Optional[str]:
        """Extract string value from INFO."""
        for key in keys:
            if key in info:
                val = info[key]
                if isinstance(val, list):
                    val = val[0]
                if val and val not in (".", "NA", ""):
                    return str(val)
        return None

    def classify_impact(self, variant: AnnotatedVariant) -> str:
        """
        Classify variant impact.

        Args:
            variant: Annotated variant

        Returns:
            Impact level string (HIGH, MODERATE, LOW, MODIFIER)
        """
        return variant.impact.value

    def filter_by_impact(
        self,
        variants: List[AnnotatedVariant],
        min_impact: ImpactLevel = ImpactLevel.MODERATE,
    ) -> List[AnnotatedVariant]:
        """
        Filter variants by minimum impact level.

        Args:
            variants: List of annotated variants
            min_impact: Minimum impact level to include

        Returns:
            Filtered list of variants
        """
        impact_order = [ImpactLevel.MODIFIER, ImpactLevel.LOW, ImpactLevel.MODERATE, ImpactLevel.HIGH]
        min_idx = impact_order.index(min_impact)

        return [v for v in variants if impact_order.index(v.impact) >= min_idx]

    def filter_by_consequence(
        self,
        variants: List[AnnotatedVariant],
        consequences: Set[VariantConsequence],
    ) -> List[AnnotatedVariant]:
        """
        Filter variants by specific consequences.

        Args:
            variants: List of annotated variants
            consequences: Set of consequences to include

        Returns:
            Filtered list of variants
        """
        return [v for v in variants if v.consequence in consequences]

    def get_lof_variants(
        self,
        variants: List[AnnotatedVariant],
        high_confidence_only: bool = False,
    ) -> List[AnnotatedVariant]:
        """
        Get loss-of-function variants.

        Args:
            variants: List of annotated variants
            high_confidence_only: Only return HC LOFTEE variants

        Returns:
            List of LoF variants
        """
        lof_variants = [v for v in variants if v.is_lof]

        if high_confidence_only:
            lof_variants = [
                v for v in lof_variants
                if v.lof_confidence in ("HC", "high_confidence", None)
            ]

        return lof_variants

    def get_damaging_missense(
        self,
        variants: List[AnnotatedVariant],
        cadd_threshold: float = 20.0,
        revel_threshold: float = 0.5,
    ) -> List[AnnotatedVariant]:
        """
        Get damaging missense variants.

        Args:
            variants: List of annotated variants
            cadd_threshold: Minimum CADD phred score
            revel_threshold: Minimum REVEL score

        Returns:
            List of damaging missense variants
        """
        damaging = []

        for v in variants:
            if v.consequence != VariantConsequence.MISSENSE:
                continue

            # Check CADD
            if v.cadd_phred is not None and v.cadd_phred >= cadd_threshold:
                damaging.append(v)
                continue

            # Check REVEL
            if v.revel_score is not None and v.revel_score >= revel_threshold:
                damaging.append(v)
                continue

        return damaging

    def summarize_by_gene(
        self, variants: List[AnnotatedVariant]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Summarize variants by gene.

        Args:
            variants: List of annotated variants

        Returns:
            Dictionary mapping gene to variant summary
        """
        from collections import defaultdict

        gene_summary: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "n_variants": 0,
                "n_lof": 0,
                "n_missense": 0,
                "n_damaging": 0,
                "samples": set(),
                "max_cadd": None,
            }
        )

        for v in variants:
            if v.gene_id is None:
                continue

            summary = gene_summary[v.gene_id]
            summary["n_variants"] += 1
            summary["samples"].add(v.variant.sample_id)

            if v.is_lof:
                summary["n_lof"] += 1
            if v.consequence == VariantConsequence.MISSENSE:
                summary["n_missense"] += 1
            if v.is_damaging:
                summary["n_damaging"] += 1

            if v.cadd_phred is not None:
                if summary["max_cadd"] is None or v.cadd_phred > summary["max_cadd"]:
                    summary["max_cadd"] = v.cadd_phred

        # Convert sample sets to counts
        for gene, summary in gene_summary.items():
            summary["n_samples"] = len(summary["samples"])
            del summary["samples"]

        return dict(gene_summary)
