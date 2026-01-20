"""
Tests for Variant Annotation

Tests for functional annotation of genetic variants.
"""

import pytest
import sys
from pathlib import Path

# Add parent modules to path for imports
_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "modules" / "01_data_loaders"))
sys.path.insert(0, str(_root / "modules" / "02_variant_processing"))

from vcf_loader import Variant
from annotation import (
    VariantAnnotator,
    AnnotatedVariant,
    VariantConsequence,
    ImpactLevel,
    LOF_CONSEQUENCES,
    MISSENSE_CONSEQUENCES,
)


class TestVariantConsequence:
    """Tests for VariantConsequence enum."""

    def test_consequence_values(self):
        """Test that consequences have expected values."""
        assert VariantConsequence.FRAMESHIFT.value == "frameshift_variant"
        assert VariantConsequence.STOP_GAINED.value == "stop_gained"
        assert VariantConsequence.MISSENSE.value == "missense_variant"
        assert VariantConsequence.SYNONYMOUS.value == "synonymous_variant"

    def test_lof_consequences(self):
        """Test LOF consequences set."""
        assert VariantConsequence.FRAMESHIFT in LOF_CONSEQUENCES
        assert VariantConsequence.STOP_GAINED in LOF_CONSEQUENCES
        assert VariantConsequence.SPLICE_ACCEPTOR in LOF_CONSEQUENCES
        assert VariantConsequence.SPLICE_DONOR in LOF_CONSEQUENCES
        assert VariantConsequence.MISSENSE not in LOF_CONSEQUENCES

    def test_missense_consequences(self):
        """Test missense consequences set."""
        assert VariantConsequence.MISSENSE in MISSENSE_CONSEQUENCES
        assert VariantConsequence.FRAMESHIFT not in MISSENSE_CONSEQUENCES


class TestImpactLevel:
    """Tests for ImpactLevel enum."""

    def test_impact_levels(self):
        """Test impact level values."""
        assert ImpactLevel.HIGH.value == "HIGH"
        assert ImpactLevel.MODERATE.value == "MODERATE"
        assert ImpactLevel.LOW.value == "LOW"
        assert ImpactLevel.MODIFIER.value == "MODIFIER"


class TestAnnotatedVariant:
    """Tests for AnnotatedVariant dataclass."""

    @pytest.fixture
    def sample_variant(self):
        """Create a sample variant."""
        return Variant(
            chrom="chr1",
            pos=12345,
            ref="A",
            alt="G",
            quality=50.0,
            filter_status="PASS",
            sample_id="sample1",
            genotype="0/1",
        )

    def test_annotated_variant_creation(self, sample_variant):
        """Test creating an annotated variant."""
        annotated = AnnotatedVariant(
            variant=sample_variant,
            gene_id="BRCA1",
            gene_symbol="BRCA1",
            transcript_id="ENST00000357654",
            consequence=VariantConsequence.MISSENSE,
            impact=ImpactLevel.MODERATE,
            cadd_phred=25.5,
            revel_score=0.75,
            gnomad_af=0.0001,
        )

        assert annotated.gene_id == "BRCA1"
        assert annotated.consequence == VariantConsequence.MISSENSE
        assert annotated.impact == ImpactLevel.MODERATE
        assert annotated.cadd_phred == 25.5
        assert annotated.revel_score == 0.75

    def test_is_lof_property(self, sample_variant):
        """Test is_lof property."""
        # LOF variant
        lof_variant = AnnotatedVariant(
            variant=sample_variant,
            consequence=VariantConsequence.FRAMESHIFT,
            impact=ImpactLevel.HIGH,
        )
        assert lof_variant.is_lof is True

        # Non-LOF variant
        missense_variant = AnnotatedVariant(
            variant=sample_variant,
            consequence=VariantConsequence.MISSENSE,
            impact=ImpactLevel.MODERATE,
        )
        assert missense_variant.is_lof is False

    def test_is_missense_property(self, sample_variant):
        """Test is_missense property."""
        missense_variant = AnnotatedVariant(
            variant=sample_variant,
            consequence=VariantConsequence.MISSENSE,
            impact=ImpactLevel.MODERATE,
        )
        assert missense_variant.is_missense is True

        lof_variant = AnnotatedVariant(
            variant=sample_variant,
            consequence=VariantConsequence.STOP_GAINED,
            impact=ImpactLevel.HIGH,
        )
        assert lof_variant.is_missense is False

    def test_is_coding_property(self, sample_variant):
        """Test is_coding property."""
        # Coding variant
        coding_variant = AnnotatedVariant(
            variant=sample_variant,
            consequence=VariantConsequence.MISSENSE,
            impact=ImpactLevel.MODERATE,
        )
        assert coding_variant.is_coding is True

        # Non-coding variant
        intronic_variant = AnnotatedVariant(
            variant=sample_variant,
            consequence=VariantConsequence.INTRON,
            impact=ImpactLevel.MODIFIER,
        )
        assert intronic_variant.is_coding is False

    def test_is_rare_property(self, sample_variant):
        """Test is_rare property."""
        # Rare variant
        rare_variant = AnnotatedVariant(
            variant=sample_variant,
            consequence=VariantConsequence.MISSENSE,
            impact=ImpactLevel.MODERATE,
            gnomad_af=0.0001,
        )
        assert rare_variant.is_rare is True

        # Common variant
        common_variant = AnnotatedVariant(
            variant=sample_variant,
            consequence=VariantConsequence.MISSENSE,
            impact=ImpactLevel.MODERATE,
            gnomad_af=0.05,
        )
        assert common_variant.is_rare is False

        # No AF data (assume rare)
        no_af_variant = AnnotatedVariant(
            variant=sample_variant,
            consequence=VariantConsequence.MISSENSE,
            impact=ImpactLevel.MODERATE,
            gnomad_af=None,
        )
        assert no_af_variant.is_rare is True


class TestVariantAnnotator:
    """Tests for VariantAnnotator class."""

    @pytest.fixture
    def annotator(self):
        """Create a VariantAnnotator instance."""
        return VariantAnnotator()

    @pytest.fixture
    def sample_variant(self):
        """Create a sample variant."""
        return Variant(
            chrom="chr1",
            pos=12345,
            ref="A",
            alt="G",
            quality=50.0,
            filter_status="PASS",
            sample_id="sample1",
            genotype="0/1",
        )

    def test_annotator_creation(self, annotator):
        """Test creating an annotator."""
        assert annotator is not None
        assert annotator._vep_cache == {}
        assert annotator._cadd_scores == {}

    def test_annotate_single_variant(self, annotator, sample_variant):
        """Test annotating a single variant."""
        annotated = annotator.annotate(sample_variant)

        assert isinstance(annotated, AnnotatedVariant)
        assert annotated.variant == sample_variant
        assert annotated.consequence is not None
        assert annotated.impact is not None

    def test_annotate_batch(self, annotator):
        """Test batch annotation."""
        variants = [
            Variant(
                chrom="chr1",
                pos=100 + i * 100,
                ref="A",
                alt="G",
                quality=50.0,
                filter_status="PASS",
                sample_id=f"sample{i}",
                genotype="0/1",
            )
            for i in range(5)
        ]

        annotated_list = annotator.annotate_batch(variants)

        assert len(annotated_list) == 5
        for annotated in annotated_list:
            assert isinstance(annotated, AnnotatedVariant)

    def test_load_cadd_scores(self, annotator, tmp_path):
        """Test loading CADD scores."""
        # Create mock CADD file
        cadd_file = tmp_path / "cadd_scores.tsv"
        cadd_content = """#Chrom\tPos\tRef\tAlt\tRawScore\tPHRED
1\t12345\tA\tG\t3.5\t25.5
1\t12346\tC\tT\t2.1\t18.2
"""
        cadd_file.write_text(cadd_content)

        annotator.load_cadd_scores(cadd_file)

        # Check scores were loaded
        assert len(annotator._cadd_scores) > 0

    def test_load_gnomad_frequencies(self, annotator, tmp_path):
        """Test loading gnomAD frequencies."""
        # Create mock gnomAD file
        gnomad_file = tmp_path / "gnomad_af.tsv"
        gnomad_content = """#CHROM\tPOS\tREF\tALT\tAF
1\t12345\tA\tG\t0.0001
1\t12346\tC\tT\t0.005
"""
        gnomad_file.write_text(gnomad_content)

        annotator.load_gnomad_frequencies(gnomad_file)

        # Check frequencies were loaded
        assert len(annotator._gnomad_af) > 0

    def test_filter_by_impact(self, annotator):
        """Test filtering variants by impact."""
        variants = [
            Variant(
                chrom="chr1",
                pos=i * 100,
                ref="A",
                alt="G",
                quality=50.0,
                filter_status="PASS",
                sample_id="sample1",
                genotype="0/1",
            )
            for i in range(10)
        ]

        annotated = annotator.annotate_batch(variants)

        # Filter by HIGH impact
        high_impact = annotator.filter_by_impact(annotated, min_impact=ImpactLevel.HIGH)
        for v in high_impact:
            assert v.impact == ImpactLevel.HIGH

    def test_filter_by_consequence(self, annotator):
        """Test filtering variants by consequence."""
        variants = [
            Variant(
                chrom="chr1",
                pos=i * 100,
                ref="A",
                alt="G",
                quality=50.0,
                filter_status="PASS",
                sample_id="sample1",
                genotype="0/1",
            )
            for i in range(10)
        ]

        annotated = annotator.annotate_batch(variants)

        # Filter for missense only
        missense = annotator.filter_by_consequence(
            annotated, consequences={VariantConsequence.MISSENSE}
        )
        for v in missense:
            assert v.consequence == VariantConsequence.MISSENSE

    def test_get_lof_variants(self, annotator):
        """Test getting LOF variants."""
        variants = [
            Variant(
                chrom="chr1",
                pos=i * 100,
                ref="A",
                alt="G",
                quality=50.0,
                filter_status="PASS",
                sample_id="sample1",
                genotype="0/1",
            )
            for i in range(10)
        ]

        annotated = annotator.annotate_batch(variants)
        lof_variants = annotator.get_lof_variants(annotated)

        for v in lof_variants:
            assert v.is_lof is True

    def test_get_damaging_missense(self, annotator):
        """Test getting damaging missense variants."""
        variants = [
            Variant(
                chrom="chr1",
                pos=i * 100,
                ref="A",
                alt="G",
                quality=50.0,
                filter_status="PASS",
                sample_id="sample1",
                genotype="0/1",
            )
            for i in range(10)
        ]

        annotated = annotator.annotate_batch(variants)

        # Get damaging missense with thresholds
        damaging = annotator.get_damaging_missense(
            annotated, cadd_threshold=20.0, revel_threshold=0.5
        )

        for v in damaging:
            assert v.consequence == VariantConsequence.MISSENSE


class TestAnnotatorEdgeCases:
    """Edge case tests for VariantAnnotator."""

    def test_annotate_empty_list(self):
        """Test annotating empty variant list."""
        annotator = VariantAnnotator()
        result = annotator.annotate_batch([])
        assert result == []

    def test_variant_key_generation(self):
        """Test variant key generation is consistent."""
        annotator = VariantAnnotator()

        variant = Variant(
            chrom="chr1",
            pos=12345,
            ref="A",
            alt="G",
            quality=50.0,
            filter_status="PASS",
            sample_id="sample1",
            genotype="0/1",
        )

        key = annotator._variant_key(variant)
        assert key == "chr1:12345:A>G"

    def test_parse_vep_consequence(self):
        """Test parsing VEP consequence strings."""
        annotator = VariantAnnotator()

        # Test known consequence
        consequence = annotator._parse_consequence("missense_variant")
        assert consequence == VariantConsequence.MISSENSE

        # Test unknown consequence (should default to UNKNOWN)
        consequence = annotator._parse_consequence("unknown_type")
        assert consequence == VariantConsequence.UNKNOWN


class TestAnnotationIntegration:
    """Integration tests for annotation workflow."""

    def test_full_annotation_workflow(self, tmp_path):
        """Test complete annotation workflow."""
        # Create annotator
        annotator = VariantAnnotator()

        # Create test variants
        variants = [
            Variant(
                chrom="chr1",
                pos=12345,
                ref="A",
                alt="G",
                quality=50.0,
                filter_status="PASS",
                sample_id="sample1",
                genotype="0/1",
            ),
            Variant(
                chrom="chr2",
                pos=54321,
                ref="C",
                alt="T",
                quality=60.0,
                filter_status="PASS",
                sample_id="sample2",
                genotype="1/1",
            ),
        ]

        # Annotate
        annotated = annotator.annotate_batch(variants)

        # Verify
        assert len(annotated) == 2
        for ann in annotated:
            assert ann.consequence is not None
            assert ann.impact is not None

        # Filter
        lof = annotator.get_lof_variants(annotated)
        coding = [v for v in annotated if v.is_coding]

        # Results should be valid lists
        assert isinstance(lof, list)
        assert isinstance(coding, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
