"""
Tests for VCF Loader
"""

import pytest
import tempfile
import os
from pathlib import Path

from ..vcf_loader import VCFLoader, Variant, VariantDataset, ValidationReport


class TestVariant:
    """Tests for Variant dataclass."""

    def test_snv_detection(self):
        """Test SNV detection."""
        variant = Variant(
            chrom="chr1", pos=100, ref="A", alt="G",
            sample_id="S1", genotype="0/1", quality=30.0
        )
        assert variant.is_snv
        assert not variant.is_indel
        assert variant.variant_type == "SNV"

    def test_insertion_detection(self):
        """Test insertion detection."""
        variant = Variant(
            chrom="chr1", pos=100, ref="A", alt="ATG",
            sample_id="S1", genotype="0/1", quality=30.0
        )
        assert not variant.is_snv
        assert variant.is_indel
        assert variant.is_insertion
        assert not variant.is_deletion
        assert variant.variant_type == "INS"

    def test_deletion_detection(self):
        """Test deletion detection."""
        variant = Variant(
            chrom="chr1", pos=100, ref="ATG", alt="A",
            sample_id="S1", genotype="0/1", quality=30.0
        )
        assert not variant.is_snv
        assert variant.is_indel
        assert variant.is_deletion
        assert not variant.is_insertion
        assert variant.variant_type == "DEL"

    def test_variant_hash(self):
        """Test variant hashing for set operations."""
        v1 = Variant(
            chrom="chr1", pos=100, ref="A", alt="G",
            sample_id="S1", genotype="0/1", quality=30.0
        )
        v2 = Variant(
            chrom="chr1", pos=100, ref="A", alt="G",
            sample_id="S1", genotype="0/1", quality=30.0
        )
        v3 = Variant(
            chrom="chr1", pos=100, ref="A", alt="G",
            sample_id="S2", genotype="0/1", quality=30.0
        )

        # Same variant in same sample should be equal
        assert hash(v1) == hash(v2)
        # Different sample should be different
        assert hash(v1) != hash(v3)


class TestVariantDataset:
    """Tests for VariantDataset."""

    def test_get_variants_by_sample(self):
        """Test filtering by sample."""
        variants = [
            Variant("chr1", 100, "A", "G", "S1", "0/1", 30.0),
            Variant("chr1", 200, "C", "T", "S1", "0/1", 30.0),
            Variant("chr1", 300, "G", "A", "S2", "0/1", 30.0),
        ]
        dataset = VariantDataset(variants=variants, samples=["S1", "S2"])

        s1_variants = dataset.get_variants_by_sample("S1")
        assert len(s1_variants) == 2

        s2_variants = dataset.get_variants_by_sample("S2")
        assert len(s2_variants) == 1

    def test_get_variants_by_chrom(self):
        """Test filtering by chromosome."""
        variants = [
            Variant("chr1", 100, "A", "G", "S1", "0/1", 30.0),
            Variant("chr2", 200, "C", "T", "S1", "0/1", 30.0),
            Variant("chr1", 300, "G", "A", "S1", "0/1", 30.0),
        ]
        dataset = VariantDataset(variants=variants, samples=["S1"])

        chr1_variants = dataset.get_variants_by_chrom("chr1")
        assert len(chr1_variants) == 2

    def test_get_variants_by_region(self):
        """Test filtering by genomic region."""
        variants = [
            Variant("chr1", 100, "A", "G", "S1", "0/1", 30.0),
            Variant("chr1", 200, "C", "T", "S1", "0/1", 30.0),
            Variant("chr1", 500, "G", "A", "S1", "0/1", 30.0),
        ]
        dataset = VariantDataset(variants=variants, samples=["S1"])

        region_variants = dataset.get_variants_by_region("chr1", 50, 250)
        assert len(region_variants) == 2

    def test_iteration(self):
        """Test dataset iteration."""
        variants = [
            Variant("chr1", 100, "A", "G", "S1", "0/1", 30.0),
            Variant("chr1", 200, "C", "T", "S1", "0/1", 30.0),
        ]
        dataset = VariantDataset(variants=variants, samples=["S1"])

        assert len(dataset) == 2
        for v in dataset:
            assert isinstance(v, Variant)


class TestVCFLoader:
    """Tests for VCFLoader."""

    @pytest.fixture
    def sample_vcf(self):
        """Create a sample VCF file for testing."""
        vcf_content = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=248956422>
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample1\tSample2
chr1\t100\trs123\tA\tG\t30\tPASS\tDP=50\tGT:GQ\t0/1:30\t0/0:40
chr1\t200\t.\tC\tT\t25\tPASS\tDP=40\tGT:GQ\t1/1:35\t0/1:25
chr1\t300\t.\tG\tA,C\t35\t.\tDP=60\tGT:GQ\t1/2:45\t0/1:30
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            f.write(vcf_content)
            return f.name

    def test_load_vcf(self, sample_vcf):
        """Test loading a VCF file."""
        loader = VCFLoader()
        dataset = loader.load(sample_vcf)

        assert len(dataset.samples) == 2
        assert "Sample1" in dataset.samples
        assert "Sample2" in dataset.samples
        assert len(dataset.variants) > 0

        # Clean up
        os.unlink(sample_vcf)

    def test_load_with_quality_filter(self, sample_vcf):
        """Test loading with quality filter."""
        loader = VCFLoader(min_quality=28)
        dataset = loader.load(sample_vcf)

        # All variants should have quality >= 28
        for variant in dataset.variants:
            assert variant.quality >= 28

        os.unlink(sample_vcf)

    def test_load_with_pass_filter(self, sample_vcf):
        """Test loading with PASS filter only."""
        loader = VCFLoader(filter_pass_only=True)
        dataset = loader.load(sample_vcf)

        # All variants should have PASS filter
        for variant in dataset.variants:
            assert variant.filter_status == "PASS"

        os.unlink(sample_vcf)

    def test_multiallelic_handling(self, sample_vcf):
        """Test handling of multiallelic variants."""
        loader = VCFLoader()
        dataset = loader.load(sample_vcf)

        # The third variant (G -> A,C with genotype 1/2) should produce variants
        # for both alternate alleles
        pos_300_variants = [v for v in dataset.variants if v.pos == 300]
        assert len(pos_300_variants) > 0

        os.unlink(sample_vcf)

    def test_validation(self, sample_vcf):
        """Test dataset validation."""
        loader = VCFLoader()
        dataset = loader.load(sample_vcf)
        report = loader.validate(dataset)

        assert isinstance(report, ValidationReport)
        assert report.n_samples == 2
        assert report.n_variants > 0
        assert "SNV" in report.variant_types

        os.unlink(sample_vcf)

    def test_file_not_found(self):
        """Test error handling for missing file."""
        loader = VCFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/to/file.vcf")

    def test_info_parsing(self, sample_vcf):
        """Test INFO field parsing."""
        loader = VCFLoader(include_info=True)
        dataset = loader.load(sample_vcf)

        # Check that INFO fields are parsed
        for variant in dataset.variants:
            if "DP" in variant.info:
                assert isinstance(variant.info["DP"], int)

        os.unlink(sample_vcf)


class TestVCFLoaderGzip:
    """Tests for gzipped VCF loading."""

    @pytest.fixture
    def sample_vcf_gz(self):
        """Create a gzipped sample VCF file for testing."""
        import gzip

        vcf_content = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample1
chr1\t100\t.\tA\tG\t30\tPASS\t.\tGT\t0/1
"""
        with tempfile.NamedTemporaryFile(suffix=".vcf.gz", delete=False) as f:
            with gzip.open(f.name, "wt") as gz:
                gz.write(vcf_content)
            return f.name

    def test_load_gzipped_vcf(self, sample_vcf_gz):
        """Test loading a gzipped VCF file."""
        loader = VCFLoader()
        dataset = loader.load(sample_vcf_gz)

        assert len(dataset.samples) == 1
        assert len(dataset.variants) > 0

        os.unlink(sample_vcf_gz)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
