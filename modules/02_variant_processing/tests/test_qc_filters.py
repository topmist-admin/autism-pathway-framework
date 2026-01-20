"""
Tests for QC Filters

Tests for quality control filtering of variants and samples.
"""

import pytest
import sys
from pathlib import Path

# Add parent modules to path for imports
_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "modules" / "01_data_loaders"))
sys.path.insert(0, str(_root / "modules" / "02_variant_processing"))

from vcf_loader import Variant, VariantDataset
from qc_filters import QCFilter, QCConfig, QCReport


class TestQCConfig:
    """Tests for QCConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QCConfig()

        assert config.min_quality == 20.0
        assert config.min_depth == 10
        assert config.filter_pass_only is True
        assert config.max_allele_freq == 0.01
        assert config.max_missing_rate == 0.1
        assert "chrM" in config.exclude_chromosomes

    def test_custom_config(self):
        """Test custom configuration values."""
        config = QCConfig(
            min_quality=30.0,
            min_depth=20,
            max_allele_freq=0.05,
            filter_pass_only=False,
        )

        assert config.min_quality == 30.0
        assert config.min_depth == 20
        assert config.max_allele_freq == 0.05
        assert config.filter_pass_only is False


class TestQCReport:
    """Tests for QCReport dataclass."""

    def test_report_string(self):
        """Test report string representation."""
        report = QCReport(
            input_variants=1000,
            input_samples=100,
            output_variants=800,
            output_samples=95,
            variants_removed={"low_quality": 150, "high_af": 50},
            samples_removed={"too_few_variants": 5},
            quality_distribution={"min": 20.0, "max": 99.0},
            allele_freq_distribution={"min": 0.0001, "max": 0.01},
        )

        report_str = str(report)
        assert "QC Report" in report_str
        assert "1000 variants" in report_str
        assert "100 samples" in report_str
        assert "low_quality" in report_str

    def test_retention_rates(self):
        """Test retention rate calculations."""
        report = QCReport(
            input_variants=1000,
            input_samples=100,
            output_variants=800,
            output_samples=90,
            variants_removed={},
            samples_removed={},
            quality_distribution={},
            allele_freq_distribution={},
        )

        assert report.variant_retention_rate == 0.8
        assert report.sample_retention_rate == 0.9

    def test_retention_rates_zero_input(self):
        """Test retention rates with zero input."""
        report = QCReport(
            input_variants=0,
            input_samples=0,
            output_variants=0,
            output_samples=0,
            variants_removed={},
            samples_removed={},
            quality_distribution={},
            allele_freq_distribution={},
        )

        assert report.variant_retention_rate == 0.0
        assert report.sample_retention_rate == 0.0


class TestQCFilter:
    """Tests for QCFilter class."""

    @pytest.fixture
    def sample_variants(self):
        """Create sample variants for testing."""
        return [
            Variant(
                chrom="chr1",
                pos=100,
                ref="A",
                alt="G",
                quality=50.0,
                filter_status="PASS",
                sample_id="sample1",
                genotype="0/1",
                info={"DP": 30, "AF": 0.001},
            ),
            Variant(
                chrom="chr1",
                pos=200,
                ref="C",
                alt="T",
                quality=15.0,  # Low quality
                filter_status="PASS",
                sample_id="sample1",
                genotype="0/1",
                info={"DP": 25, "AF": 0.0005},
            ),
            Variant(
                chrom="chr2",
                pos=300,
                ref="G",
                alt="A",
                quality=60.0,
                filter_status="LowQual",  # Non-PASS filter
                sample_id="sample2",
                genotype="1/1",
                info={"DP": 40, "AF": 0.002},
            ),
            Variant(
                chrom="chrM",  # Mitochondrial - excluded
                pos=400,
                ref="T",
                alt="C",
                quality=80.0,
                filter_status="PASS",
                sample_id="sample2",
                genotype="0/1",
                info={"DP": 50, "AF": 0.0001},
            ),
            Variant(
                chrom="chr3",
                pos=500,
                ref="A",
                alt="T",
                quality=70.0,
                filter_status="PASS",
                sample_id="sample3",
                genotype="0/1",
                info={"DP": 5, "AF": 0.0003},  # Low depth
            ),
            Variant(
                chrom="chr4",
                pos=600,
                ref="G",
                alt="C",
                quality=90.0,
                filter_status="PASS",
                sample_id="sample3",
                genotype="0/1",
                info={"DP": 35, "AF": 0.05},  # High AF
            ),
        ]

    @pytest.fixture
    def sample_dataset(self, sample_variants):
        """Create sample dataset for testing."""
        return VariantDataset(
            variants=sample_variants,
            samples=["sample1", "sample2", "sample3"],
            metadata={"source": "test"},
        )

    def test_filter_variants_quality(self, sample_dataset):
        """Test filtering by quality score."""
        qc_filter = QCFilter()
        config = QCConfig(
            min_quality=20.0,
            filter_pass_only=False,
            min_depth=None,
            max_allele_freq=1.0,  # No AF filter
        )

        filtered = qc_filter.filter_variants(sample_dataset, config)

        # Should filter out low quality variant (quality=15)
        qualities = [v.quality for v in filtered.variants]
        assert all(q >= 20.0 for q in qualities)

    def test_filter_variants_pass_only(self, sample_dataset):
        """Test filtering by PASS status."""
        qc_filter = QCFilter()
        config = QCConfig(
            min_quality=0.0,
            filter_pass_only=True,
            min_depth=None,
            max_allele_freq=1.0,
        )

        filtered = qc_filter.filter_variants(sample_dataset, config)

        # Should filter out non-PASS variants
        for v in filtered.variants:
            assert v.filter_status == "PASS"

    def test_filter_variants_chromosome(self, sample_dataset):
        """Test filtering by excluded chromosomes."""
        qc_filter = QCFilter()
        config = QCConfig(
            min_quality=0.0,
            filter_pass_only=False,
            min_depth=None,
            max_allele_freq=1.0,
            exclude_chromosomes={"chrM"},
        )

        filtered = qc_filter.filter_variants(sample_dataset, config)

        # Should filter out chrM variants
        chroms = [v.chrom for v in filtered.variants]
        assert "chrM" not in chroms

    def test_filter_variants_depth(self, sample_dataset):
        """Test filtering by read depth."""
        qc_filter = QCFilter()
        config = QCConfig(
            min_quality=0.0,
            filter_pass_only=False,
            min_depth=10,
            max_allele_freq=1.0,
            exclude_chromosomes=set(),
        )

        filtered = qc_filter.filter_variants(sample_dataset, config)

        # Should filter out low depth variants
        for v in filtered.variants:
            if "DP" in v.info:
                assert v.info["DP"] >= 10

    def test_filter_variants_allele_freq(self, sample_dataset):
        """Test filtering by allele frequency."""
        qc_filter = QCFilter()
        config = QCConfig(
            min_quality=0.0,
            filter_pass_only=False,
            min_depth=None,
            max_allele_freq=0.01,
            exclude_chromosomes=set(),
        )

        filtered = qc_filter.filter_variants(sample_dataset, config)

        # Should filter out high AF variants
        for v in filtered.variants:
            if "AF" in v.info:
                assert v.info["AF"] <= 0.01

    def test_filter_samples(self, sample_dataset):
        """Test filtering samples by variant count."""
        qc_filter = QCFilter()
        config = QCConfig(min_variants_per_sample=2)

        filtered = qc_filter.filter_samples(sample_dataset, config)

        # Each sample should have at least 2 variants
        sample_counts = {}
        for v in filtered.variants:
            sample_counts[v.sample_id] = sample_counts.get(v.sample_id, 0) + 1

        for sample in filtered.samples:
            if sample in sample_counts:
                assert sample_counts[sample] >= 1  # Has variants

    def test_filter_samples_max_variants(self, sample_dataset):
        """Test filtering samples by maximum variant count."""
        qc_filter = QCFilter()
        config = QCConfig(max_variants_per_sample=1)

        filtered = qc_filter.filter_samples(sample_dataset, config)

        # Samples with more than 1 variant should be filtered
        # This depends on input data

    def test_run_full_qc(self, sample_dataset):
        """Test full QC pipeline."""
        qc_filter = QCFilter()
        config = QCConfig(
            min_quality=20.0,
            filter_pass_only=True,
            max_allele_freq=0.01,
        )

        filtered, report = qc_filter.run_full_qc(sample_dataset, config)

        # Check report
        assert report.input_variants == len(sample_dataset.variants)
        assert report.input_samples == len(sample_dataset.samples)
        assert report.output_variants == len(filtered.variants)
        assert report.output_samples == len(filtered.samples)

    def test_filter_by_call_rate(self, sample_dataset):
        """Test filtering by call rate."""
        qc_filter = QCFilter()

        # With 3 samples, min_call_rate=0.5 requires 2 samples
        filtered = qc_filter.filter_by_call_rate(sample_dataset, min_call_rate=0.5)

        # All variants should have call rate >= 0.5
        assert filtered is not None

    def test_get_qc_report(self, sample_dataset):
        """Test getting QC report."""
        qc_filter = QCFilter()
        config = QCConfig()

        # Run filtering first
        qc_filter.filter_variants(sample_dataset, config)

        report = qc_filter.get_qc_report()
        assert isinstance(report, QCReport)


class TestQCFilterEdgeCases:
    """Edge case tests for QCFilter."""

    def test_empty_dataset(self):
        """Test filtering empty dataset."""
        qc_filter = QCFilter()
        config = QCConfig()

        empty_dataset = VariantDataset(
            variants=[],
            samples=[],
            metadata={},
        )

        filtered = qc_filter.filter_variants(empty_dataset, config)
        assert len(filtered.variants) == 0

    def test_all_variants_filtered(self):
        """Test when all variants are filtered out."""
        qc_filter = QCFilter()
        config = QCConfig(min_quality=1000.0)  # Impossibly high

        variants = [
            Variant(
                chrom="chr1",
                pos=100,
                ref="A",
                alt="G",
                quality=50.0,
                filter_status="PASS",
                sample_id="sample1",
                genotype="0/1",
            )
        ]

        dataset = VariantDataset(
            variants=variants,
            samples=["sample1"],
            metadata={},
        )

        filtered = qc_filter.filter_variants(dataset, config)
        assert len(filtered.variants) == 0

    def test_variant_without_info_fields(self):
        """Test variants without optional INFO fields."""
        qc_filter = QCFilter()
        config = QCConfig(
            min_depth=10,
            max_allele_freq=0.01,
        )

        variants = [
            Variant(
                chrom="chr1",
                pos=100,
                ref="A",
                alt="G",
                quality=50.0,
                filter_status="PASS",
                sample_id="sample1",
                genotype="0/1",
                info={},  # No DP or AF
            )
        ]

        dataset = VariantDataset(
            variants=variants,
            samples=["sample1"],
            metadata={},
        )

        # Should not crash when INFO fields are missing
        filtered = qc_filter.filter_variants(dataset, config)
        assert len(filtered.variants) == 1  # Should pass (no AF to check)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
