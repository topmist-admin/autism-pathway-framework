"""
Tests for Gene Burden Calculator

Tests for gene-level burden score computation from annotated variants.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent modules to path for imports
_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "modules" / "01_data_loaders"))
sys.path.insert(0, str(_root / "modules" / "02_variant_processing"))

from vcf_loader import Variant
from annotation import (
    AnnotatedVariant,
    VariantConsequence,
    ImpactLevel,
)
from gene_burden import (
    GeneBurdenCalculator,
    GeneBurdenMatrix,
    WeightConfig,
)


class TestWeightConfig:
    """Tests for WeightConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WeightConfig()

        assert config.use_cadd_weighting is True
        assert config.cadd_threshold == 20.0
        assert config.use_revel_weighting is True
        assert config.revel_threshold == 0.5
        assert config.min_impact == "MODERATE"
        assert config.aggregation == "weighted_sum"

    def test_consequence_weights(self):
        """Test default consequence weights."""
        config = WeightConfig()

        # LOF should have highest weight
        assert config.consequence_weights["frameshift_variant"] == 1.0
        assert config.consequence_weights["stop_gained"] == 1.0

        # Missense moderate weight
        assert config.consequence_weights["missense_variant"] == 0.5

        # Synonymous zero weight
        assert config.consequence_weights["synonymous_variant"] == 0.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = WeightConfig(
            cadd_threshold=25.0,
            revel_threshold=0.7,
            use_af_weighting=True,
            aggregation="max",
        )

        assert config.cadd_threshold == 25.0
        assert config.revel_threshold == 0.7
        assert config.use_af_weighting is True
        assert config.aggregation == "max"


class TestGeneBurdenMatrix:
    """Tests for GeneBurdenMatrix dataclass."""

    @pytest.fixture
    def sample_matrix(self):
        """Create a sample burden matrix."""
        samples = ["sample1", "sample2", "sample3"]
        genes = ["GENE_A", "GENE_B", "GENE_C"]
        scores = np.array([
            [1.0, 0.5, 0.0],  # sample1
            [0.0, 1.5, 0.3],  # sample2
            [0.8, 0.0, 1.2],  # sample3
        ])

        return GeneBurdenMatrix(
            samples=samples,
            genes=genes,
            scores=scores,
        )

    def test_matrix_creation(self, sample_matrix):
        """Test creating a burden matrix."""
        assert sample_matrix.n_samples == 3
        assert sample_matrix.n_genes == 3
        assert sample_matrix.shape == (3, 3)

    def test_index_mappings(self, sample_matrix):
        """Test sample and gene index mappings."""
        assert sample_matrix.sample_index["sample1"] == 0
        assert sample_matrix.sample_index["sample2"] == 1
        assert sample_matrix.gene_index["GENE_A"] == 0
        assert sample_matrix.gene_index["GENE_C"] == 2

    def test_get_sample(self, sample_matrix):
        """Test getting burden for a specific sample."""
        burden = sample_matrix.get_sample("sample1")

        assert burden["GENE_A"] == 1.0
        assert burden["GENE_B"] == 0.5
        assert "GENE_C" not in burden  # Zero score not included

    def test_get_sample_not_found(self, sample_matrix):
        """Test getting burden for non-existent sample."""
        with pytest.raises(KeyError):
            sample_matrix.get_sample("nonexistent")

    def test_get_gene(self, sample_matrix):
        """Test getting scores for a specific gene."""
        scores = sample_matrix.get_gene("GENE_A")

        assert len(scores) == 3
        assert scores[0] == 1.0
        assert scores[1] == 0.0
        assert scores[2] == 0.8

    def test_get_gene_not_found(self, sample_matrix):
        """Test getting scores for non-existent gene."""
        with pytest.raises(KeyError):
            sample_matrix.get_gene("NONEXISTENT")

    def test_get_score(self, sample_matrix):
        """Test getting score for specific sample and gene."""
        score = sample_matrix.get_score("sample1", "GENE_A")
        assert score == 1.0

        score = sample_matrix.get_score("sample2", "GENE_B")
        assert score == 1.5

    def test_get_score_missing(self, sample_matrix):
        """Test getting score for missing sample/gene."""
        score = sample_matrix.get_score("nonexistent", "GENE_A")
        assert score == 0.0

        score = sample_matrix.get_score("sample1", "NONEXISTENT")
        assert score == 0.0

    def test_to_dataframe(self, sample_matrix):
        """Test converting to pandas DataFrame."""
        df = sample_matrix.to_dataframe()

        assert df.shape == (3, 3)
        assert list(df.index) == ["sample1", "sample2", "sample3"]
        assert list(df.columns) == ["GENE_A", "GENE_B", "GENE_C"]
        assert df.loc["sample1", "GENE_A"] == 1.0

    def test_to_sparse(self, sample_matrix):
        """Test converting to sparse matrix."""
        sparse = sample_matrix.to_sparse()

        assert sparse.shape == (3, 3)
        # Check a specific value
        assert sparse[0, 0] == 1.0

    def test_get_nonzero_genes(self, sample_matrix):
        """Test getting genes with nonzero burden."""
        nonzero = sample_matrix.get_nonzero_genes()

        assert "GENE_A" in nonzero
        assert "GENE_B" in nonzero
        assert "GENE_C" in nonzero

    def test_filter_genes(self, sample_matrix):
        """Test filtering to subset of genes."""
        filtered = sample_matrix.filter_genes({"GENE_A", "GENE_B"})

        assert filtered.n_genes == 2
        assert "GENE_A" in filtered.genes
        assert "GENE_B" in filtered.genes
        assert "GENE_C" not in filtered.genes

    def test_filter_samples(self, sample_matrix):
        """Test filtering to subset of samples."""
        filtered = sample_matrix.filter_samples({"sample1", "sample3"})

        assert filtered.n_samples == 2
        assert "sample1" in filtered.samples
        assert "sample3" in filtered.samples
        assert "sample2" not in filtered.samples

    def test_normalize_zscore(self, sample_matrix):
        """Test z-score normalization."""
        normalized = sample_matrix.normalize(method="zscore")

        # Z-scores should have mean ~0 and std ~1 per gene
        assert normalized.shape == sample_matrix.shape

        # Check that normalization was applied
        gene_means = np.mean(normalized.scores, axis=0)
        for mean in gene_means:
            assert abs(mean) < 1e-10  # Should be ~0

    def test_normalize_minmax(self, sample_matrix):
        """Test min-max normalization."""
        normalized = sample_matrix.normalize(method="minmax")

        # Values should be in [0, 1]
        assert np.min(normalized.scores) >= 0
        assert np.max(normalized.scores) <= 1

    def test_normalize_rank(self, sample_matrix):
        """Test rank normalization."""
        normalized = sample_matrix.normalize(method="rank")

        # Ranks should be in (0, 1]
        assert np.min(normalized.scores) > 0
        assert np.max(normalized.scores) <= 1

    def test_normalize_invalid_method(self, sample_matrix):
        """Test invalid normalization method."""
        with pytest.raises(ValueError):
            sample_matrix.normalize(method="invalid")


class TestGeneBurdenCalculator:
    """Tests for GeneBurdenCalculator class."""

    @pytest.fixture
    def sample_annotated_variants(self):
        """Create sample annotated variants."""
        variants = []

        # Sample 1, Gene A - frameshift (LOF)
        v1 = Variant(
            chrom="chr1", pos=100, ref="A", alt="AG",
            quality=50.0, filter_status="PASS",
            sample_id="sample1", genotype="0/1",
        )
        variants.append(AnnotatedVariant(
            variant=v1,
            gene_id="GENE_A",
            gene_symbol="GeneA",
            consequence=VariantConsequence.FRAMESHIFT,
            impact=ImpactLevel.HIGH,
            cadd_phred=30.0,
        ))

        # Sample 1, Gene B - missense
        v2 = Variant(
            chrom="chr2", pos=200, ref="C", alt="T",
            quality=60.0, filter_status="PASS",
            sample_id="sample1", genotype="0/1",
        )
        variants.append(AnnotatedVariant(
            variant=v2,
            gene_id="GENE_B",
            gene_symbol="GeneB",
            consequence=VariantConsequence.MISSENSE,
            impact=ImpactLevel.MODERATE,
            cadd_phred=25.0,
            revel_score=0.7,
        ))

        # Sample 2, Gene A - stop gained (LOF)
        v3 = Variant(
            chrom="chr1", pos=150, ref="G", alt="T",
            quality=70.0, filter_status="PASS",
            sample_id="sample2", genotype="1/1",
        )
        variants.append(AnnotatedVariant(
            variant=v3,
            gene_id="GENE_A",
            gene_symbol="GeneA",
            consequence=VariantConsequence.STOP_GAINED,
            impact=ImpactLevel.HIGH,
            cadd_phred=35.0,
        ))

        # Sample 2, Gene C - synonymous (should be filtered)
        v4 = Variant(
            chrom="chr3", pos=300, ref="A", alt="G",
            quality=55.0, filter_status="PASS",
            sample_id="sample2", genotype="0/1",
        )
        variants.append(AnnotatedVariant(
            variant=v4,
            gene_id="GENE_C",
            gene_symbol="GeneC",
            consequence=VariantConsequence.SYNONYMOUS,
            impact=ImpactLevel.LOW,
            cadd_phred=5.0,
        ))

        return variants

    def test_calculator_creation(self):
        """Test creating a calculator."""
        calculator = GeneBurdenCalculator()
        assert calculator.config is not None

    def test_calculator_with_custom_config(self):
        """Test creating calculator with custom config."""
        config = WeightConfig(cadd_threshold=25.0)
        calculator = GeneBurdenCalculator(config)
        assert calculator.config.cadd_threshold == 25.0

    def test_compute_burden(self, sample_annotated_variants):
        """Test computing gene burden."""
        calculator = GeneBurdenCalculator()
        matrix = calculator.compute(sample_annotated_variants)

        assert isinstance(matrix, GeneBurdenMatrix)
        assert matrix.n_samples == 2  # sample1, sample2
        assert matrix.n_genes >= 2  # At least GENE_A, GENE_B

    def test_compute_burden_weighted_sum(self, sample_annotated_variants):
        """Test weighted sum aggregation."""
        config = WeightConfig(aggregation="weighted_sum")
        calculator = GeneBurdenCalculator(config)
        matrix = calculator.compute(sample_annotated_variants)

        # Sample 1 should have scores for GENE_A and GENE_B
        sample1_burden = matrix.get_sample("sample1")
        assert "GENE_A" in sample1_burden
        assert "GENE_B" in sample1_burden

    def test_compute_burden_max_aggregation(self, sample_annotated_variants):
        """Test max aggregation."""
        config = WeightConfig(aggregation="max")
        calculator = GeneBurdenCalculator(config)
        matrix = calculator.compute(sample_annotated_variants)

        assert matrix is not None

    def test_compute_burden_count_aggregation(self, sample_annotated_variants):
        """Test count aggregation."""
        config = WeightConfig(aggregation="count")
        calculator = GeneBurdenCalculator(config)
        matrix = calculator.compute(sample_annotated_variants)

        # Scores should be integer counts
        for score in matrix.scores.flatten():
            if score > 0:
                assert score == int(score)

    def test_filter_variants_by_impact(self, sample_annotated_variants):
        """Test variant filtering by impact."""
        config = WeightConfig(min_impact="HIGH")
        calculator = GeneBurdenCalculator(config)
        matrix = calculator.compute(sample_annotated_variants)

        # Only HIGH impact variants should be included
        # GENE_B (missense/MODERATE) should be excluded
        assert "GENE_B" not in matrix.genes or matrix.get_score("sample1", "GENE_B") == 0

    def test_filter_synonymous(self, sample_annotated_variants):
        """Test synonymous variant filtering."""
        config = WeightConfig(include_synonymous=False)
        calculator = GeneBurdenCalculator(config)
        matrix = calculator.compute(sample_annotated_variants)

        # Synonymous variant in GENE_C should be filtered
        score = matrix.get_score("sample2", "GENE_C")
        assert score == 0.0

    def test_compute_lof_burden(self, sample_annotated_variants):
        """Test LOF-only burden computation."""
        calculator = GeneBurdenCalculator()
        matrix = calculator.compute_lof_burden(sample_annotated_variants)

        # Should only include LOF variants
        # Missense in GENE_B should be excluded
        assert matrix is not None

    def test_compute_missense_burden(self, sample_annotated_variants):
        """Test missense-only burden computation."""
        calculator = GeneBurdenCalculator()
        matrix = calculator.compute_missense_burden(sample_annotated_variants)

        # Should only include missense variants
        assert matrix is not None

    def test_combine_burdens(self):
        """Test combining multiple burden matrices."""
        # Create two matrices with same samples/genes
        samples = ["s1", "s2"]
        genes = ["g1", "g2"]

        matrix1 = GeneBurdenMatrix(
            samples=samples,
            genes=genes,
            scores=np.array([[1.0, 0.5], [0.3, 0.7]]),
        )

        matrix2 = GeneBurdenMatrix(
            samples=samples,
            genes=genes,
            scores=np.array([[0.5, 0.2], [0.1, 0.3]]),
        )

        combined = GeneBurdenCalculator.combine_burdens([matrix1, matrix2])

        # Scores should be summed
        assert combined.get_score("s1", "g1") == 1.5
        assert combined.get_score("s1", "g2") == 0.7

    def test_combine_burdens_with_weights(self):
        """Test combining burdens with weights."""
        samples = ["s1", "s2"]
        genes = ["g1"]

        matrix1 = GeneBurdenMatrix(
            samples=samples,
            genes=genes,
            scores=np.array([[1.0], [0.5]]),
        )

        matrix2 = GeneBurdenMatrix(
            samples=samples,
            genes=genes,
            scores=np.array([[0.5], [0.5]]),
        )

        combined = GeneBurdenCalculator.combine_burdens(
            [matrix1, matrix2], weights=[2.0, 1.0]
        )

        # Score = 2.0 * 1.0 + 1.0 * 0.5 = 2.5
        assert combined.get_score("s1", "g1") == 2.5

    def test_combine_burdens_empty_list(self):
        """Test combining empty burden list."""
        with pytest.raises(ValueError):
            GeneBurdenCalculator.combine_burdens([])

    def test_combine_burdens_single_matrix(self):
        """Test combining single burden matrix."""
        matrix = GeneBurdenMatrix(
            samples=["s1"],
            genes=["g1"],
            scores=np.array([[1.0]]),
        )

        combined = GeneBurdenCalculator.combine_burdens([matrix])
        assert combined == matrix

    def test_contributing_variants_tracked(self, sample_annotated_variants):
        """Test that contributing variants are tracked."""
        calculator = GeneBurdenCalculator()
        matrix = calculator.compute(sample_annotated_variants)

        # Check contributing variants are recorded
        assert matrix.contributing_variants is not None

    def test_compute_with_specified_samples(self, sample_annotated_variants):
        """Test computing burden with specified sample list."""
        calculator = GeneBurdenCalculator()
        matrix = calculator.compute(
            sample_annotated_variants,
            samples=["sample1", "sample2", "sample3"]  # sample3 has no variants
        )

        assert "sample3" in matrix.samples
        # sample3 should have all zero scores
        sample3_burden = matrix.scores[matrix.sample_index["sample3"], :]
        assert np.all(sample3_burden == 0)


class TestGeneBurdenEdgeCases:
    """Edge case tests for gene burden computation."""

    def test_empty_variant_list(self):
        """Test computing burden from empty variant list."""
        calculator = GeneBurdenCalculator()
        matrix = calculator.compute([])

        assert matrix.n_samples == 0
        assert matrix.n_genes == 0

    def test_variants_without_gene_id(self):
        """Test handling variants without gene ID."""
        variant = Variant(
            chrom="chr1", pos=100, ref="A", alt="G",
            quality=50.0, filter_status="PASS",
            sample_id="sample1", genotype="0/1",
        )
        annotated = AnnotatedVariant(
            variant=variant,
            gene_id=None,  # No gene ID
            consequence=VariantConsequence.INTERGENIC,
            impact=ImpactLevel.MODIFIER,
        )

        calculator = GeneBurdenCalculator()
        matrix = calculator.compute([annotated])

        # Should handle gracefully
        assert matrix.n_genes == 0

    def test_all_variants_filtered(self):
        """Test when all variants are filtered."""
        variant = Variant(
            chrom="chr1", pos=100, ref="A", alt="G",
            quality=50.0, filter_status="PASS",
            sample_id="sample1", genotype="0/1",
        )
        annotated = AnnotatedVariant(
            variant=variant,
            gene_id="GENE_A",
            consequence=VariantConsequence.SYNONYMOUS,
            impact=ImpactLevel.LOW,
        )

        config = WeightConfig(
            include_synonymous=False,
            min_impact="MODERATE",
        )
        calculator = GeneBurdenCalculator(config)
        matrix = calculator.compute([annotated])

        # All variants filtered
        assert matrix.n_genes == 0 or np.all(matrix.scores == 0)

    def test_cadd_weighting(self):
        """Test CADD-based weighting."""
        variant = Variant(
            chrom="chr1", pos=100, ref="A", alt="G",
            quality=50.0, filter_status="PASS",
            sample_id="sample1", genotype="0/1",
        )
        annotated = AnnotatedVariant(
            variant=variant,
            gene_id="GENE_A",
            consequence=VariantConsequence.MISSENSE,
            impact=ImpactLevel.MODERATE,
            cadd_phred=30.0,
        )

        config = WeightConfig(
            use_cadd_weighting=True,
            cadd_weight_scale=0.05,
        )
        calculator = GeneBurdenCalculator(config)
        matrix = calculator.compute([annotated])

        # Weight should be CADD * scale = 30 * 0.05 = 1.5
        score = matrix.get_score("sample1", "GENE_A")
        assert score == pytest.approx(1.5, rel=0.1)

    def test_af_weighting(self):
        """Test allele frequency weighting."""
        variant = Variant(
            chrom="chr1", pos=100, ref="A", alt="G",
            quality=50.0, filter_status="PASS",
            sample_id="sample1", genotype="0/1",
        )
        annotated = AnnotatedVariant(
            variant=variant,
            gene_id="GENE_A",
            consequence=VariantConsequence.FRAMESHIFT,
            impact=ImpactLevel.HIGH,
            gnomad_af=0.001,
        )

        config = WeightConfig(
            use_af_weighting=True,
            af_weight_beta=1.0,
            use_cadd_weighting=False,
            use_revel_weighting=False,
        )
        calculator = GeneBurdenCalculator(config)
        matrix = calculator.compute([annotated])

        # Base weight = 1.0, AF weight = (1 - 0.001)^1 ≈ 0.999
        score = matrix.get_score("sample1", "GENE_A")
        assert score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
