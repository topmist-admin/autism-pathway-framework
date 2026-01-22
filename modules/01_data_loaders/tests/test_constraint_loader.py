"""
Tests for Constraint Loader
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add parent modules to path for imports
_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "modules" / "01_data_loaders"))

from constraint_loader import ConstraintLoader, GeneConstraints, SFARIGenes


class TestGeneConstraints:
    """Tests for GeneConstraints dataclass."""

    @pytest.fixture
    def sample_constraints(self):
        """Create sample gene constraints for testing."""
        return GeneConstraints(
            gene_ids=["GENE1", "GENE2", "GENE3", "GENE4"],
            pli_scores={
                "GENE1": 0.95,  # Highly constrained
                "GENE2": 0.5,   # Moderately constrained
                "GENE3": 0.1,   # Not constrained
                "GENE4": 0.99,  # Very highly constrained
            },
            loeuf_scores={
                "GENE1": 0.15,  # Low LOEUF = constrained
                "GENE2": 0.5,
                "GENE3": 1.2,   # High LOEUF = not constrained
                "GENE4": 0.1,
            },
            mis_z_scores={
                "GENE1": 3.5,
                "GENE2": 1.0,
                "GENE3": -0.5,
                "GENE4": 4.0,
            },
        )

    def test_get_pli(self, sample_constraints):
        """Test getting pLI scores."""
        assert sample_constraints.get_pli("GENE1") == 0.95
        assert sample_constraints.get_pli("NONEXISTENT") is None

    def test_get_loeuf(self, sample_constraints):
        """Test getting LOEUF scores."""
        assert sample_constraints.get_loeuf("GENE1") == 0.15
        assert sample_constraints.get_loeuf("NONEXISTENT") is None

    def test_is_constrained_pli(self, sample_constraints):
        """Test constraint detection using pLI."""
        assert sample_constraints.is_constrained("GENE1", pli_threshold=0.9)
        assert sample_constraints.is_constrained("GENE4", pli_threshold=0.9)
        assert not sample_constraints.is_constrained("GENE2", pli_threshold=0.9)
        assert not sample_constraints.is_constrained("GENE3", pli_threshold=0.9)

    def test_is_constrained_loeuf(self, sample_constraints):
        """Test constraint detection using LOEUF."""
        assert sample_constraints.is_constrained("GENE1", pli_threshold=1.0, loeuf_threshold=0.35)
        assert sample_constraints.is_constrained("GENE4", pli_threshold=1.0, loeuf_threshold=0.35)
        assert not sample_constraints.is_constrained("GENE3", pli_threshold=1.0, loeuf_threshold=0.35)

    def test_get_constrained_genes(self, sample_constraints):
        """Test getting all constrained genes."""
        constrained = sample_constraints.get_constrained_genes(pli_threshold=0.9)
        assert constrained == {"GENE1", "GENE4"}

    def test_get_constraint_percentile(self, sample_constraints):
        """Test getting constraint percentiles."""
        # GENE4 has highest pLI (0.99), should be near 100th percentile
        percentile = sample_constraints.get_constraint_percentile("GENE4", metric="pli")
        assert percentile is not None
        assert percentile > 50

        # GENE3 has lowest pLI (0.1), should be near 0th percentile
        percentile = sample_constraints.get_constraint_percentile("GENE3", metric="pli")
        assert percentile is not None
        assert percentile < 50


class TestSFARIGenes:
    """Tests for SFARIGenes dataclass."""

    @pytest.fixture
    def sample_sfari(self):
        """Create sample SFARI gene data for testing."""
        return SFARIGenes(
            gene_ids=["CHD8", "SHANK3", "NRXN1", "SYNGAP1", "DYRK1A"],
            scores={
                "CHD8": 1,      # High confidence
                "SHANK3": 1,   # High confidence
                "NRXN1": 2,    # Strong candidate
                "SYNGAP1": 2,  # Strong candidate
                "DYRK1A": 3,   # Suggestive
            },
            syndromic={
                "CHD8": False,
                "SHANK3": True,
                "NRXN1": False,
                "SYNGAP1": True,
                "DYRK1A": True,
            },
            evidence={
                "CHD8": ["Rare Single Gene Variant", "Functional"],
                "SHANK3": ["Rare Single Gene Variant", "Syndromic"],
            },
        )

    def test_get_score(self, sample_sfari):
        """Test getting SFARI scores."""
        assert sample_sfari.get_score("CHD8") == 1
        assert sample_sfari.get_score("NRXN1") == 2
        assert sample_sfari.get_score("NONEXISTENT") is None

    def test_is_sfari_gene(self, sample_sfari):
        """Test SFARI gene detection."""
        assert sample_sfari.is_sfari_gene("CHD8")
        assert not sample_sfari.is_sfari_gene("NONEXISTENT")

    def test_is_high_confidence(self, sample_sfari):
        """Test high confidence detection."""
        assert sample_sfari.is_high_confidence("CHD8")
        assert sample_sfari.is_high_confidence("SHANK3")
        assert not sample_sfari.is_high_confidence("NRXN1")

    def test_is_syndromic(self, sample_sfari):
        """Test syndromic detection."""
        assert sample_sfari.is_syndromic("SHANK3")
        assert not sample_sfari.is_syndromic("CHD8")

    def test_get_genes_by_score(self, sample_sfari):
        """Test getting genes by score threshold."""
        # Score <= 1 (high confidence only)
        high_conf = sample_sfari.get_genes_by_score(max_score=1)
        assert high_conf == {"CHD8", "SHANK3"}

        # Score <= 2 (high confidence + strong candidate)
        strong = sample_sfari.get_genes_by_score(max_score=2)
        assert strong == {"CHD8", "SHANK3", "NRXN1", "SYNGAP1"}

    def test_get_high_confidence_genes(self, sample_sfari):
        """Test getting high confidence genes."""
        high_conf = sample_sfari.get_high_confidence_genes()
        assert high_conf == {"CHD8", "SHANK3"}

    def test_get_syndromic_genes(self, sample_sfari):
        """Test getting syndromic genes."""
        syndromic = sample_sfari.get_syndromic_genes()
        assert syndromic == {"SHANK3", "SYNGAP1", "DYRK1A"}

    def test_contains(self, sample_sfari):
        """Test __contains__ method."""
        assert "CHD8" in sample_sfari
        assert "NONEXISTENT" not in sample_sfari


class TestConstraintLoader:
    """Tests for ConstraintLoader."""

    @pytest.fixture
    def sample_gnomad_file(self):
        """Create a sample gnomAD constraint file for testing."""
        content = """gene\ttranscript\tpLI\toe_lof_upper\tmis_z\tsyn_z
GENE1\tENST001\t0.95\t0.15\t3.5\t0.1
GENE2\tENST002\t0.5\t0.5\t1.0\t0.2
GENE3\tENST003\t0.1\t1.2\t-0.5\t-0.1
GENE4\tENST004\t0.99\t0.1\t4.0\t0.3
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            return f.name

    @pytest.fixture
    def sample_sfari_file(self):
        """Create a sample SFARI genes file for testing."""
        content = """gene-symbol,gene-score,syndromic,genetic-category
CHD8,1,0,Rare Single Gene Variant
SHANK3,1,1,Rare Single Gene Variant
NRXN1,2,0,Rare Single Gene Variant
SYNGAP1,2,1,Rare Single Gene Variant
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            return f.name

    def test_load_gnomad_constraints(self, sample_gnomad_file):
        """Test loading gnomAD constraint file."""
        loader = ConstraintLoader()
        constraints = loader.load_gnomad_constraints(sample_gnomad_file)

        assert len(constraints) == 4
        assert constraints.get_pli("GENE1") == 0.95
        assert constraints.get_loeuf("GENE1") == 0.15
        assert constraints.get_mis_z("GENE1") == 3.5

        os.unlink(sample_gnomad_file)

    def test_load_sfari_genes(self, sample_sfari_file):
        """Test loading SFARI genes file."""
        loader = ConstraintLoader()
        sfari = loader.load_sfari_genes(sample_sfari_file)

        assert len(sfari) == 4
        assert sfari.get_score("CHD8") == 1
        assert sfari.is_syndromic("SHANK3")

        os.unlink(sample_sfari_file)

    def test_get_constrained_genes(self, sample_gnomad_file):
        """Test getting constrained genes from loader."""
        loader = ConstraintLoader()
        loader.load_gnomad_constraints(sample_gnomad_file)

        constrained = loader.get_constrained_genes(pli_threshold=0.9)
        assert set(constrained) == {"GENE1", "GENE4"}

        os.unlink(sample_gnomad_file)

    def test_get_autism_genes(self, sample_sfari_file):
        """Test getting autism genes from loader."""
        loader = ConstraintLoader()
        loader.load_sfari_genes(sample_sfari_file)

        autism_genes = loader.get_autism_genes(max_score=2)
        assert autism_genes == {"CHD8", "SHANK3", "NRXN1", "SYNGAP1"}

        os.unlink(sample_sfari_file)

    def test_get_constrained_autism_genes(self, sample_gnomad_file, sample_sfari_file):
        """Test getting genes that are both constrained and autism-associated."""
        loader = ConstraintLoader()
        loader.load_gnomad_constraints(sample_gnomad_file)
        loader.load_sfari_genes(sample_sfari_file)

        # Note: In this test data, SFARI genes don't overlap with gnomAD genes
        # In real data, there would be overlap
        overlap = loader.get_constrained_autism_genes()
        assert isinstance(overlap, set)

        os.unlink(sample_gnomad_file)
        os.unlink(sample_sfari_file)

    def test_create_gene_prior_weights(self, sample_gnomad_file, sample_sfari_file):
        """Test creating gene prior weights."""
        loader = ConstraintLoader()
        loader.load_gnomad_constraints(sample_gnomad_file)
        loader.load_sfari_genes(sample_sfari_file)

        weights = loader.create_gene_prior_weights(
            constraint_weight=0.5,
            sfari_weight=0.5
        )

        assert isinstance(weights, dict)
        # All weights should be between 0 and 1
        for gene, weight in weights.items():
            assert 0 <= weight <= 1

        os.unlink(sample_gnomad_file)
        os.unlink(sample_sfari_file)

    def test_file_not_found(self):
        """Test error handling for missing files."""
        loader = ConstraintLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_gnomad_constraints("/nonexistent/path.txt")

        with pytest.raises(FileNotFoundError):
            loader.load_sfari_genes("/nonexistent/path.csv")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
