"""
Tests for Pathway Loader
"""

import pytest
import tempfile
import os
from pathlib import Path

from ..pathway_loader import PathwayLoader, PathwayDatabase


class TestPathwayDatabase:
    """Tests for PathwayDatabase dataclass."""

    @pytest.fixture
    def sample_database(self):
        """Create a sample pathway database for testing."""
        return PathwayDatabase(
            pathways={
                "P1": {"GENE1", "GENE2", "GENE3"},
                "P2": {"GENE2", "GENE4"},
                "P3": {"GENE5", "GENE6", "GENE7", "GENE8"},
            },
            pathway_names={
                "P1": "Pathway One",
                "P2": "Pathway Two",
                "P3": "Pathway Three",
            },
            source="test",
        )

    def test_get_pathway_genes(self, sample_database):
        """Test getting genes in a pathway."""
        genes = sample_database.get_pathway_genes("P1")
        assert genes == {"GENE1", "GENE2", "GENE3"}

        # Non-existent pathway
        genes = sample_database.get_pathway_genes("P99")
        assert genes == set()

    def test_get_gene_pathways(self, sample_database):
        """Test getting pathways for a gene."""
        # GENE2 is in P1 and P2
        pathways = sample_database.get_gene_pathways("GENE2")
        assert pathways == {"P1", "P2"}

        # GENE5 is only in P3
        pathways = sample_database.get_gene_pathways("GENE5")
        assert pathways == {"P3"}

        # Non-existent gene
        pathways = sample_database.get_gene_pathways("GENE99")
        assert pathways == set()

    def test_get_pathway_size(self, sample_database):
        """Test getting pathway size."""
        assert sample_database.get_pathway_size("P1") == 3
        assert sample_database.get_pathway_size("P2") == 2
        assert sample_database.get_pathway_size("P3") == 4

    def test_filter_by_size(self, sample_database):
        """Test filtering pathways by size."""
        # Keep only pathways with 3+ genes
        filtered = sample_database.filter_by_size(min_size=3)
        assert len(filtered) == 2
        assert "P1" in filtered.pathways
        assert "P3" in filtered.pathways
        assert "P2" not in filtered.pathways

        # Keep only pathways with 2-3 genes
        filtered = sample_database.filter_by_size(min_size=2, max_size=3)
        assert len(filtered) == 2
        assert "P1" in filtered.pathways
        assert "P2" in filtered.pathways
        assert "P3" not in filtered.pathways

    def test_get_all_genes(self, sample_database):
        """Test getting all genes across pathways."""
        all_genes = sample_database.get_all_genes()
        expected = {"GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6", "GENE7", "GENE8"}
        assert all_genes == expected

    def test_len(self, sample_database):
        """Test database length."""
        assert len(sample_database) == 3


class TestPathwayLoader:
    """Tests for PathwayLoader."""

    @pytest.fixture
    def sample_gmt(self):
        """Create a sample GMT file for testing."""
        gmt_content = """PATHWAY_A\tDescription A\tGENE1\tGENE2\tGENE3
PATHWAY_B\tDescription B\tGENE2\tGENE4\tGENE5
PATHWAY_C\tDescription C\tGENE6\tGENE7
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gmt", delete=False) as f:
            f.write(gmt_content)
            return f.name

    def test_load_gmt(self, sample_gmt):
        """Test loading GMT file."""
        loader = PathwayLoader()
        db = loader.load_gmt(sample_gmt)

        assert len(db) == 3
        assert "PATHWAY_A" in db.pathways
        assert db.get_pathway_genes("PATHWAY_A") == {"GENE1", "GENE2", "GENE3"}
        assert db.pathway_descriptions["PATHWAY_A"] == "Description A"

        os.unlink(sample_gmt)

    def test_load_gmt_file_not_found(self):
        """Test error handling for missing file."""
        loader = PathwayLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_gmt("/nonexistent/path/to/file.gmt")

    def test_merge_databases(self, sample_gmt):
        """Test merging multiple databases."""
        loader = PathwayLoader()

        # Create two databases
        db1 = loader.load_gmt(sample_gmt, source="DB1")

        # Create second GMT
        gmt_content2 = """PATHWAY_X\tDescription X\tGENE10\tGENE11
PATHWAY_Y\tDescription Y\tGENE12\tGENE13
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gmt", delete=False) as f:
            f.write(gmt_content2)
            gmt2_path = f.name

        db2 = loader.load_gmt(gmt2_path, source="DB2")

        # Merge
        merged = loader.merge([db1, db2])

        assert len(merged) == 5
        assert "DB1:PATHWAY_A" in merged.pathways
        assert "DB2:PATHWAY_X" in merged.pathways
        assert merged.source == "DB1+DB2"

        os.unlink(sample_gmt)
        os.unlink(gmt2_path)

    def test_merge_empty_list(self):
        """Test merging empty list raises error."""
        loader = PathwayLoader()
        with pytest.raises(ValueError):
            loader.merge([])

    def test_subset_by_genes(self, sample_gmt):
        """Test subsetting database by gene list."""
        loader = PathwayLoader()
        db = loader.load_gmt(sample_gmt)

        # Subset to only include certain genes
        subset = loader.subset_by_genes(db, {"GENE1", "GENE2", "GENE6"})

        # PATHWAY_A has GENE1, GENE2 (2 of its 3 genes)
        # PATHWAY_B has GENE2 (1 of its 3 genes)
        # PATHWAY_C has GENE6 (1 of its 2 genes)
        assert len(subset) == 3
        assert subset.get_pathway_genes("PATHWAY_A") == {"GENE1", "GENE2"}
        assert subset.get_pathway_genes("PATHWAY_C") == {"GENE6"}

        os.unlink(sample_gmt)


class TestGOLoader:
    """Tests for Gene Ontology loading."""

    @pytest.fixture
    def sample_obo(self):
        """Create a sample OBO file for testing."""
        obo_content = """format-version: 1.2
ontology: go

[Term]
id: GO:0000001
name: mitochondrion inheritance
namespace: biological_process
def: "The distribution of mitochondria" [GOC:mcc]

[Term]
id: GO:0000002
name: mitochondrial genome maintenance
namespace: biological_process
def: "The maintenance of the structure" [GOC:ai]

[Term]
id: GO:0000003
name: reproduction
namespace: biological_process
def: "The production of new individuals" [GOC:go_curators]

[Term]
id: GO:0000004
name: obsolete term
namespace: biological_process
is_obsolete: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".obo", delete=False) as f:
            f.write(obo_content)
            return f.name

    @pytest.fixture
    def sample_gaf(self):
        """Create a sample GAF file for testing."""
        gaf_content = """!gaf-version: 2.1
UniProtKB\tA0A024RBG1\tGENE1\t\tGO:0000001\tPMID:123\tIDA\t\tP\tDesc\tAlias\tprotein\ttaxon:9606
UniProtKB\tA0A024RBG2\tGENE2\t\tGO:0000001\tPMID:456\tIDA\t\tP\tDesc\tAlias\tprotein\ttaxon:9606
UniProtKB\tA0A024RBG3\tGENE2\t\tGO:0000002\tPMID:789\tIDA\t\tP\tDesc\tAlias\tprotein\ttaxon:9606
UniProtKB\tA0A024RBG4\tGENE3\t\tGO:0000003\tPMID:111\tIDA\t\tP\tDesc\tAlias\tprotein\ttaxon:9606
UniProtKB\tA0A024RBG5\tGENE4\tNOT\tGO:0000001\tPMID:222\tIDA\t\tP\tDesc\tAlias\tprotein\ttaxon:9606
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gaf", delete=False) as f:
            f.write(gaf_content)
            return f.name

    def test_load_go(self, sample_obo, sample_gaf):
        """Test loading Gene Ontology."""
        loader = PathwayLoader()
        db = loader.load_go(
            sample_obo,
            sample_gaf,
            namespaces=["biological_process"]
        )

        # Should have 3 non-obsolete BP terms
        # But only those with annotations will be in the database
        assert "GO:0000001" in db.pathways
        assert "GO:0000002" in db.pathways
        assert "GO:0000003" in db.pathways

        # Check gene annotations
        # GO:0000001 should have GENE1 and GENE2 (NOT GENE4 due to NOT qualifier)
        assert db.get_pathway_genes("GO:0000001") == {"GENE1", "GENE2"}

        # GO:0000002 should have GENE2
        assert db.get_pathway_genes("GO:0000002") == {"GENE2"}

        os.unlink(sample_obo)
        os.unlink(sample_gaf)


class TestMSigDBLoader:
    """Tests for MSigDB loading."""

    @pytest.fixture
    def sample_msigdb_gmt(self):
        """Create a sample MSigDB GMT file for testing."""
        gmt_content = """HALLMARK_APOPTOSIS\thttp://www.gsea-msigdb.org/\tGENE1\tGENE2\tGENE3
HALLMARK_HYPOXIA\thttp://www.gsea-msigdb.org/\tGENE4\tGENE5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gmt", delete=False) as f:
            f.write(gmt_content)
            return f.name

    def test_load_msigdb(self, sample_msigdb_gmt):
        """Test loading MSigDB gene sets."""
        loader = PathwayLoader()
        db = loader.load_msigdb(sample_msigdb_gmt, collection="H")

        assert len(db) == 2
        assert "HALLMARK_APOPTOSIS" in db.pathways
        assert db.source == "MSigDB:H"

        os.unlink(sample_msigdb_gmt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
