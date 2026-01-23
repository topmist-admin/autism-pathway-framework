"""Tests for pathway-drug mapping."""

import sys
from pathlib import Path

# Add module to path
_module_dir = Path(__file__).parent.parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

import pytest
import tempfile
import os
from pathway_drug_mapping import (
    DrugMechanism,
    DrugStatus,
    DrugCandidate,
    DrugTargetDatabase,
    PathwayDrugMapperConfig,
    PathwayDrugMapper,
    create_sample_drug_database,
)


class TestDrugMechanism:
    """Tests for DrugMechanism enum."""

    def test_all_mechanisms_defined(self):
        """Test all mechanism types are defined."""
        mechanisms = [
            DrugMechanism.ANTAGONIST,
            DrugMechanism.AGONIST,
            DrugMechanism.INHIBITOR,
            DrugMechanism.ACTIVATOR,
            DrugMechanism.MODULATOR,
            DrugMechanism.BLOCKER,
            DrugMechanism.ENHANCER,
            DrugMechanism.STABILIZER,
            DrugMechanism.UNKNOWN,
        ]
        assert len(mechanisms) == 9

    def test_mechanism_values(self):
        """Test mechanism string values."""
        assert DrugMechanism.INHIBITOR.value == "inhibitor"
        assert DrugMechanism.AGONIST.value == "agonist"


class TestDrugStatus:
    """Tests for DrugStatus enum."""

    def test_all_statuses_defined(self):
        """Test all status types are defined."""
        statuses = [
            DrugStatus.APPROVED,
            DrugStatus.INVESTIGATIONAL,
            DrugStatus.EXPERIMENTAL,
            DrugStatus.WITHDRAWN,
            DrugStatus.UNKNOWN,
        ]
        assert len(statuses) == 5


class TestDrugCandidate:
    """Tests for DrugCandidate dataclass."""

    def test_creation(self):
        """Test basic drug candidate creation."""
        drug = DrugCandidate(
            drug_id="DB001",
            drug_name="Test Drug",
            target_genes=["GENE1", "GENE2"],
            mechanism="Inhibitor",
            asd_relevance_score=0.7,
        )

        assert drug.drug_id == "DB001"
        assert drug.drug_name == "Test Drug"
        assert len(drug.target_genes) == 2
        assert drug.asd_relevance_score == 0.7

    def test_empty_drug_id_raises(self):
        """Test that empty drug_id raises ValueError."""
        with pytest.raises(ValueError, match="drug_id cannot be empty"):
            DrugCandidate(drug_id="", drug_name="Test")

    def test_invalid_asd_relevance_raises(self):
        """Test that invalid ASD relevance score raises ValueError."""
        with pytest.raises(ValueError, match="asd_relevance_score must be between"):
            DrugCandidate(drug_id="DB001", drug_name="Test", asd_relevance_score=1.5)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        drug = DrugCandidate(
            drug_id="DB001",
            drug_name="Test Drug",
            target_genes=["GENE1"],
            mechanism="Inhibitor",
            mechanism_type=DrugMechanism.INHIBITOR,
            status=DrugStatus.APPROVED,
        )

        d = drug.to_dict()
        assert d["drug_id"] == "DB001"
        assert d["mechanism_type"] == "inhibitor"
        assert d["status"] == "approved"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "drug_id": "DB001",
            "drug_name": "Test Drug",
            "target_genes": ["GENE1"],
            "mechanism_type": "inhibitor",
            "status": "approved",
            "asd_relevance_score": 0.5,
        }

        drug = DrugCandidate.from_dict(data)
        assert drug.drug_id == "DB001"
        assert drug.mechanism_type == DrugMechanism.INHIBITOR
        assert drug.status == DrugStatus.APPROVED


class TestDrugTargetDatabase:
    """Tests for DrugTargetDatabase."""

    def test_empty_database(self):
        """Test empty database creation."""
        db = DrugTargetDatabase()
        assert len(db.drugs) == 0
        assert len(db.gene_to_drugs) == 0

    def test_add_drug(self):
        """Test adding a drug."""
        db = DrugTargetDatabase()
        drug = DrugCandidate(
            drug_id="DB001",
            drug_name="Test Drug",
            target_genes=["GENE1", "GENE2"],
            pathways=["pathway1"],
        )
        db.add_drug(drug)

        assert "DB001" in db.drugs
        assert "GENE1" in db.gene_to_drugs
        assert "GENE2" in db.gene_to_drugs
        assert "pathway1" in db.pathway_to_drugs

    def test_get_drug(self):
        """Test retrieving a drug."""
        db = DrugTargetDatabase()
        drug = DrugCandidate(drug_id="DB001", drug_name="Test")
        db.add_drug(drug)

        retrieved = db.get_drug("DB001")
        assert retrieved is not None
        assert retrieved.drug_id == "DB001"

        missing = db.get_drug("MISSING")
        assert missing is None

    def test_get_drugs_for_gene(self):
        """Test getting drugs targeting a gene."""
        db = DrugTargetDatabase()
        db.add_drug(DrugCandidate(
            drug_id="DB001",
            drug_name="Drug1",
            target_genes=["GENE1", "GENE2"],
        ))
        db.add_drug(DrugCandidate(
            drug_id="DB002",
            drug_name="Drug2",
            target_genes=["GENE1", "GENE3"],
        ))

        gene1_drugs = db.get_drugs_for_gene("GENE1")
        assert len(gene1_drugs) == 2

        gene3_drugs = db.get_drugs_for_gene("GENE3")
        assert len(gene3_drugs) == 1

    def test_get_drugs_for_pathway(self):
        """Test getting drugs affecting a pathway."""
        db = DrugTargetDatabase()
        db.add_drug(DrugCandidate(
            drug_id="DB001",
            drug_name="Drug1",
            pathways=["synaptic", "glutamate"],
        ))
        db.add_drug(DrugCandidate(
            drug_id="DB002",
            drug_name="Drug2",
            pathways=["synaptic"],
        ))

        synaptic_drugs = db.get_drugs_for_pathway("synaptic")
        assert len(synaptic_drugs) == 2

        glutamate_drugs = db.get_drugs_for_pathway("glutamate")
        assert len(glutamate_drugs) == 1

    def test_search_drugs(self):
        """Test searching drugs by criteria."""
        db = DrugTargetDatabase()
        db.add_drug(DrugCandidate(
            drug_id="DB001",
            drug_name="Memantine",
            mechanism_type=DrugMechanism.ANTAGONIST,
            status=DrugStatus.APPROVED,
            asd_relevance_score=0.7,
        ))
        db.add_drug(DrugCandidate(
            drug_id="DB002",
            drug_name="Experimental Drug",
            status=DrugStatus.EXPERIMENTAL,
            asd_relevance_score=0.3,
        ))

        # Search by name
        results = db.search_drugs(name_pattern="meman")
        assert len(results) == 1
        assert results[0].drug_id == "DB001"

        # Search by status
        results = db.search_drugs(status=DrugStatus.APPROVED)
        assert len(results) == 1

        # Search by ASD relevance
        results = db.search_drugs(min_asd_relevance=0.5)
        assert len(results) == 1

    def test_get_statistics(self):
        """Test database statistics."""
        db = DrugTargetDatabase()
        db.add_drug(DrugCandidate(
            drug_id="DB001",
            drug_name="Drug1",
            target_genes=["GENE1", "GENE2"],
            pathways=["pathway1"],
            status=DrugStatus.APPROVED,
            asd_relevance_score=0.5,
        ))

        stats = db.get_statistics()
        assert stats["total_drugs"] == 1
        assert stats["total_genes_targeted"] == 2
        assert stats["total_pathways_covered"] == 1
        assert stats["approved_drugs"] == 1

    def test_load_from_csv(self):
        """Test loading from CSV file."""
        db = DrugTargetDatabase()

        # Create temp CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("drug_id,drug_name,targets,mechanism,status\n")
            f.write("DB001,TestDrug,GENE1;GENE2,Inhibitor,approved\n")
            temp_path = f.name

        try:
            loaded = db.load_drugbank(temp_path)
            assert loaded == 1
            assert "DB001" in db.drugs
        finally:
            os.unlink(temp_path)

    def test_save_and_load_json(self):
        """Test JSON save and load."""
        db = DrugTargetDatabase()
        db.add_drug(DrugCandidate(
            drug_id="DB001",
            drug_name="Test Drug",
            target_genes=["GENE1"],
            asd_relevance_score=0.7,
        ))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            db.save_to_json(temp_path)

            db2 = DrugTargetDatabase()
            loaded = db2.load_from_json(temp_path)
            assert loaded == 1
            assert "DB001" in db2.drugs
            assert db2.drugs["DB001"].asd_relevance_score == 0.7
        finally:
            os.unlink(temp_path)


class TestPathwayDrugMapperConfig:
    """Tests for PathwayDrugMapperConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PathwayDrugMapperConfig()
        assert config.min_asd_relevance == 0.0
        assert config.include_approved is True
        assert config.include_withdrawn is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = PathwayDrugMapperConfig(
            min_asd_relevance=0.5,
            include_experimental=True,
        )
        assert config.min_asd_relevance == 0.5
        assert config.include_experimental is True


class TestPathwayDrugMapper:
    """Tests for PathwayDrugMapper."""

    @pytest.fixture
    def sample_db(self):
        """Create sample database for testing."""
        return create_sample_drug_database()

    @pytest.fixture
    def mapper(self, sample_db):
        """Create mapper with sample database."""
        return PathwayDrugMapper(sample_db)

    def test_map_pathway(self, mapper):
        """Test mapping a pathway to drugs."""
        candidates = mapper.map(
            pathway_id="synaptic_transmission",
            pathway_genes=["GRIN1", "GRIN2A", "GABBR1"],
        )

        assert len(candidates) > 0
        # Should find Memantine and Arbaclofen
        drug_ids = [c.drug_id for c in candidates]
        assert "DB00334" in drug_ids or "DB01104" in drug_ids

    def test_map_with_disrupted_genes(self, mapper):
        """Test mapping with disrupted genes."""
        candidates = mapper.map(
            pathway_id="mtor_signaling",
            pathway_genes=["MTOR", "TSC1", "TSC2"],
            disrupted_genes=["MTOR"],
        )

        # Should prioritize drugs targeting MTOR
        assert len(candidates) > 0
        top_targets = candidates[0].target_genes
        assert "MTOR" in top_targets

    def test_map_multiple_pathways(self, mapper):
        """Test mapping multiple pathways."""
        pathway_scores = {
            "synaptic_transmission": 2.5,
            "mtor_signaling": 2.0,
            "chromatin_remodeling": 1.0,  # Below threshold
        }

        results = mapper.map_multiple_pathways(
            pathway_scores=pathway_scores,
            min_pathway_zscore=1.5,
        )

        # Should have results for synaptic and mtor but not chromatin
        assert "synaptic_transmission" in results or "mtor_signaling" in results

    def test_filter_by_status(self, sample_db):
        """Test filtering by drug status."""
        config = PathwayDrugMapperConfig(
            include_approved=True,
            include_investigational=False,
        )
        mapper = PathwayDrugMapper(sample_db, config)

        candidates = mapper.map(
            pathway_id="synaptic_transmission",
        )

        # Should not include investigational drugs
        for c in candidates:
            assert c.status != DrugStatus.INVESTIGATIONAL

    def test_filter_by_asd_relevance(self, sample_db):
        """Test filtering by ASD relevance score."""
        config = PathwayDrugMapperConfig(min_asd_relevance=0.6)
        mapper = PathwayDrugMapper(sample_db, config)

        candidates = mapper.map(pathway_id="synaptic_transmission")

        for c in candidates:
            assert c.asd_relevance_score >= 0.6


class TestSampleDrugDatabase:
    """Tests for sample drug database."""

    def test_create_sample_database(self):
        """Test creating sample database."""
        db = create_sample_drug_database()

        assert len(db.drugs) > 0
        assert len(db.gene_to_drugs) > 0
        assert len(db.pathway_to_drugs) > 0

    def test_sample_drugs_have_required_fields(self):
        """Test sample drugs have required fields."""
        db = create_sample_drug_database()

        for drug in db.drugs.values():
            assert drug.drug_id
            assert drug.drug_name
            assert drug.mechanism
            assert 0 <= drug.asd_relevance_score <= 1

    def test_sample_pathways_covered(self):
        """Test sample database covers key pathways."""
        db = create_sample_drug_database()

        expected_pathways = [
            "synaptic_transmission",
            "chromatin_remodeling",
            "mtor_signaling",
        ]

        for pathway in expected_pathways:
            assert pathway in db.pathway_to_drugs
