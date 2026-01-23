"""
Drug-pathway mapping for therapeutic hypothesis generation.

This module provides the drug-target database and pathway-drug mapping
functionality for generating therapeutic hypotheses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import logging
import csv
import json
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class DrugMechanism(Enum):
    """Drug mechanism categories relevant to ASD."""

    ANTAGONIST = "antagonist"
    AGONIST = "agonist"
    INHIBITOR = "inhibitor"
    ACTIVATOR = "activator"
    MODULATOR = "modulator"
    BLOCKER = "blocker"
    ENHANCER = "enhancer"
    STABILIZER = "stabilizer"
    UNKNOWN = "unknown"


class DrugStatus(Enum):
    """Drug development status."""

    APPROVED = "approved"
    INVESTIGATIONAL = "investigational"
    EXPERIMENTAL = "experimental"
    WITHDRAWN = "withdrawn"
    UNKNOWN = "unknown"


@dataclass
class DrugCandidate:
    """
    A drug candidate for therapeutic hypothesis.

    Attributes:
        drug_id: Unique drug identifier (e.g., DrugBank ID)
        drug_name: Human-readable drug name
        target_genes: Genes targeted by this drug
        mechanism: Mechanism of action
        mechanism_type: Categorized mechanism type
        indications: Known therapeutic indications
        contraindications: Known contraindications
        asd_relevance_score: Score for ASD relevance (0-1)
        status: Development/approval status
        pathways: Pathways affected by this drug
        metadata: Additional drug information
    """

    drug_id: str
    drug_name: str
    target_genes: List[str] = field(default_factory=list)
    mechanism: str = ""
    mechanism_type: DrugMechanism = DrugMechanism.UNKNOWN
    indications: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    asd_relevance_score: float = 0.0
    status: DrugStatus = DrugStatus.UNKNOWN
    pathways: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate drug candidate."""
        if not self.drug_id:
            raise ValueError("drug_id cannot be empty")
        if not 0 <= self.asd_relevance_score <= 1:
            raise ValueError("asd_relevance_score must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "drug_id": self.drug_id,
            "drug_name": self.drug_name,
            "target_genes": self.target_genes,
            "mechanism": self.mechanism,
            "mechanism_type": self.mechanism_type.value,
            "indications": self.indications,
            "contraindications": self.contraindications,
            "asd_relevance_score": self.asd_relevance_score,
            "status": self.status.value,
            "pathways": self.pathways,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DrugCandidate":
        """Create from dictionary."""
        return cls(
            drug_id=data["drug_id"],
            drug_name=data.get("drug_name", data["drug_id"]),
            target_genes=data.get("target_genes", []),
            mechanism=data.get("mechanism", ""),
            mechanism_type=DrugMechanism(data.get("mechanism_type", "unknown")),
            indications=data.get("indications", []),
            contraindications=data.get("contraindications", []),
            asd_relevance_score=data.get("asd_relevance_score", 0.0),
            status=DrugStatus(data.get("status", "unknown")),
            pathways=data.get("pathways", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DrugTargetDatabase:
    """
    Database of drug-target-pathway relationships.

    Stores and queries relationships between drugs, their target genes,
    and the biological pathways they affect.
    """

    # Drug information
    drugs: Dict[str, DrugCandidate] = field(default_factory=dict)

    # Mapping structures
    gene_to_drugs: Dict[str, Set[str]] = field(default_factory=dict)
    pathway_to_drugs: Dict[str, Set[str]] = field(default_factory=dict)
    drug_to_pathways: Dict[str, Set[str]] = field(default_factory=dict)

    # Metadata
    source_files: List[str] = field(default_factory=list)
    version: str = "1.0.0"

    def add_drug(self, drug: DrugCandidate) -> None:
        """Add a drug to the database."""
        self.drugs[drug.drug_id] = drug

        # Update gene mappings
        for gene in drug.target_genes:
            if gene not in self.gene_to_drugs:
                self.gene_to_drugs[gene] = set()
            self.gene_to_drugs[gene].add(drug.drug_id)

        # Update pathway mappings
        for pathway in drug.pathways:
            if pathway not in self.pathway_to_drugs:
                self.pathway_to_drugs[pathway] = set()
            self.pathway_to_drugs[pathway].add(drug.drug_id)

            if drug.drug_id not in self.drug_to_pathways:
                self.drug_to_pathways[drug.drug_id] = set()
            self.drug_to_pathways[drug.drug_id].add(pathway)

    def get_drug(self, drug_id: str) -> Optional[DrugCandidate]:
        """Get drug by ID."""
        return self.drugs.get(drug_id)

    def get_drugs_for_gene(self, gene_id: str) -> List[DrugCandidate]:
        """Get all drugs targeting a gene."""
        drug_ids = self.gene_to_drugs.get(gene_id, set())
        return [self.drugs[did] for did in drug_ids if did in self.drugs]

    def get_drugs_for_pathway(self, pathway_id: str) -> List[DrugCandidate]:
        """Get all drugs affecting a pathway."""
        drug_ids = self.pathway_to_drugs.get(pathway_id, set())
        return [self.drugs[did] for did in drug_ids if did in self.drugs]

    def get_pathways_for_drug(self, drug_id: str) -> List[str]:
        """Get pathways affected by a drug."""
        return list(self.drug_to_pathways.get(drug_id, set()))

    def search_drugs(
        self,
        name_pattern: Optional[str] = None,
        mechanism_type: Optional[DrugMechanism] = None,
        status: Optional[DrugStatus] = None,
        min_asd_relevance: float = 0.0,
    ) -> List[DrugCandidate]:
        """Search drugs by criteria."""
        results = []
        for drug in self.drugs.values():
            if name_pattern and name_pattern.lower() not in drug.drug_name.lower():
                continue
            if mechanism_type and drug.mechanism_type != mechanism_type:
                continue
            if status and drug.status != status:
                continue
            if drug.asd_relevance_score < min_asd_relevance:
                continue
            results.append(drug)
        return results

    def load_drugbank(self, file_path: str) -> int:
        """
        Load drug data from DrugBank-format CSV.

        Expected columns: drug_id, drug_name, targets, mechanism, status, indications

        Returns number of drugs loaded.
        """
        loaded = 0
        path = Path(file_path)

        if not path.exists():
            logger.warning(f"DrugBank file not found: {file_path}")
            return 0

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    targets = row.get("targets", "").split(";")
                    targets = [t.strip() for t in targets if t.strip()]

                    indications = row.get("indications", "").split(";")
                    indications = [i.strip() for i in indications if i.strip()]

                    contraindications = row.get("contraindications", "").split(";")
                    contraindications = [c.strip() for c in contraindications if c.strip()]

                    status_str = row.get("status", "unknown").lower()
                    try:
                        status = DrugStatus(status_str)
                    except ValueError:
                        status = DrugStatus.UNKNOWN

                    drug = DrugCandidate(
                        drug_id=row["drug_id"],
                        drug_name=row.get("drug_name", row["drug_id"]),
                        target_genes=targets,
                        mechanism=row.get("mechanism", ""),
                        indications=indications,
                        contraindications=contraindications,
                        status=status,
                    )
                    self.add_drug(drug)
                    loaded += 1
                except Exception as e:
                    logger.warning(f"Error loading drug row: {e}")

        self.source_files.append(file_path)
        logger.info(f"Loaded {loaded} drugs from {file_path}")
        return loaded

    def load_pathway_associations(self, file_path: str) -> int:
        """
        Load drug-pathway associations.

        Expected columns: drug_id, pathway_id

        Returns number of associations loaded.
        """
        loaded = 0
        path = Path(file_path)

        if not path.exists():
            logger.warning(f"Pathway associations file not found: {file_path}")
            return 0

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                drug_id = row.get("drug_id")
                pathway_id = row.get("pathway_id")

                if not drug_id or not pathway_id:
                    continue

                # Add to existing drug or create placeholder
                if drug_id in self.drugs:
                    if pathway_id not in self.drugs[drug_id].pathways:
                        self.drugs[drug_id].pathways.append(pathway_id)
                else:
                    # Create minimal drug entry
                    drug = DrugCandidate(
                        drug_id=drug_id,
                        drug_name=drug_id,
                        pathways=[pathway_id],
                    )
                    self.add_drug(drug)

                # Update mappings
                if pathway_id not in self.pathway_to_drugs:
                    self.pathway_to_drugs[pathway_id] = set()
                self.pathway_to_drugs[pathway_id].add(drug_id)

                if drug_id not in self.drug_to_pathways:
                    self.drug_to_pathways[drug_id] = set()
                self.drug_to_pathways[drug_id].add(pathway_id)

                loaded += 1

        self.source_files.append(file_path)
        logger.info(f"Loaded {loaded} pathway associations from {file_path}")
        return loaded

    def load_asd_relevance_scores(self, file_path: str) -> int:
        """
        Load ASD relevance scores for drugs.

        Expected columns: drug_id, asd_relevance_score

        Returns number of scores loaded.
        """
        loaded = 0
        path = Path(file_path)

        if not path.exists():
            logger.warning(f"ASD relevance file not found: {file_path}")
            return 0

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                drug_id = row.get("drug_id")
                score = float(row.get("asd_relevance_score", 0))

                if drug_id and drug_id in self.drugs:
                    self.drugs[drug_id].asd_relevance_score = max(0, min(1, score))
                    loaded += 1

        logger.info(f"Loaded {loaded} ASD relevance scores from {file_path}")
        return loaded

    def load_from_json(self, file_path: str) -> int:
        """Load database from JSON file."""
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"JSON file not found: {file_path}")
            return 0

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded = 0
        for drug_data in data.get("drugs", []):
            try:
                drug = DrugCandidate.from_dict(drug_data)
                self.add_drug(drug)
                loaded += 1
            except Exception as e:
                logger.warning(f"Error loading drug from JSON: {e}")

        self.source_files.append(file_path)
        logger.info(f"Loaded {loaded} drugs from {file_path}")
        return loaded

    def save_to_json(self, file_path: str) -> None:
        """Save database to JSON file."""
        data = {
            "version": self.version,
            "drugs": [drug.to_dict() for drug in self.drugs.values()],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "total_drugs": len(self.drugs),
            "total_genes_targeted": len(self.gene_to_drugs),
            "total_pathways_covered": len(self.pathway_to_drugs),
            "approved_drugs": len([d for d in self.drugs.values() if d.status == DrugStatus.APPROVED]),
            "drugs_with_asd_relevance": len([d for d in self.drugs.values() if d.asd_relevance_score > 0]),
            "source_files": self.source_files,
        }


@dataclass
class PathwayDrugMapperConfig:
    """Configuration for pathway-drug mapping."""

    # Minimum scores for inclusion
    min_asd_relevance: float = 0.0
    min_pathway_overlap: int = 1  # Min genes overlapping

    # Drug status filter
    include_approved: bool = True
    include_investigational: bool = True
    include_experimental: bool = False
    include_withdrawn: bool = False

    # Mechanism preferences (for ranking)
    preferred_mechanisms: List[DrugMechanism] = field(default_factory=list)

    # Gene overlap weighting
    weight_by_gene_overlap: bool = True


class PathwayDrugMapper:
    """
    Maps disrupted pathways to potential drug candidates.

    Uses drug-target-pathway relationships to identify drugs that may
    therapeutically address pathway disruptions.
    """

    def __init__(
        self,
        drug_db: DrugTargetDatabase,
        config: Optional[PathwayDrugMapperConfig] = None,
    ):
        """
        Initialize pathway-drug mapper.

        Args:
            drug_db: Drug-target database
            config: Mapper configuration
        """
        self.drug_db = drug_db
        self.config = config or PathwayDrugMapperConfig()

    def map(
        self,
        pathway_id: str,
        pathway_genes: Optional[List[str]] = None,
        disrupted_genes: Optional[List[str]] = None,
    ) -> List[DrugCandidate]:
        """
        Map a pathway to drug candidates.

        Args:
            pathway_id: Pathway identifier
            pathway_genes: Genes in the pathway (for overlap calculation)
            disrupted_genes: Specifically disrupted genes (prioritized)

        Returns:
            List of drug candidates ranked by relevance
        """
        candidates = []

        # Get drugs directly associated with pathway
        pathway_drugs = self.drug_db.get_drugs_for_pathway(pathway_id)
        for drug in pathway_drugs:
            if self._filter_drug(drug):
                candidates.append(drug)

        # Get drugs targeting genes in pathway
        if pathway_genes:
            for gene in pathway_genes:
                gene_drugs = self.drug_db.get_drugs_for_gene(gene)
                for drug in gene_drugs:
                    if drug not in candidates and self._filter_drug(drug):
                        candidates.append(drug)

        # Calculate relevance scores
        scored_candidates = []
        for drug in candidates:
            score = self._calculate_mapping_score(
                drug=drug,
                pathway_id=pathway_id,
                pathway_genes=pathway_genes or [],
                disrupted_genes=disrupted_genes or [],
            )
            drug.metadata["mapping_score"] = score
            scored_candidates.append((score, drug))

        # Sort by score descending
        scored_candidates.sort(key=lambda x: -x[0])

        return [drug for _, drug in scored_candidates]

    def map_multiple_pathways(
        self,
        pathway_scores: Dict[str, float],
        pathway_genes: Optional[Dict[str, List[str]]] = None,
        disrupted_genes: Optional[List[str]] = None,
        min_pathway_zscore: float = 1.5,
    ) -> Dict[str, List[DrugCandidate]]:
        """
        Map multiple disrupted pathways to drug candidates.

        Args:
            pathway_scores: Pathway ID -> disruption score (e.g., z-score)
            pathway_genes: Pathway ID -> list of genes
            disrupted_genes: Overall list of disrupted genes
            min_pathway_zscore: Minimum z-score to consider pathway disrupted

        Returns:
            Pathway ID -> list of drug candidates
        """
        results = {}
        pathway_genes = pathway_genes or {}

        for pathway_id, score in pathway_scores.items():
            if score < min_pathway_zscore:
                continue

            genes = pathway_genes.get(pathway_id)
            candidates = self.map(
                pathway_id=pathway_id,
                pathway_genes=genes,
                disrupted_genes=disrupted_genes,
            )
            if candidates:
                results[pathway_id] = candidates

        return results

    def _filter_drug(self, drug: DrugCandidate) -> bool:
        """Check if drug passes filter criteria."""
        # ASD relevance filter
        if drug.asd_relevance_score < self.config.min_asd_relevance:
            return False

        # Status filter
        status_filters = {
            DrugStatus.APPROVED: self.config.include_approved,
            DrugStatus.INVESTIGATIONAL: self.config.include_investigational,
            DrugStatus.EXPERIMENTAL: self.config.include_experimental,
            DrugStatus.WITHDRAWN: self.config.include_withdrawn,
        }

        if drug.status in status_filters:
            if not status_filters[drug.status]:
                return False

        return True

    def _calculate_mapping_score(
        self,
        drug: DrugCandidate,
        pathway_id: str,
        pathway_genes: List[str],
        disrupted_genes: List[str],
    ) -> float:
        """Calculate relevance score for drug-pathway mapping."""
        score = 0.0

        # Base score from ASD relevance
        score += drug.asd_relevance_score * 0.3

        # Gene overlap score
        if pathway_genes and self.config.weight_by_gene_overlap:
            overlap = set(drug.target_genes) & set(pathway_genes)
            overlap_ratio = len(overlap) / max(len(pathway_genes), 1)
            score += overlap_ratio * 0.3

        # Disrupted gene targeting bonus
        if disrupted_genes:
            disrupted_overlap = set(drug.target_genes) & set(disrupted_genes)
            if disrupted_overlap:
                score += 0.2 * len(disrupted_overlap) / len(disrupted_genes)

        # Mechanism preference bonus
        if drug.mechanism_type in self.config.preferred_mechanisms:
            score += 0.1

        # Approval status bonus
        if drug.status == DrugStatus.APPROVED:
            score += 0.1

        return min(score, 1.0)

    def get_all_pathway_drug_associations(self) -> Dict[str, List[str]]:
        """Get all pathway -> drug ID associations."""
        return {
            pathway: list(drug_ids)
            for pathway, drug_ids in self.drug_db.pathway_to_drugs.items()
        }


def create_sample_drug_database() -> DrugTargetDatabase:
    """
    Create a sample drug database for testing/demonstration.

    Includes drugs relevant to ASD-related pathways.
    """
    db = DrugTargetDatabase()

    # Sample drugs targeting synaptic pathways
    db.add_drug(DrugCandidate(
        drug_id="DB00334",
        drug_name="Memantine",
        target_genes=["GRIN1", "GRIN2A", "GRIN2B"],
        mechanism="NMDA receptor antagonist",
        mechanism_type=DrugMechanism.ANTAGONIST,
        indications=["Alzheimer's disease"],
        contraindications=["Hypersensitivity"],
        asd_relevance_score=0.7,
        status=DrugStatus.APPROVED,
        pathways=["synaptic_transmission", "glutamate_signaling"],
    ))

    db.add_drug(DrugCandidate(
        drug_id="DB01104",
        drug_name="Arbaclofen",
        target_genes=["GABBR1", "GABBR2"],
        mechanism="GABA-B receptor agonist",
        mechanism_type=DrugMechanism.AGONIST,
        indications=["Fragile X syndrome (investigational)"],
        asd_relevance_score=0.8,
        status=DrugStatus.INVESTIGATIONAL,
        pathways=["synaptic_transmission", "gaba_signaling"],
    ))

    # Chromatin remodeling pathway
    db.add_drug(DrugCandidate(
        drug_id="DB06603",
        drug_name="Panobinostat",
        target_genes=["HDAC1", "HDAC2", "HDAC3"],
        mechanism="HDAC inhibitor",
        mechanism_type=DrugMechanism.INHIBITOR,
        indications=["Multiple myeloma"],
        contraindications=["Pregnancy", "Severe hepatic impairment"],
        asd_relevance_score=0.4,
        status=DrugStatus.APPROVED,
        pathways=["chromatin_remodeling", "histone_modification"],
    ))

    # mTOR pathway (relevant to TSC-ASD)
    db.add_drug(DrugCandidate(
        drug_id="DB00877",
        drug_name="Sirolimus",
        target_genes=["MTOR"],
        mechanism="mTOR inhibitor",
        mechanism_type=DrugMechanism.INHIBITOR,
        indications=["Organ rejection prophylaxis", "TSC-associated conditions"],
        contraindications=["Hypersensitivity", "Pregnancy"],
        asd_relevance_score=0.75,
        status=DrugStatus.APPROVED,
        pathways=["mtor_signaling", "cell_growth"],
    ))

    db.add_drug(DrugCandidate(
        drug_id="DB01590",
        drug_name="Everolimus",
        target_genes=["MTOR"],
        mechanism="mTOR inhibitor",
        mechanism_type=DrugMechanism.INHIBITOR,
        indications=["TSC", "Breast cancer", "Renal cell carcinoma"],
        asd_relevance_score=0.8,
        status=DrugStatus.APPROVED,
        pathways=["mtor_signaling", "cell_growth"],
    ))

    # Oxytocin pathway
    db.add_drug(DrugCandidate(
        drug_id="DB00107",
        drug_name="Oxytocin",
        target_genes=["OXTR"],
        mechanism="Oxytocin receptor agonist",
        mechanism_type=DrugMechanism.AGONIST,
        indications=["Labor induction"],
        asd_relevance_score=0.6,
        status=DrugStatus.APPROVED,
        pathways=["oxytocin_signaling", "social_behavior"],
    ))

    # Serotonin pathway
    db.add_drug(DrugCandidate(
        drug_id="DB00196",
        drug_name="Fluoxetine",
        target_genes=["SLC6A4"],
        mechanism="Selective serotonin reuptake inhibitor",
        mechanism_type=DrugMechanism.INHIBITOR,
        indications=["Depression", "OCD", "Anxiety"],
        asd_relevance_score=0.5,
        status=DrugStatus.APPROVED,
        pathways=["serotonin_signaling", "monoamine_transport"],
    ))

    # SHANK-related (experimental)
    db.add_drug(DrugCandidate(
        drug_id="EXP001",
        drug_name="IGF-1 (Mecasermin)",
        target_genes=["IGF1R", "SHANK3"],
        mechanism="IGF-1 receptor agonist",
        mechanism_type=DrugMechanism.AGONIST,
        indications=["Growth hormone deficiency"],
        asd_relevance_score=0.65,
        status=DrugStatus.INVESTIGATIONAL,
        pathways=["igf1_signaling", "synaptic_development"],
    ))

    return db
