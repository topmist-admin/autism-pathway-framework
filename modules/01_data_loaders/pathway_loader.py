"""
Pathway Loader

Handles loading of biological pathway databases including:
- Gene Ontology (GO)
- Reactome
- KEGG
- GMT (Gene Matrix Transposed) format files
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PathwayDatabase:
    """
    Collection of biological pathways with gene annotations.

    Attributes:
        pathways: Mapping from pathway ID to set of gene IDs
        pathway_names: Mapping from pathway ID to human-readable name
        pathway_descriptions: Mapping from pathway ID to description
        gene_to_pathways: Reverse mapping from gene ID to pathway IDs
        source: Database source (GO, Reactome, KEGG, etc.)
        metadata: Additional metadata about the database
    """

    pathways: Dict[str, Set[str]]
    pathway_names: Dict[str, str]
    pathway_descriptions: Dict[str, str] = field(default_factory=dict)
    gene_to_pathways: Dict[str, Set[str]] = field(default_factory=dict)
    source: str = "unknown"
    metadata: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """Build reverse mapping if not provided."""
        if not self.gene_to_pathways:
            self.gene_to_pathways = self._build_gene_to_pathways()

    def _build_gene_to_pathways(self) -> Dict[str, Set[str]]:
        """Build gene to pathways reverse mapping."""
        gene_to_pathways: Dict[str, Set[str]] = defaultdict(set)
        for pathway_id, genes in self.pathways.items():
            for gene in genes:
                gene_to_pathways[gene].add(pathway_id)
        return dict(gene_to_pathways)

    def __len__(self) -> int:
        return len(self.pathways)

    def get_pathway_genes(self, pathway_id: str) -> Set[str]:
        """Get genes in a pathway."""
        return self.pathways.get(pathway_id, set())

    def get_gene_pathways(self, gene_id: str) -> Set[str]:
        """Get pathways containing a gene."""
        return self.gene_to_pathways.get(gene_id, set())

    def get_pathway_size(self, pathway_id: str) -> int:
        """Get number of genes in a pathway."""
        return len(self.pathways.get(pathway_id, set()))

    def filter_by_size(
        self, min_size: int = 5, max_size: int = 500
    ) -> "PathwayDatabase":
        """
        Filter pathways by size.

        Args:
            min_size: Minimum number of genes
            max_size: Maximum number of genes

        Returns:
            New PathwayDatabase with filtered pathways
        """
        filtered_pathways = {
            pid: genes
            for pid, genes in self.pathways.items()
            if min_size <= len(genes) <= max_size
        }
        filtered_names = {
            pid: name
            for pid, name in self.pathway_names.items()
            if pid in filtered_pathways
        }
        filtered_desc = {
            pid: desc
            for pid, desc in self.pathway_descriptions.items()
            if pid in filtered_pathways
        }

        return PathwayDatabase(
            pathways=filtered_pathways,
            pathway_names=filtered_names,
            pathway_descriptions=filtered_desc,
            source=self.source,
            metadata={
                **self.metadata,
                "filtered": True,
                "min_size": min_size,
                "max_size": max_size,
            },
        )

    def get_all_genes(self) -> Set[str]:
        """Get all genes across all pathways."""
        all_genes: Set[str] = set()
        for genes in self.pathways.values():
            all_genes.update(genes)
        return all_genes

    def to_gmt(self, output_path: str) -> None:
        """Export to GMT format."""
        with open(output_path, "w") as f:
            for pathway_id, genes in self.pathways.items():
                name = self.pathway_names.get(pathway_id, pathway_id)
                desc = self.pathway_descriptions.get(pathway_id, "")
                gene_list = "\t".join(sorted(genes))
                f.write(f"{pathway_id}\t{name}\t{desc}\t{gene_list}\n")


class PathwayLoader:
    """
    Loader for various pathway database formats.

    Supports:
    - Gene Ontology (OBO + GAF)
    - Reactome (GMT)
    - KEGG (GMT)
    - Generic GMT files
    """

    # GO namespaces
    GO_BIOLOGICAL_PROCESS = "biological_process"
    GO_MOLECULAR_FUNCTION = "molecular_function"
    GO_CELLULAR_COMPONENT = "cellular_component"

    def __init__(self, gene_id_type: str = "symbol"):
        """
        Initialize pathway loader.

        Args:
            gene_id_type: Type of gene identifiers to use (symbol, ensembl, entrez)
        """
        self.gene_id_type = gene_id_type

    def load_gmt(
        self, gmt_path: str, source: str = "GMT"
    ) -> PathwayDatabase:
        """
        Load pathways from GMT (Gene Matrix Transposed) format.

        GMT format: pathway_id<TAB>description<TAB>gene1<TAB>gene2<TAB>...

        Args:
            gmt_path: Path to GMT file
            source: Source name for the database

        Returns:
            PathwayDatabase
        """
        path = Path(gmt_path)
        if not path.exists():
            raise FileNotFoundError(f"GMT file not found: {gmt_path}")

        pathways: Dict[str, Set[str]] = {}
        pathway_names: Dict[str, str] = {}
        pathway_descriptions: Dict[str, str] = {}

        with open(path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                fields = line.split("\t")
                if len(fields) < 3:
                    logger.warning(f"Skipping malformed line {line_num} in {gmt_path}")
                    continue

                pathway_id = fields[0]
                description = fields[1]
                genes = set(fields[2:]) - {"", "na", "NA"}

                if not genes:
                    continue

                pathways[pathway_id] = genes
                pathway_names[pathway_id] = pathway_id  # Use ID as name
                pathway_descriptions[pathway_id] = description

        logger.info(
            f"Loaded {len(pathways)} pathways from {gmt_path} "
            f"({sum(len(g) for g in pathways.values())} gene-pathway associations)"
        )

        return PathwayDatabase(
            pathways=pathways,
            pathway_names=pathway_names,
            pathway_descriptions=pathway_descriptions,
            source=source,
            metadata={"file": str(path)},
        )

    def load_reactome(self, gmt_path: str) -> PathwayDatabase:
        """
        Load Reactome pathways from GMT format.

        Args:
            gmt_path: Path to Reactome GMT file

        Returns:
            PathwayDatabase
        """
        db = self.load_gmt(gmt_path, source="Reactome")

        # Parse Reactome-specific pathway names
        # Format often: "R-HSA-123456%Pathway Name"
        updated_names = {}
        for pathway_id in db.pathways:
            if "%" in pathway_id:
                parts = pathway_id.split("%")
                updated_names[pathway_id] = parts[-1]
            else:
                updated_names[pathway_id] = pathway_id

        db.pathway_names = updated_names
        return db

    def load_kegg(self, gmt_path: str) -> PathwayDatabase:
        """
        Load KEGG pathways from GMT format.

        Args:
            gmt_path: Path to KEGG GMT file

        Returns:
            PathwayDatabase
        """
        return self.load_gmt(gmt_path, source="KEGG")

    def load_go(
        self,
        obo_path: str,
        gaf_path: str,
        namespaces: Optional[List[str]] = None,
    ) -> PathwayDatabase:
        """
        Load Gene Ontology terms and annotations.

        Args:
            obo_path: Path to GO OBO ontology file
            gaf_path: Path to GAF gene annotation file
            namespaces: List of GO namespaces to include
                       (biological_process, molecular_function, cellular_component)

        Returns:
            PathwayDatabase
        """
        if namespaces is None:
            namespaces = [self.GO_BIOLOGICAL_PROCESS]

        # Parse OBO file for term definitions
        terms = self._parse_obo(obo_path, namespaces)

        # Parse GAF file for gene annotations
        annotations = self._parse_gaf(gaf_path)

        # Build pathway database
        pathways: Dict[str, Set[str]] = defaultdict(set)
        pathway_names: Dict[str, str] = {}
        pathway_descriptions: Dict[str, str] = {}

        for go_id, term_info in terms.items():
            pathway_names[go_id] = term_info["name"]
            pathway_descriptions[go_id] = term_info.get("def", "")

        for gene_id, go_ids in annotations.items():
            for go_id in go_ids:
                if go_id in terms:
                    pathways[go_id].add(gene_id)

        # Remove empty terms
        pathways = {k: v for k, v in pathways.items() if v}
        pathway_names = {k: v for k, v in pathway_names.items() if k in pathways}
        pathway_descriptions = {k: v for k, v in pathway_descriptions.items() if k in pathways}

        logger.info(
            f"Loaded {len(pathways)} GO terms from {obo_path} "
            f"with {len(annotations)} annotated genes"
        )

        return PathwayDatabase(
            pathways=dict(pathways),
            pathway_names=pathway_names,
            pathway_descriptions=pathway_descriptions,
            source="GO",
            metadata={
                "obo_file": obo_path,
                "gaf_file": gaf_path,
                "namespaces": namespaces,
            },
        )

    def _parse_obo(
        self, obo_path: str, namespaces: List[str]
    ) -> Dict[str, Dict[str, str]]:
        """
        Parse GO OBO ontology file.

        Args:
            obo_path: Path to OBO file
            namespaces: List of namespaces to include

        Returns:
            Dictionary of GO term ID to term info
        """
        path = Path(obo_path)
        if not path.exists():
            raise FileNotFoundError(f"OBO file not found: {obo_path}")

        terms: Dict[str, Dict[str, str]] = {}
        current_term: Optional[Dict[str, str]] = None

        with open(path, "r") as f:
            for line in f:
                line = line.strip()

                if line == "[Term]":
                    # Save previous term if valid
                    if current_term and current_term.get("namespace") in namespaces:
                        if not current_term.get("is_obsolete"):
                            terms[current_term["id"]] = current_term
                    current_term = {}
                    continue

                if line == "[Typedef]":
                    # Save previous term if valid
                    if current_term and current_term.get("namespace") in namespaces:
                        if not current_term.get("is_obsolete"):
                            terms[current_term["id"]] = current_term
                    current_term = None
                    continue

                if current_term is None:
                    continue

                if ": " in line:
                    key, value = line.split(": ", 1)
                    if key == "id":
                        current_term["id"] = value
                    elif key == "name":
                        current_term["name"] = value
                    elif key == "namespace":
                        current_term["namespace"] = value
                    elif key == "def":
                        current_term["def"] = value.split('"')[1] if '"' in value else value
                    elif key == "is_obsolete" and value == "true":
                        current_term["is_obsolete"] = True

        # Don't forget last term
        if current_term and current_term.get("namespace") in namespaces:
            if not current_term.get("is_obsolete"):
                terms[current_term["id"]] = current_term

        return terms

    def _parse_gaf(self, gaf_path: str) -> Dict[str, Set[str]]:
        """
        Parse GO Annotation File (GAF) format.

        Args:
            gaf_path: Path to GAF file

        Returns:
            Dictionary of gene symbol to GO term IDs
        """
        path = Path(gaf_path)
        if not path.exists():
            raise FileNotFoundError(f"GAF file not found: {gaf_path}")

        annotations: Dict[str, Set[str]] = defaultdict(set)

        # Determine file opening method
        open_func = open
        mode = "r"
        if str(path).endswith(".gz"):
            import gzip
            open_func = gzip.open
            mode = "rt"

        with open_func(path, mode) as f:
            for line in f:
                if line.startswith("!"):
                    continue

                fields = line.strip().split("\t")
                if len(fields) < 5:
                    continue

                # GAF format columns:
                # 0: DB
                # 1: DB_Object_ID
                # 2: DB_Object_Symbol (gene symbol)
                # 3: Qualifier
                # 4: GO_ID
                gene_symbol = fields[2]
                go_id = fields[4]

                # Skip NOT qualifiers
                if fields[3].startswith("NOT"):
                    continue

                annotations[gene_symbol].add(go_id)

        return dict(annotations)

    def merge(self, databases: List[PathwayDatabase]) -> PathwayDatabase:
        """
        Merge multiple pathway databases.

        Args:
            databases: List of PathwayDatabase objects to merge

        Returns:
            Merged PathwayDatabase
        """
        if not databases:
            raise ValueError("No databases to merge")

        if len(databases) == 1:
            return databases[0]

        merged_pathways: Dict[str, Set[str]] = {}
        merged_names: Dict[str, str] = {}
        merged_descriptions: Dict[str, str] = {}
        sources = []

        for db in databases:
            sources.append(db.source)

            for pathway_id, genes in db.pathways.items():
                # Prefix pathway ID with source to avoid collisions
                prefixed_id = f"{db.source}:{pathway_id}"
                merged_pathways[prefixed_id] = genes.copy()
                merged_names[prefixed_id] = db.pathway_names.get(pathway_id, pathway_id)
                merged_descriptions[prefixed_id] = db.pathway_descriptions.get(
                    pathway_id, ""
                )

        logger.info(
            f"Merged {len(databases)} databases: {sources} "
            f"-> {len(merged_pathways)} total pathways"
        )

        return PathwayDatabase(
            pathways=merged_pathways,
            pathway_names=merged_names,
            pathway_descriptions=merged_descriptions,
            source="+".join(sources),
            metadata={"merged_from": sources, "n_original_dbs": len(databases)},
        )

    def load_msigdb(
        self,
        gmt_path: str,
        collection: Optional[str] = None,
    ) -> PathwayDatabase:
        """
        Load MSigDB gene sets from GMT format.

        Args:
            gmt_path: Path to MSigDB GMT file
            collection: Optional collection name (H, C1, C2, etc.)

        Returns:
            PathwayDatabase
        """
        db = self.load_gmt(gmt_path, source=f"MSigDB:{collection}" if collection else "MSigDB")
        return db

    def subset_by_genes(
        self, database: PathwayDatabase, genes: Set[str]
    ) -> PathwayDatabase:
        """
        Create a subset of pathway database containing only specified genes.

        Args:
            database: Source PathwayDatabase
            genes: Set of genes to include

        Returns:
            New PathwayDatabase with only specified genes
        """
        subset_pathways: Dict[str, Set[str]] = {}

        for pathway_id, pathway_genes in database.pathways.items():
            intersection = pathway_genes & genes
            if intersection:
                subset_pathways[pathway_id] = intersection

        return PathwayDatabase(
            pathways=subset_pathways,
            pathway_names={
                k: v for k, v in database.pathway_names.items() if k in subset_pathways
            },
            pathway_descriptions={
                k: v
                for k, v in database.pathway_descriptions.items()
                if k in subset_pathways
            },
            source=database.source,
            metadata={
                **database.metadata,
                "subset": True,
                "n_query_genes": len(genes),
            },
        )
