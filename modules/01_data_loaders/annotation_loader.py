"""
Annotation Loader

Handles loading of various gene and variant annotation files:
- Gene ID mappings (Ensembl, Entrez, Symbol)
- Functional annotations
- External database cross-references
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class GeneInfo:
    """Information about a single gene."""

    symbol: str
    ensembl_id: Optional[str] = None
    entrez_id: Optional[str] = None
    name: Optional[str] = None
    chromosome: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    strand: Optional[str] = None
    biotype: Optional[str] = None
    aliases: List[str] = field(default_factory=list)


@dataclass
class GeneAnnotationDB:
    """
    Database of gene annotations and ID mappings.

    Supports bidirectional lookups between:
    - Gene symbols
    - Ensembl gene IDs
    - Entrez gene IDs
    """

    genes: Dict[str, GeneInfo]  # symbol -> GeneInfo
    symbol_to_ensembl: Dict[str, str] = field(default_factory=dict)
    symbol_to_entrez: Dict[str, str] = field(default_factory=dict)
    ensembl_to_symbol: Dict[str, str] = field(default_factory=dict)
    entrez_to_symbol: Dict[str, str] = field(default_factory=dict)
    alias_to_symbol: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.genes)

    def __contains__(self, gene_id: str) -> bool:
        return self.get_symbol(gene_id) is not None

    def get_symbol(self, gene_id: str) -> Optional[str]:
        """
        Get canonical gene symbol from any identifier.

        Args:
            gene_id: Gene symbol, Ensembl ID, Entrez ID, or alias

        Returns:
            Canonical gene symbol or None
        """
        # Direct symbol lookup
        if gene_id in self.genes:
            return gene_id

        # Ensembl ID
        if gene_id in self.ensembl_to_symbol:
            return self.ensembl_to_symbol[gene_id]

        # Entrez ID
        if gene_id in self.entrez_to_symbol:
            return self.entrez_to_symbol[gene_id]

        # Alias
        if gene_id in self.alias_to_symbol:
            return self.alias_to_symbol[gene_id]

        # Case-insensitive symbol lookup
        gene_id_upper = gene_id.upper()
        for symbol in self.genes:
            if symbol.upper() == gene_id_upper:
                return symbol

        return None

    def get_ensembl(self, gene_id: str) -> Optional[str]:
        """Get Ensembl ID for a gene."""
        symbol = self.get_symbol(gene_id)
        if symbol:
            return self.symbol_to_ensembl.get(symbol)
        return None

    def get_entrez(self, gene_id: str) -> Optional[str]:
        """Get Entrez ID for a gene."""
        symbol = self.get_symbol(gene_id)
        if symbol:
            return self.symbol_to_entrez.get(symbol)
        return None

    def get_info(self, gene_id: str) -> Optional[GeneInfo]:
        """Get full gene information."""
        symbol = self.get_symbol(gene_id)
        if symbol:
            return self.genes.get(symbol)
        return None

    def get_coordinates(self, gene_id: str) -> Optional[Tuple[str, int, int]]:
        """Get genomic coordinates (chrom, start, end)."""
        info = self.get_info(gene_id)
        if info and info.chromosome and info.start and info.end:
            return (info.chromosome, info.start, info.end)
        return None

    def normalize_gene_list(
        self, gene_ids: List[str], report_missing: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        Normalize a list of gene identifiers to canonical symbols.

        Args:
            gene_ids: List of gene identifiers (any format)
            report_missing: Log warning for unmapped genes

        Returns:
            Tuple of (normalized_symbols, unmapped_ids)
        """
        normalized = []
        unmapped = []

        for gene_id in gene_ids:
            symbol = self.get_symbol(gene_id)
            if symbol:
                normalized.append(symbol)
            else:
                unmapped.append(gene_id)

        if report_missing and unmapped:
            logger.warning(f"Could not map {len(unmapped)} gene IDs: {unmapped[:5]}...")

        return normalized, unmapped

    def get_genes_in_region(
        self, chrom: str, start: int, end: int
    ) -> List[str]:
        """
        Get genes overlapping a genomic region.

        Args:
            chrom: Chromosome name
            start: Start position
            end: End position

        Returns:
            List of gene symbols
        """
        genes_in_region = []

        for symbol, info in self.genes.items():
            if info.chromosome != chrom:
                continue
            if info.start is None or info.end is None:
                continue
            # Check overlap
            if info.start <= end and info.end >= start:
                genes_in_region.append(symbol)

        return genes_in_region


class AnnotationLoader:
    """
    Loader for gene annotations and ID mappings.

    Supports multiple file formats:
    - NCBI gene_info format
    - Ensembl BioMart exports
    - HGNC downloads
    - Custom TSV/CSV formats
    """

    def __init__(self):
        """Initialize annotation loader."""
        self._db: Optional[GeneAnnotationDB] = None

    def load_ncbi_gene_info(
        self,
        gene_info_path: str,
        tax_id: str = "9606",  # Human
    ) -> GeneAnnotationDB:
        """
        Load NCBI gene_info file.

        File from: ftp://ftp.ncbi.nih.gov/gene/DATA/gene_info.gz

        Args:
            gene_info_path: Path to gene_info file (can be gzipped)
            tax_id: Taxonomy ID to filter (9606 for human)

        Returns:
            GeneAnnotationDB object
        """
        import gzip

        path = Path(gene_info_path)
        if not path.exists():
            raise FileNotFoundError(f"Gene info file not found: {gene_info_path}")

        open_func = gzip.open if str(path).endswith(".gz") else open
        mode = "rt" if str(path).endswith(".gz") else "r"

        genes: Dict[str, GeneInfo] = {}
        symbol_to_ensembl: Dict[str, str] = {}
        symbol_to_entrez: Dict[str, str] = {}
        ensembl_to_symbol: Dict[str, str] = {}
        entrez_to_symbol: Dict[str, str] = {}
        alias_to_symbol: Dict[str, str] = {}

        with open_func(path, mode) as f:
            header = f.readline().strip().split("\t")

            # Find column indices
            try:
                tax_col = header.index("#tax_id") if "#tax_id" in header else header.index("tax_id")
                entrez_col = header.index("GeneID")
                symbol_col = header.index("Symbol")
                synonyms_col = header.index("Synonyms")
                dbxrefs_col = header.index("dbXrefs")
                chrom_col = header.index("chromosome")
                name_col = header.index("description")
                type_col = header.index("type_of_gene")
            except ValueError as e:
                raise ValueError(f"Required column not found: {e}")

            for line in f:
                fields = line.strip().split("\t")

                # Filter by taxonomy
                if fields[tax_col] != tax_id:
                    continue

                entrez_id = fields[entrez_col]
                symbol = fields[symbol_col]

                if not symbol or symbol == "-":
                    continue

                # Parse aliases
                aliases = []
                if fields[synonyms_col] != "-":
                    aliases = fields[synonyms_col].split("|")

                # Parse Ensembl ID from dbXrefs
                ensembl_id = None
                if fields[dbxrefs_col] != "-":
                    for xref in fields[dbxrefs_col].split("|"):
                        if xref.startswith("Ensembl:"):
                            ensembl_id = xref.replace("Ensembl:", "")
                            break

                # Create gene info
                gene_info = GeneInfo(
                    symbol=symbol,
                    ensembl_id=ensembl_id,
                    entrez_id=entrez_id,
                    name=fields[name_col] if fields[name_col] != "-" else None,
                    chromosome=fields[chrom_col] if fields[chrom_col] != "-" else None,
                    biotype=fields[type_col] if fields[type_col] != "-" else None,
                    aliases=aliases,
                )

                genes[symbol] = gene_info

                # Build mappings
                symbol_to_entrez[symbol] = entrez_id
                entrez_to_symbol[entrez_id] = symbol

                if ensembl_id:
                    symbol_to_ensembl[symbol] = ensembl_id
                    ensembl_to_symbol[ensembl_id] = symbol

                for alias in aliases:
                    if alias not in alias_to_symbol:
                        alias_to_symbol[alias] = symbol

        self._db = GeneAnnotationDB(
            genes=genes,
            symbol_to_ensembl=symbol_to_ensembl,
            symbol_to_entrez=symbol_to_entrez,
            ensembl_to_symbol=ensembl_to_symbol,
            entrez_to_symbol=entrez_to_symbol,
            alias_to_symbol=alias_to_symbol,
            metadata={
                "source": "NCBI",
                "file": str(path),
                "tax_id": tax_id,
            },
        )

        logger.info(f"Loaded {len(genes)} genes from NCBI gene_info")

        return self._db

    def load_ensembl_biomart(
        self,
        biomart_path: str,
        symbol_col: str = "Gene name",
        ensembl_col: str = "Gene stable ID",
        entrez_col: str = "NCBI gene (formerly Entrezgene) ID",
        chrom_col: str = "Chromosome/scaffold name",
        start_col: str = "Gene start (bp)",
        end_col: str = "Gene end (bp)",
    ) -> GeneAnnotationDB:
        """
        Load Ensembl BioMart export.

        Args:
            biomart_path: Path to BioMart TSV export
            symbol_col: Column name for gene symbol
            ensembl_col: Column name for Ensembl ID
            entrez_col: Column name for Entrez ID
            chrom_col: Column name for chromosome
            start_col: Column name for start position
            end_col: Column name for end position

        Returns:
            GeneAnnotationDB object
        """
        import csv

        path = Path(biomart_path)
        if not path.exists():
            raise FileNotFoundError(f"BioMart file not found: {biomart_path}")

        genes: Dict[str, GeneInfo] = {}
        symbol_to_ensembl: Dict[str, str] = {}
        symbol_to_entrez: Dict[str, str] = {}
        ensembl_to_symbol: Dict[str, str] = {}
        entrez_to_symbol: Dict[str, str] = {}

        with open(path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                symbol = row.get(symbol_col, "").strip()
                ensembl_id = row.get(ensembl_col, "").strip()

                if not symbol:
                    continue

                entrez_id = row.get(entrez_col, "").strip()
                chrom = row.get(chrom_col, "").strip()

                start = None
                end = None
                try:
                    start = int(row.get(start_col, ""))
                    end = int(row.get(end_col, ""))
                except (ValueError, TypeError):
                    pass

                gene_info = GeneInfo(
                    symbol=symbol,
                    ensembl_id=ensembl_id if ensembl_id else None,
                    entrez_id=entrez_id if entrez_id else None,
                    chromosome=chrom if chrom else None,
                    start=start,
                    end=end,
                )

                genes[symbol] = gene_info

                if ensembl_id:
                    symbol_to_ensembl[symbol] = ensembl_id
                    ensembl_to_symbol[ensembl_id] = symbol

                if entrez_id:
                    symbol_to_entrez[symbol] = entrez_id
                    entrez_to_symbol[entrez_id] = symbol

        self._db = GeneAnnotationDB(
            genes=genes,
            symbol_to_ensembl=symbol_to_ensembl,
            symbol_to_entrez=symbol_to_entrez,
            ensembl_to_symbol=ensembl_to_symbol,
            entrez_to_symbol=entrez_to_symbol,
            metadata={
                "source": "Ensembl BioMart",
                "file": str(path),
            },
        )

        logger.info(f"Loaded {len(genes)} genes from Ensembl BioMart")

        return self._db

    def load_hgnc(
        self,
        hgnc_path: str,
    ) -> GeneAnnotationDB:
        """
        Load HGNC gene names database.

        File from: https://www.genenames.org/download/custom/

        Args:
            hgnc_path: Path to HGNC TSV file

        Returns:
            GeneAnnotationDB object
        """
        import csv

        path = Path(hgnc_path)
        if not path.exists():
            raise FileNotFoundError(f"HGNC file not found: {hgnc_path}")

        genes: Dict[str, GeneInfo] = {}
        symbol_to_ensembl: Dict[str, str] = {}
        symbol_to_entrez: Dict[str, str] = {}
        ensembl_to_symbol: Dict[str, str] = {}
        entrez_to_symbol: Dict[str, str] = {}
        alias_to_symbol: Dict[str, str] = {}

        with open(path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                symbol = row.get("symbol", "").strip()
                if not symbol:
                    continue

                ensembl_id = row.get("ensembl_gene_id", "").strip()
                entrez_id = row.get("entrez_id", "").strip()
                name = row.get("name", "").strip()

                # Parse aliases
                aliases = []
                alias_str = row.get("alias_symbol", "")
                if alias_str:
                    aliases = [a.strip() for a in alias_str.split("|") if a.strip()]

                # Parse previous symbols
                prev_str = row.get("prev_symbol", "")
                if prev_str:
                    prev_symbols = [p.strip() for p in prev_str.split("|") if p.strip()]
                    aliases.extend(prev_symbols)

                gene_info = GeneInfo(
                    symbol=symbol,
                    ensembl_id=ensembl_id if ensembl_id else None,
                    entrez_id=entrez_id if entrez_id else None,
                    name=name if name else None,
                    aliases=aliases,
                )

                genes[symbol] = gene_info

                if ensembl_id:
                    symbol_to_ensembl[symbol] = ensembl_id
                    ensembl_to_symbol[ensembl_id] = symbol

                if entrez_id:
                    symbol_to_entrez[symbol] = entrez_id
                    entrez_to_symbol[entrez_id] = symbol

                for alias in aliases:
                    if alias not in alias_to_symbol:
                        alias_to_symbol[alias] = symbol

        self._db = GeneAnnotationDB(
            genes=genes,
            symbol_to_ensembl=symbol_to_ensembl,
            symbol_to_entrez=symbol_to_entrez,
            ensembl_to_symbol=ensembl_to_symbol,
            entrez_to_symbol=entrez_to_symbol,
            alias_to_symbol=alias_to_symbol,
            metadata={
                "source": "HGNC",
                "file": str(path),
            },
        )

        logger.info(f"Loaded {len(genes)} genes from HGNC")

        return self._db

    def load_from_csv(
        self,
        csv_path: str,
        symbol_col: str = "symbol",
        ensembl_col: Optional[str] = "ensembl_id",
        entrez_col: Optional[str] = "entrez_id",
    ) -> GeneAnnotationDB:
        """
        Load gene annotations from custom CSV file.

        Args:
            csv_path: Path to CSV file
            symbol_col: Column name for gene symbol
            ensembl_col: Column name for Ensembl ID
            entrez_col: Column name for Entrez ID

        Returns:
            GeneAnnotationDB object
        """
        import csv

        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        genes: Dict[str, GeneInfo] = {}
        symbol_to_ensembl: Dict[str, str] = {}
        symbol_to_entrez: Dict[str, str] = {}
        ensembl_to_symbol: Dict[str, str] = {}
        entrez_to_symbol: Dict[str, str] = {}

        with open(path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                symbol = row.get(symbol_col, "").strip()
                if not symbol:
                    continue

                ensembl_id = row.get(ensembl_col, "").strip() if ensembl_col else None
                entrez_id = row.get(entrez_col, "").strip() if entrez_col else None

                gene_info = GeneInfo(
                    symbol=symbol,
                    ensembl_id=ensembl_id if ensembl_id else None,
                    entrez_id=entrez_id if entrez_id else None,
                )

                genes[symbol] = gene_info

                if ensembl_id:
                    symbol_to_ensembl[symbol] = ensembl_id
                    ensembl_to_symbol[ensembl_id] = symbol

                if entrez_id:
                    symbol_to_entrez[symbol] = entrez_id
                    entrez_to_symbol[entrez_id] = symbol

        self._db = GeneAnnotationDB(
            genes=genes,
            symbol_to_ensembl=symbol_to_ensembl,
            symbol_to_entrez=symbol_to_entrez,
            ensembl_to_symbol=ensembl_to_symbol,
            entrez_to_symbol=entrez_to_symbol,
            metadata={
                "source": "CSV",
                "file": str(path),
            },
        )

        logger.info(f"Loaded {len(genes)} genes from CSV")

        return self._db

    def get_database(self) -> GeneAnnotationDB:
        """Get the loaded annotation database."""
        if self._db is None:
            raise RuntimeError("No annotation database loaded.")
        return self._db

    def normalize_gene_ids(
        self, gene_ids: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Normalize gene identifiers to canonical symbols.

        Args:
            gene_ids: List of gene identifiers

        Returns:
            Tuple of (normalized_symbols, unmapped_ids)
        """
        if self._db is None:
            raise RuntimeError("No annotation database loaded.")

        return self._db.normalize_gene_list(gene_ids)
