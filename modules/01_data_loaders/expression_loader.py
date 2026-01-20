"""
Expression Loader

Handles loading of developmental expression data, primarily from BrainSpan.
BrainSpan provides gene expression data across brain regions and developmental stages.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
import logging

import numpy as np

logger = logging.getLogger(__name__)


# BrainSpan developmental stages
BRAINSPAN_STAGES = [
    "8pcw", "9pcw", "12pcw", "13pcw", "16pcw", "17pcw", "19pcw", "21pcw", "24pcw",
    "25pcw", "26pcw", "35pcw", "37pcw",  # Prenatal
    "4mos", "10mos",  # Infant
    "1yrs", "2yrs", "3yrs", "4yrs",  # Early childhood
    "8yrs", "11yrs", "13yrs", "15yrs", "18yrs", "19yrs",  # Adolescence
    "21yrs", "23yrs", "30yrs", "36yrs", "37yrs", "40yrs",  # Adult
]

# Prenatal stages (post-conception weeks)
PRENATAL_STAGES = [
    "8pcw", "9pcw", "12pcw", "13pcw", "16pcw", "17pcw", "19pcw",
    "21pcw", "24pcw", "25pcw", "26pcw", "35pcw", "37pcw"
]

# BrainSpan brain regions
BRAINSPAN_REGIONS = [
    "A1C",   # Primary auditory cortex
    "AMY",   # Amygdala
    "CBC",   # Cerebellar cortex
    "DFC",   # Dorsolateral prefrontal cortex
    "HIP",   # Hippocampus
    "IPC",   # Posterior inferior parietal cortex
    "ITC",   # Inferior temporal cortex
    "M1C",   # Primary motor cortex
    "MD",    # Mediodorsal nucleus of thalamus
    "MFC",   # Medial prefrontal cortex
    "OFC",   # Orbital prefrontal cortex
    "S1C",   # Primary somatosensory cortex
    "STC",   # Superior temporal cortex
    "STR",   # Striatum
    "V1C",   # Primary visual cortex
    "VFC",   # Ventrolateral prefrontal cortex
]

# Cortical regions
CORTICAL_REGIONS = [
    "A1C", "DFC", "IPC", "ITC", "M1C", "MFC", "OFC", "S1C", "STC", "V1C", "VFC"
]


@dataclass
class DevelopmentalExpression:
    """
    Developmental gene expression data (e.g., BrainSpan).

    Attributes:
        genes: List of gene identifiers
        stages: List of developmental stages
        regions: List of brain regions
        expression: 3D array of shape (n_genes, n_stages, n_regions)
        gene_index: Mapping from gene ID to index
        stage_index: Mapping from stage to index
        region_index: Mapping from region to index
        metadata: Additional metadata
    """

    genes: List[str]
    stages: List[str]
    regions: List[str]
    expression: np.ndarray  # shape: (n_genes, n_stages, n_regions)
    gene_index: Dict[str, int] = field(default_factory=dict)
    stage_index: Dict[str, int] = field(default_factory=dict)
    region_index: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """Build index mappings if not provided."""
        if not self.gene_index:
            self.gene_index = {gene: i for i, gene in enumerate(self.genes)}
        if not self.stage_index:
            self.stage_index = {stage: i for i, stage in enumerate(self.stages)}
        if not self.region_index:
            self.region_index = {region: i for i, region in enumerate(self.regions)}

    def __len__(self) -> int:
        return len(self.genes)

    def get_expression(
        self, gene_id: str, stage: Optional[str] = None, region: Optional[str] = None
    ) -> np.ndarray:
        """
        Get expression for a gene, optionally filtered by stage/region.

        Args:
            gene_id: Gene identifier
            stage: Optional developmental stage
            region: Optional brain region

        Returns:
            Expression values (scalar, 1D, or 2D array depending on filters)
        """
        if gene_id not in self.gene_index:
            raise KeyError(f"Gene not found: {gene_id}")

        gene_idx = self.gene_index[gene_id]
        expr = self.expression[gene_idx]

        if stage is not None and region is not None:
            stage_idx = self.stage_index[stage]
            region_idx = self.region_index[region]
            return expr[stage_idx, region_idx]
        elif stage is not None:
            stage_idx = self.stage_index[stage]
            return expr[stage_idx, :]
        elif region is not None:
            region_idx = self.region_index[region]
            return expr[:, region_idx]
        else:
            return expr

    def get_prenatal_expression(self, gene_id: str) -> np.ndarray:
        """
        Get expression during prenatal stages.

        Args:
            gene_id: Gene identifier

        Returns:
            Expression array of shape (n_prenatal_stages, n_regions)
        """
        if gene_id not in self.gene_index:
            raise KeyError(f"Gene not found: {gene_id}")

        gene_idx = self.gene_index[gene_id]
        prenatal_indices = [
            self.stage_index[s] for s in PRENATAL_STAGES if s in self.stage_index
        ]
        return self.expression[gene_idx, prenatal_indices, :]

    def get_cortical_expression(self, gene_id: str, stage: str) -> float:
        """
        Get mean cortical expression for a gene at a specific stage.

        Args:
            gene_id: Gene identifier
            stage: Developmental stage

        Returns:
            Mean expression across cortical regions
        """
        if gene_id not in self.gene_index:
            raise KeyError(f"Gene not found: {gene_id}")

        gene_idx = self.gene_index[gene_id]
        stage_idx = self.stage_index[stage]
        cortical_indices = [
            self.region_index[r] for r in CORTICAL_REGIONS if r in self.region_index
        ]

        return float(np.mean(self.expression[gene_idx, stage_idx, cortical_indices]))

    def get_mean_expression(self, gene_id: str) -> float:
        """Get mean expression across all stages and regions."""
        if gene_id not in self.gene_index:
            raise KeyError(f"Gene not found: {gene_id}")

        gene_idx = self.gene_index[gene_id]
        return float(np.nanmean(self.expression[gene_idx]))

    def get_max_expression_stage(self, gene_id: str, region: str) -> Tuple[str, float]:
        """
        Get the developmental stage with maximum expression.

        Args:
            gene_id: Gene identifier
            region: Brain region

        Returns:
            Tuple of (stage, expression_value)
        """
        expr = self.get_expression(gene_id, region=region)
        max_idx = int(np.nanargmax(expr))
        return self.stages[max_idx], float(expr[max_idx])

    def is_prenatally_expressed(
        self, gene_id: str, threshold: float = 1.0
    ) -> bool:
        """
        Check if gene is expressed during prenatal development.

        Args:
            gene_id: Gene identifier
            threshold: Minimum expression threshold (log2 RPKM or similar)

        Returns:
            True if gene is expressed prenatally above threshold
        """
        try:
            prenatal_expr = self.get_prenatal_expression(gene_id)
            return bool(np.nanmax(prenatal_expr) >= threshold)
        except KeyError:
            return False


class ExpressionLoader:
    """
    Loader for developmental expression data.

    Primarily supports BrainSpan atlas format but can be extended
    for other developmental expression datasets.
    """

    def __init__(self, log_transform: bool = True):
        """
        Initialize expression loader.

        Args:
            log_transform: Whether to log2-transform expression values
        """
        self.log_transform = log_transform
        self._expression_data: Optional[DevelopmentalExpression] = None

    def load_brainspan(self, data_dir: str) -> DevelopmentalExpression:
        """
        Load BrainSpan developmental expression data.

        Expected files in data_dir:
        - expression_matrix.csv (or .txt): Gene x Sample expression matrix
        - columns_metadata.csv: Sample metadata (donor, age, region)
        - rows_metadata.csv: Gene metadata (gene_symbol, gene_id)

        Args:
            data_dir: Path to BrainSpan data directory

        Returns:
            DevelopmentalExpression object
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"BrainSpan data directory not found: {data_dir}")

        # Try to find expression matrix
        expr_file = self._find_file(data_path, ["expression_matrix", "expression"])
        cols_file = self._find_file(data_path, ["columns_metadata", "column_metadata", "samples"])
        rows_file = self._find_file(data_path, ["rows_metadata", "row_metadata", "genes"])

        # Load expression matrix
        expression_df = self._load_matrix(expr_file)

        # Load metadata
        cols_meta = self._load_metadata(cols_file)
        rows_meta = self._load_metadata(rows_file)

        # Parse into structured format
        genes = self._extract_genes(rows_meta)
        stages, regions, sample_to_stage_region = self._extract_stages_regions(cols_meta)

        # Build 3D expression tensor
        expression_tensor = self._build_expression_tensor(
            expression_df, genes, stages, regions, sample_to_stage_region
        )

        if self.log_transform:
            # Add pseudocount and log transform
            expression_tensor = np.log2(expression_tensor + 1)

        self._expression_data = DevelopmentalExpression(
            genes=genes,
            stages=stages,
            regions=regions,
            expression=expression_tensor,
            metadata={
                "source": "BrainSpan",
                "data_dir": str(data_path),
                "log_transformed": self.log_transform,
            },
        )

        logger.info(
            f"Loaded BrainSpan data: {len(genes)} genes, "
            f"{len(stages)} stages, {len(regions)} regions"
        )

        return self._expression_data

    def _find_file(self, data_path: Path, prefixes: List[str]) -> Path:
        """Find a file matching one of the prefixes."""
        for prefix in prefixes:
            for ext in [".csv", ".txt", ".tsv"]:
                candidate = data_path / f"{prefix}{ext}"
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(
            f"Could not find file with prefixes {prefixes} in {data_path}"
        )

    def _load_matrix(self, file_path: Path) -> np.ndarray:
        """Load expression matrix from file."""
        import csv

        delimiter = "\t" if file_path.suffix == ".tsv" else ","

        with open(file_path, "r") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader)  # Skip header
            data = []
            for row in reader:
                # Skip first column if it's gene names
                values = row[1:] if len(row) > len(header) else row
                data.append([float(v) if v and v != "NA" else np.nan for v in values])

        return np.array(data)

    def _load_metadata(self, file_path: Path) -> List[Dict[str, str]]:
        """Load metadata from file."""
        import csv

        delimiter = "\t" if file_path.suffix == ".tsv" else ","

        metadata = []
        with open(file_path, "r") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                metadata.append(dict(row))

        return metadata

    def _extract_genes(self, rows_meta: List[Dict[str, str]]) -> List[str]:
        """Extract gene symbols from row metadata."""
        genes = []
        for row in rows_meta:
            # Try different possible column names
            gene = (
                row.get("gene_symbol")
                or row.get("gene_name")
                or row.get("symbol")
                or row.get("Gene")
            )
            if gene:
                genes.append(gene)
        return genes

    def _extract_stages_regions(
        self, cols_meta: List[Dict[str, str]]
    ) -> Tuple[List[str], List[str], Dict[int, Tuple[str, str]]]:
        """Extract stages and regions from column metadata."""
        stages_set: Set[str] = set()
        regions_set: Set[str] = set()
        sample_to_stage_region: Dict[int, Tuple[str, str]] = {}

        for i, row in enumerate(cols_meta):
            # Try different possible column names for age/stage
            age = (
                row.get("age")
                or row.get("developmental_stage")
                or row.get("stage")
            )
            # Try different possible column names for region
            region = (
                row.get("structure_acronym")
                or row.get("region")
                or row.get("brain_region")
            )

            if age and region:
                stages_set.add(age)
                regions_set.add(region)
                sample_to_stage_region[i] = (age, region)

        # Sort stages by developmental order
        stages = self._sort_stages(list(stages_set))
        regions = sorted(list(regions_set))

        return stages, regions, sample_to_stage_region

    def _sort_stages(self, stages: List[str]) -> List[str]:
        """Sort developmental stages in chronological order."""
        def stage_key(stage: str) -> Tuple[int, int]:
            # Extract numeric value and unit
            stage_lower = stage.lower()
            if "pcw" in stage_lower:
                num = int("".join(filter(str.isdigit, stage)))
                return (0, num)  # Prenatal
            elif "mos" in stage_lower:
                num = int("".join(filter(str.isdigit, stage)))
                return (1, num)  # Months
            elif "yrs" in stage_lower or "y" in stage_lower:
                num = int("".join(filter(str.isdigit, stage)))
                return (2, num)  # Years
            else:
                return (3, 0)  # Unknown

        return sorted(stages, key=stage_key)

    def _build_expression_tensor(
        self,
        expression_df: np.ndarray,
        genes: List[str],
        stages: List[str],
        regions: List[str],
        sample_to_stage_region: Dict[int, Tuple[str, str]],
    ) -> np.ndarray:
        """Build 3D expression tensor from 2D matrix."""
        n_genes = len(genes)
        n_stages = len(stages)
        n_regions = len(regions)

        # Initialize with NaN
        tensor = np.full((n_genes, n_stages, n_regions), np.nan)

        stage_idx = {s: i for i, s in enumerate(stages)}
        region_idx = {r: i for i, r in enumerate(regions)}

        # Fill in values
        for sample_idx, (stage, region) in sample_to_stage_region.items():
            if sample_idx >= expression_df.shape[1]:
                continue
            if stage in stage_idx and region in region_idx:
                si = stage_idx[stage]
                ri = region_idx[region]
                tensor[:, si, ri] = expression_df[:, sample_idx]

        return tensor

    def get_expression_by_stage(self, gene_id: str, stage: str) -> float:
        """
        Get mean expression for a gene at a specific developmental stage.

        Args:
            gene_id: Gene identifier
            stage: Developmental stage

        Returns:
            Mean expression across all regions
        """
        if self._expression_data is None:
            raise RuntimeError("No expression data loaded. Call load_brainspan first.")

        expr = self._expression_data.get_expression(gene_id, stage=stage)
        return float(np.nanmean(expr))

    def get_prenatal_expressed_genes(
        self, threshold: float = 1.0
    ) -> List[str]:
        """
        Get list of genes expressed during prenatal development.

        Args:
            threshold: Minimum expression threshold

        Returns:
            List of gene IDs expressed prenatally
        """
        if self._expression_data is None:
            raise RuntimeError("No expression data loaded. Call load_brainspan first.")

        expressed = []
        for gene in self._expression_data.genes:
            if self._expression_data.is_prenatally_expressed(gene, threshold):
                expressed.append(gene)

        return expressed

    def get_stage_specific_genes(
        self,
        stage: str,
        top_n: int = 100,
        min_fold_change: float = 2.0,
    ) -> List[Tuple[str, float]]:
        """
        Get genes specifically expressed at a developmental stage.

        Args:
            stage: Target developmental stage
            top_n: Number of top genes to return
            min_fold_change: Minimum fold change vs other stages

        Returns:
            List of (gene_id, fold_change) tuples
        """
        if self._expression_data is None:
            raise RuntimeError("No expression data loaded. Call load_brainspan first.")

        stage_idx = self._expression_data.stage_index[stage]
        results = []

        for gene_idx, gene in enumerate(self._expression_data.genes):
            # Mean expression at target stage
            target_expr = np.nanmean(
                self._expression_data.expression[gene_idx, stage_idx, :]
            )

            # Mean expression at other stages
            other_stages = [
                i for i in range(len(self._expression_data.stages)) if i != stage_idx
            ]
            other_expr = np.nanmean(
                self._expression_data.expression[gene_idx, other_stages, :]
            )

            # Calculate fold change
            if other_expr > 0:
                fold_change = target_expr / other_expr
                if fold_change >= min_fold_change:
                    results.append((gene, fold_change))

        # Sort by fold change and return top N
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    def load_from_csv(
        self,
        expression_path: str,
        genes_col: str = "gene_symbol",
        stage_cols: Optional[Dict[str, str]] = None,
    ) -> DevelopmentalExpression:
        """
        Load expression data from a simple CSV format.

        Args:
            expression_path: Path to CSV file
            genes_col: Column name containing gene symbols
            stage_cols: Mapping from column names to stage names

        Returns:
            DevelopmentalExpression object
        """
        import csv

        path = Path(expression_path)
        if not path.exists():
            raise FileNotFoundError(f"Expression file not found: {expression_path}")

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError("Empty expression file")

        # Get genes
        genes = [row[genes_col] for row in rows if row.get(genes_col)]

        # Determine stage columns
        if stage_cols is None:
            # Assume all non-gene columns are expression values
            all_cols = set(rows[0].keys()) - {genes_col}
            stage_cols = {col: col for col in all_cols}

        stages = list(stage_cols.values())
        regions = ["bulk"]  # Single region for bulk data

        # Build expression matrix
        n_genes = len(genes)
        n_stages = len(stages)
        expression = np.zeros((n_genes, n_stages, 1))

        col_to_stage = {v: k for k, v in stage_cols.items()}
        for gene_idx, row in enumerate(rows):
            for stage_idx, stage in enumerate(stages):
                col = col_to_stage.get(stage, stage)
                val = row.get(col, "0")
                try:
                    expression[gene_idx, stage_idx, 0] = float(val) if val else 0.0
                except ValueError:
                    expression[gene_idx, stage_idx, 0] = 0.0

        if self.log_transform:
            expression = np.log2(expression + 1)

        return DevelopmentalExpression(
            genes=genes,
            stages=stages,
            regions=regions,
            expression=expression,
            metadata={
                "source": "CSV",
                "file": str(path),
                "log_transformed": self.log_transform,
            },
        )
