"""
Single Cell Loader

Handles loading of single-cell RNA sequencing atlases, primarily:
- Allen Brain Atlas
- Other cortical single-cell atlases (e.g., developmental brain atlases)

Supports h5ad (AnnData) format which is the standard for single-cell data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
import logging

import numpy as np

logger = logging.getLogger(__name__)


# Common brain cell types
NEURONAL_CELL_TYPES = [
    "excitatory_neuron",
    "inhibitory_neuron",
    "glutamatergic",
    "GABAergic",
    "dopaminergic",
    "serotonergic",
    "cholinergic",
]

GLIAL_CELL_TYPES = [
    "astrocyte",
    "oligodendrocyte",
    "microglia",
    "OPC",  # Oligodendrocyte precursor cell
]

EXCITATORY_SUBTYPES = [
    "L2/3_IT",  # Layer 2/3 intratelencephalic
    "L4_IT",
    "L5_IT",
    "L5_ET",    # Layer 5 extratelencephalic
    "L5/6_NP",  # Near-projecting
    "L6_IT",
    "L6_CT",    # Corticothalamic
    "L6b",
]

INHIBITORY_SUBTYPES = [
    "Pvalb",    # Parvalbumin
    "Sst",      # Somatostatin
    "Vip",      # Vasoactive intestinal peptide
    "Lamp5",
    "Sncg",
]


@dataclass
class SingleCellAtlas:
    """
    Single-cell expression atlas.

    Attributes:
        genes: List of gene identifiers
        cell_types: List of cell type names
        expression: 2D array of shape (n_genes, n_cell_types) - mean expression per cell type
        cell_type_hierarchy: Hierarchical grouping of cell types
        cell_counts: Number of cells per cell type
        metadata: Additional metadata
    """

    genes: List[str]
    cell_types: List[str]
    expression: np.ndarray  # shape: (n_genes, n_cell_types)
    cell_type_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    cell_counts: Dict[str, int] = field(default_factory=dict)
    gene_index: Dict[str, int] = field(default_factory=dict)
    cell_type_index: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """Build index mappings if not provided."""
        if not self.gene_index:
            self.gene_index = {gene: i for i, gene in enumerate(self.genes)}
        if not self.cell_type_index:
            self.cell_type_index = {ct: i for i, ct in enumerate(self.cell_types)}

    def __len__(self) -> int:
        return len(self.genes)

    def get_expression(self, gene_id: str, cell_type: Optional[str] = None) -> np.ndarray:
        """
        Get expression for a gene across cell types.

        Args:
            gene_id: Gene identifier
            cell_type: Optional specific cell type

        Returns:
            Expression values (scalar if cell_type specified, array otherwise)
        """
        if gene_id not in self.gene_index:
            raise KeyError(f"Gene not found: {gene_id}")

        gene_idx = self.gene_index[gene_id]

        if cell_type is not None:
            if cell_type not in self.cell_type_index:
                raise KeyError(f"Cell type not found: {cell_type}")
            ct_idx = self.cell_type_index[cell_type]
            return self.expression[gene_idx, ct_idx]

        return self.expression[gene_idx, :]

    def get_expression_by_cell_type(self, gene_id: str) -> Dict[str, float]:
        """
        Get expression for a gene as a dictionary by cell type.

        Args:
            gene_id: Gene identifier

        Returns:
            Dictionary mapping cell type to expression value
        """
        expr = self.get_expression(gene_id)
        return {ct: float(expr[i]) for i, ct in enumerate(self.cell_types)}

    def get_cell_type_specific_genes(
        self,
        cell_type: str,
        fold_change: float = 2.0,
        min_expression: float = 1.0,
        top_n: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Get genes specifically expressed in a cell type.

        Args:
            cell_type: Target cell type
            fold_change: Minimum fold change vs other cell types
            min_expression: Minimum expression in target cell type
            top_n: Optional limit on number of genes to return

        Returns:
            List of (gene_id, fold_change) tuples sorted by fold change
        """
        if cell_type not in self.cell_type_index:
            raise KeyError(f"Cell type not found: {cell_type}")

        ct_idx = self.cell_type_index[cell_type]
        other_idx = [i for i in range(len(self.cell_types)) if i != ct_idx]

        results = []
        for gene_idx, gene in enumerate(self.genes):
            target_expr = self.expression[gene_idx, ct_idx]

            if target_expr < min_expression:
                continue

            # Mean expression in other cell types
            other_expr = np.mean(self.expression[gene_idx, other_idx])

            if other_expr > 0:
                fc = target_expr / other_expr
                if fc >= fold_change:
                    results.append((gene, fc))

        results.sort(key=lambda x: x[1], reverse=True)

        if top_n is not None:
            results = results[:top_n]

        return results

    def get_marker_genes(
        self, cell_type: str, n_markers: int = 50
    ) -> List[str]:
        """
        Get marker genes for a cell type.

        Args:
            cell_type: Target cell type
            n_markers: Number of markers to return

        Returns:
            List of marker gene IDs
        """
        specific_genes = self.get_cell_type_specific_genes(
            cell_type, fold_change=1.5, top_n=n_markers
        )
        return [g[0] for g in specific_genes]

    def is_enriched_in(
        self,
        gene_id: str,
        cell_type: str,
        fold_change_threshold: float = 2.0,
    ) -> bool:
        """
        Check if a gene is enriched in a specific cell type.

        Args:
            gene_id: Gene identifier
            cell_type: Target cell type
            fold_change_threshold: Minimum fold change to be considered enriched

        Returns:
            True if gene is enriched in the cell type
        """
        if gene_id not in self.gene_index:
            return False
        if cell_type not in self.cell_type_index:
            return False

        gene_idx = self.gene_index[gene_id]
        ct_idx = self.cell_type_index[cell_type]

        target_expr = self.expression[gene_idx, ct_idx]
        other_idx = [i for i in range(len(self.cell_types)) if i != ct_idx]
        other_expr = np.mean(self.expression[gene_idx, other_idx])

        if other_expr == 0:
            return target_expr > 0

        return (target_expr / other_expr) >= fold_change_threshold

    def get_genes_enriched_in(
        self, cell_type: str, fold_change_threshold: float = 2.0
    ) -> Set[str]:
        """
        Get all genes enriched in a cell type.

        Args:
            cell_type: Target cell type
            fold_change_threshold: Minimum fold change

        Returns:
            Set of gene IDs
        """
        specific = self.get_cell_type_specific_genes(
            cell_type, fold_change=fold_change_threshold
        )
        return {g[0] for g in specific}

    def get_broad_cell_type(self, cell_type: str) -> Optional[str]:
        """
        Get the broad cell type category for a specific cell type.

        Args:
            cell_type: Specific cell type

        Returns:
            Broad cell type category or None
        """
        for broad, specific_list in self.cell_type_hierarchy.items():
            if cell_type in specific_list or cell_type == broad:
                return broad
        return None


class SingleCellLoader:
    """
    Loader for single-cell expression atlases.

    Supports:
    - h5ad (AnnData) format
    - CSV/TSV aggregated expression matrices
    - Allen Brain Atlas format
    """

    def __init__(self, log_transform: bool = True):
        """
        Initialize single-cell loader.

        Args:
            log_transform: Whether to log-transform expression values
        """
        self.log_transform = log_transform
        self._atlas: Optional[SingleCellAtlas] = None

    def load_h5ad(
        self,
        h5ad_path: str,
        cell_type_key: str = "cell_type",
        aggregate: bool = True,
    ) -> SingleCellAtlas:
        """
        Load single-cell atlas from h5ad (AnnData) format.

        Args:
            h5ad_path: Path to h5ad file
            cell_type_key: Key in obs containing cell type annotations
            aggregate: Whether to aggregate to mean expression per cell type

        Returns:
            SingleCellAtlas object
        """
        path = Path(h5ad_path)
        if not path.exists():
            raise FileNotFoundError(f"h5ad file not found: {h5ad_path}")

        try:
            import anndata
        except ImportError:
            raise ImportError(
                "anndata package required for h5ad loading. "
                "Install with: pip install anndata"
            )

        logger.info(f"Loading h5ad file: {h5ad_path}")
        adata = anndata.read_h5ad(h5ad_path)

        # Get genes
        genes = list(adata.var_names)

        # Get cell types
        if cell_type_key not in adata.obs.columns:
            raise ValueError(
                f"Cell type key '{cell_type_key}' not found in obs. "
                f"Available keys: {list(adata.obs.columns)}"
            )

        cell_types = list(adata.obs[cell_type_key].unique())
        cell_types.sort()

        # Aggregate expression per cell type
        if aggregate:
            expression, cell_counts = self._aggregate_by_cell_type(
                adata, cell_type_key, cell_types
            )
        else:
            # Use raw expression (will be very large)
            expression = adata.X
            if hasattr(expression, "toarray"):
                expression = expression.toarray()
            cell_counts = {ct: 1 for ct in cell_types}

        if self.log_transform and not aggregate:
            expression = np.log2(expression + 1)

        # Build cell type hierarchy
        hierarchy = self._infer_hierarchy(cell_types)

        self._atlas = SingleCellAtlas(
            genes=genes,
            cell_types=cell_types,
            expression=expression,
            cell_type_hierarchy=hierarchy,
            cell_counts=cell_counts,
            metadata={
                "source": "h5ad",
                "file": str(path),
                "n_cells": adata.n_obs,
                "cell_type_key": cell_type_key,
            },
        )

        logger.info(
            f"Loaded single-cell atlas: {len(genes)} genes, "
            f"{len(cell_types)} cell types"
        )

        return self._atlas

    def _aggregate_by_cell_type(
        self,
        adata,
        cell_type_key: str,
        cell_types: List[str],
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """Aggregate expression by cell type (mean expression)."""
        n_genes = adata.n_vars
        n_cell_types = len(cell_types)

        expression = np.zeros((n_genes, n_cell_types))
        cell_counts = {}

        for ct_idx, cell_type in enumerate(cell_types):
            mask = adata.obs[cell_type_key] == cell_type
            subset = adata[mask]
            cell_counts[cell_type] = subset.n_obs

            if subset.n_obs > 0:
                # Get expression matrix
                X = subset.X
                if hasattr(X, "toarray"):
                    X = X.toarray()

                # Mean expression
                mean_expr = np.mean(X, axis=0)
                if len(mean_expr.shape) > 1:
                    mean_expr = mean_expr.flatten()

                expression[:, ct_idx] = mean_expr

        if self.log_transform:
            expression = np.log2(expression + 1)

        return expression, cell_counts

    def _infer_hierarchy(self, cell_types: List[str]) -> Dict[str, List[str]]:
        """Infer cell type hierarchy from names."""
        hierarchy: Dict[str, List[str]] = defaultdict(list)

        for ct in cell_types:
            ct_lower = ct.lower()

            # Neuronal types
            if any(x in ct_lower for x in ["excitatory", "glutamat", "l2", "l3", "l4", "l5", "l6", "_it", "_et", "_ct"]):
                hierarchy["excitatory_neuron"].append(ct)
            elif any(x in ct_lower for x in ["inhibitory", "gaba", "pvalb", "sst", "vip", "lamp5", "sncg"]):
                hierarchy["inhibitory_neuron"].append(ct)
            elif "neuron" in ct_lower:
                hierarchy["neuron"].append(ct)

            # Glial types
            elif "astro" in ct_lower:
                hierarchy["astrocyte"].append(ct)
            elif "oligo" in ct_lower and "opc" not in ct_lower:
                hierarchy["oligodendrocyte"].append(ct)
            elif "opc" in ct_lower:
                hierarchy["OPC"].append(ct)
            elif "microglia" in ct_lower:
                hierarchy["microglia"].append(ct)

            # Other
            elif "endo" in ct_lower:
                hierarchy["endothelial"].append(ct)
            elif "peri" in ct_lower:
                hierarchy["pericyte"].append(ct)

        return dict(hierarchy)

    def load_allen_brain(self, h5ad_path: str) -> SingleCellAtlas:
        """
        Load Allen Brain Atlas single-cell data.

        Allen Brain Atlas uses standard h5ad format with specific cell type naming.

        Args:
            h5ad_path: Path to Allen Brain h5ad file

        Returns:
            SingleCellAtlas object
        """
        # Allen Brain typically uses "cluster" or "cell_type" as the key
        for key in ["cell_type", "cluster", "class", "subclass"]:
            try:
                return self.load_h5ad(h5ad_path, cell_type_key=key)
            except ValueError:
                continue

        raise ValueError(
            "Could not find cell type annotation in Allen Brain file. "
            "Please specify cell_type_key manually."
        )

    def load_from_csv(
        self,
        csv_path: str,
        gene_col: str = "gene",
        cell_type_cols: Optional[List[str]] = None,
    ) -> SingleCellAtlas:
        """
        Load aggregated expression from CSV format.

        Expected format: gene column + one column per cell type with mean expression.

        Args:
            csv_path: Path to CSV file
            gene_col: Column name containing gene symbols
            cell_type_cols: Optional list of cell type column names

        Returns:
            SingleCellAtlas object
        """
        import csv

        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            if header is None:
                raise ValueError("Empty CSV file")

            # Determine cell type columns
            if cell_type_cols is None:
                cell_type_cols = [c for c in header if c != gene_col]

            rows = list(reader)

        genes = [row[gene_col] for row in rows if row.get(gene_col)]
        n_genes = len(genes)
        n_cell_types = len(cell_type_cols)

        expression = np.zeros((n_genes, n_cell_types))

        for gene_idx, row in enumerate(rows):
            for ct_idx, ct in enumerate(cell_type_cols):
                val = row.get(ct, "0")
                try:
                    expression[gene_idx, ct_idx] = float(val) if val else 0.0
                except ValueError:
                    expression[gene_idx, ct_idx] = 0.0

        if self.log_transform:
            expression = np.log2(expression + 1)

        hierarchy = self._infer_hierarchy(cell_type_cols)

        self._atlas = SingleCellAtlas(
            genes=genes,
            cell_types=cell_type_cols,
            expression=expression,
            cell_type_hierarchy=hierarchy,
            metadata={
                "source": "CSV",
                "file": str(path),
            },
        )

        return self._atlas

    def get_cell_type_markers(self, cell_type: str) -> List[str]:
        """
        Get marker genes for a cell type.

        Args:
            cell_type: Target cell type

        Returns:
            List of marker gene IDs
        """
        if self._atlas is None:
            raise RuntimeError("No atlas loaded. Call load_h5ad or load_allen_brain first.")

        return self._atlas.get_marker_genes(cell_type)

    def get_expression_by_cell_type(self, gene_id: str) -> Dict[str, float]:
        """
        Get expression for a gene across cell types.

        Args:
            gene_id: Gene identifier

        Returns:
            Dictionary mapping cell type to expression
        """
        if self._atlas is None:
            raise RuntimeError("No atlas loaded. Call load_h5ad or load_allen_brain first.")

        return self._atlas.get_expression_by_cell_type(gene_id)

    def get_neuronal_genes(
        self, fold_change: float = 2.0
    ) -> Set[str]:
        """
        Get genes enriched in neurons vs non-neuronal cells.

        Args:
            fold_change: Minimum fold change

        Returns:
            Set of neuronal gene IDs
        """
        if self._atlas is None:
            raise RuntimeError("No atlas loaded.")

        neuronal_genes: Set[str] = set()

        # Get genes enriched in any neuronal type
        for broad_type in ["excitatory_neuron", "inhibitory_neuron", "neuron"]:
            if broad_type in self._atlas.cell_type_hierarchy:
                for ct in self._atlas.cell_type_hierarchy[broad_type]:
                    neuronal_genes.update(
                        self._atlas.get_genes_enriched_in(ct, fold_change)
                    )

        return neuronal_genes

    def get_excitatory_neuron_genes(
        self, fold_change: float = 2.0
    ) -> Set[str]:
        """
        Get genes enriched in excitatory neurons.

        Args:
            fold_change: Minimum fold change

        Returns:
            Set of excitatory neuron gene IDs
        """
        if self._atlas is None:
            raise RuntimeError("No atlas loaded.")

        genes: Set[str] = set()

        if "excitatory_neuron" in self._atlas.cell_type_hierarchy:
            for ct in self._atlas.cell_type_hierarchy["excitatory_neuron"]:
                genes.update(self._atlas.get_genes_enriched_in(ct, fold_change))

        return genes
