"""
Constraint Loader

Handles loading of gene constraint scores and autism gene annotations:
- gnomAD constraint metrics (pLI, LOEUF, missense Z-scores)
- SFARI Gene database (autism gene scores and evidence)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class GeneConstraints:
    """
    Gene constraint scores from gnomAD.

    Attributes:
        gene_ids: List of all gene identifiers
        pli_scores: Probability of Loss-of-function Intolerance (0-1)
        loeuf_scores: Loss-of-function Observed/Expected Upper Fraction (lower = more constrained)
        mis_z_scores: Missense Z-scores (higher = more constrained for missense)
        syn_z_scores: Synonymous Z-scores (for QC)
        metadata: Additional metadata
    """

    gene_ids: List[str]
    pli_scores: Dict[str, float]
    loeuf_scores: Dict[str, float]
    mis_z_scores: Dict[str, float] = field(default_factory=dict)
    syn_z_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.gene_ids)

    def get_pli(self, gene_id: str) -> Optional[float]:
        """Get pLI score for a gene."""
        return self.pli_scores.get(gene_id)

    def get_loeuf(self, gene_id: str) -> Optional[float]:
        """Get LOEUF score for a gene."""
        return self.loeuf_scores.get(gene_id)

    def get_mis_z(self, gene_id: str) -> Optional[float]:
        """Get missense Z-score for a gene."""
        return self.mis_z_scores.get(gene_id)

    def is_constrained(
        self,
        gene_id: str,
        pli_threshold: float = 0.9,
        loeuf_threshold: Optional[float] = None,
    ) -> bool:
        """
        Check if a gene is highly constrained.

        Args:
            gene_id: Gene identifier
            pli_threshold: Minimum pLI score to be considered constrained
            loeuf_threshold: Maximum LOEUF score (alternative criterion)

        Returns:
            True if gene meets constraint criteria
        """
        pli = self.pli_scores.get(gene_id)

        if pli is not None and pli >= pli_threshold:
            return True

        if loeuf_threshold is not None:
            loeuf = self.loeuf_scores.get(gene_id)
            if loeuf is not None and loeuf <= loeuf_threshold:
                return True

        return False

    def get_constrained_genes(
        self,
        pli_threshold: float = 0.9,
        loeuf_threshold: Optional[float] = None,
    ) -> Set[str]:
        """
        Get all genes meeting constraint criteria.

        Args:
            pli_threshold: Minimum pLI score
            loeuf_threshold: Maximum LOEUF score (optional)

        Returns:
            Set of constrained gene IDs
        """
        constrained = set()

        for gene_id in self.gene_ids:
            if self.is_constrained(gene_id, pli_threshold, loeuf_threshold):
                constrained.add(gene_id)

        return constrained

    def get_constraint_percentile(self, gene_id: str, metric: str = "pli") -> Optional[float]:
        """
        Get the percentile rank for a gene's constraint score.

        Args:
            gene_id: Gene identifier
            metric: "pli", "loeuf", or "mis_z"

        Returns:
            Percentile (0-100) or None if gene not found
        """
        if metric == "pli":
            scores = self.pli_scores
        elif metric == "loeuf":
            scores = self.loeuf_scores
        elif metric == "mis_z":
            scores = self.mis_z_scores
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if gene_id not in scores:
            return None

        gene_score = scores[gene_id]
        all_scores = [s for s in scores.values() if s is not None]

        if metric == "loeuf":
            # Lower is more constrained for LOEUF
            n_less = sum(1 for s in all_scores if s >= gene_score)
        else:
            # Higher is more constrained for pLI and mis_z
            n_less = sum(1 for s in all_scores if s <= gene_score)

        return 100.0 * n_less / len(all_scores)


@dataclass
class SFARIGenes:
    """
    SFARI Autism Gene database.

    Attributes:
        gene_ids: List of all SFARI gene identifiers
        scores: Gene scores (1=High Confidence, 2=Strong Candidate, 3=Suggestive)
        syndromic: Whether gene is associated with syndromic ASD
        evidence: Supporting evidence categories for each gene
        gene_names: Human-readable gene names
        metadata: Additional metadata
    """

    gene_ids: List[str]
    scores: Dict[str, int]
    syndromic: Dict[str, bool] = field(default_factory=dict)
    evidence: Dict[str, List[str]] = field(default_factory=dict)
    gene_names: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)

    # Score definitions
    SCORE_HIGH_CONFIDENCE = 1
    SCORE_STRONG_CANDIDATE = 2
    SCORE_SUGGESTIVE = 3

    def __len__(self) -> int:
        return len(self.gene_ids)

    def __contains__(self, gene_id: str) -> bool:
        return gene_id in self.scores

    def get_score(self, gene_id: str) -> Optional[int]:
        """Get SFARI score for a gene."""
        return self.scores.get(gene_id)

    def is_sfari_gene(self, gene_id: str) -> bool:
        """Check if gene is in SFARI database."""
        return gene_id in self.scores

    def is_high_confidence(self, gene_id: str) -> bool:
        """Check if gene is a high-confidence ASD gene (score 1)."""
        return self.scores.get(gene_id) == self.SCORE_HIGH_CONFIDENCE

    def is_syndromic(self, gene_id: str) -> bool:
        """Check if gene is associated with syndromic ASD."""
        return self.syndromic.get(gene_id, False)

    def get_genes_by_score(self, max_score: int = 2) -> Set[str]:
        """
        Get genes with score <= threshold.

        Args:
            max_score: Maximum score to include (1, 2, or 3)

        Returns:
            Set of gene IDs
        """
        return {
            gene_id for gene_id, score in self.scores.items()
            if score <= max_score
        }

    def get_high_confidence_genes(self) -> Set[str]:
        """Get all high-confidence ASD genes (score 1)."""
        return self.get_genes_by_score(max_score=1)

    def get_strong_candidate_genes(self) -> Set[str]:
        """Get high-confidence and strong candidate genes (score <= 2)."""
        return self.get_genes_by_score(max_score=2)

    def get_all_sfari_genes(self) -> Set[str]:
        """Get all SFARI genes regardless of score."""
        return set(self.gene_ids)

    def get_syndromic_genes(self) -> Set[str]:
        """Get genes associated with syndromic ASD."""
        return {gene_id for gene_id, is_syn in self.syndromic.items() if is_syn}

    def get_evidence(self, gene_id: str) -> List[str]:
        """Get evidence categories for a gene."""
        return self.evidence.get(gene_id, [])


class ConstraintLoader:
    """
    Loader for gene constraint scores and autism gene annotations.

    Supports:
    - gnomAD constraint metrics
    - SFARI Gene database
    """

    def __init__(self):
        """Initialize constraint loader."""
        self._constraints: Optional[GeneConstraints] = None
        self._sfari: Optional[SFARIGenes] = None

    def load_gnomad_constraints(
        self,
        tsv_path: str,
        gene_col: str = "gene",
        pli_col: str = "pLI",
        loeuf_col: str = "oe_lof_upper",
        mis_z_col: str = "mis_z",
    ) -> GeneConstraints:
        """
        Load gnomAD constraint metrics.

        Expected format: TSV with columns for gene symbol and constraint metrics.
        gnomAD v2.1.1 file: gnomad.v2.1.1.lof_metrics.by_gene.txt

        Args:
            tsv_path: Path to gnomAD constraint TSV file
            gene_col: Column name for gene symbol
            pli_col: Column name for pLI score
            loeuf_col: Column name for LOEUF score
            mis_z_col: Column name for missense Z-score

        Returns:
            GeneConstraints object
        """
        path = Path(tsv_path)
        if not path.exists():
            raise FileNotFoundError(f"gnomAD constraint file not found: {tsv_path}")

        gene_ids = []
        pli_scores: Dict[str, float] = {}
        loeuf_scores: Dict[str, float] = {}
        mis_z_scores: Dict[str, float] = {}
        syn_z_scores: Dict[str, float] = {}

        with open(path, "r") as f:
            # Read header
            header = f.readline().strip().split("\t")

            # Find column indices
            col_indices = {col: header.index(col) if col in header else -1 for col in [
                gene_col, pli_col, loeuf_col, mis_z_col, "syn_z"
            ]}

            if col_indices[gene_col] == -1:
                # Try alternative column names
                for alt in ["gene_symbol", "Gene", "SYMBOL"]:
                    if alt in header:
                        col_indices[gene_col] = header.index(alt)
                        break

            if col_indices[gene_col] == -1:
                raise ValueError(f"Gene column not found. Available: {header}")

            for line in f:
                fields = line.strip().split("\t")
                if len(fields) <= col_indices[gene_col]:
                    continue

                gene = fields[col_indices[gene_col]]
                if not gene or gene == "NA":
                    continue

                gene_ids.append(gene)

                # Parse pLI
                if col_indices.get(pli_col, -1) >= 0:
                    val = fields[col_indices[pli_col]]
                    if val and val != "NA":
                        try:
                            pli_scores[gene] = float(val)
                        except ValueError:
                            pass

                # Parse LOEUF
                if col_indices.get(loeuf_col, -1) >= 0:
                    val = fields[col_indices[loeuf_col]]
                    if val and val != "NA":
                        try:
                            loeuf_scores[gene] = float(val)
                        except ValueError:
                            pass

                # Parse missense Z
                if col_indices.get(mis_z_col, -1) >= 0:
                    val = fields[col_indices[mis_z_col]]
                    if val and val != "NA":
                        try:
                            mis_z_scores[gene] = float(val)
                        except ValueError:
                            pass

                # Parse synonymous Z
                if col_indices.get("syn_z", -1) >= 0:
                    val = fields[col_indices["syn_z"]]
                    if val and val != "NA":
                        try:
                            syn_z_scores[gene] = float(val)
                        except ValueError:
                            pass

        self._constraints = GeneConstraints(
            gene_ids=gene_ids,
            pli_scores=pli_scores,
            loeuf_scores=loeuf_scores,
            mis_z_scores=mis_z_scores,
            syn_z_scores=syn_z_scores,
            metadata={
                "source": "gnomAD",
                "file": str(path),
                "n_genes_with_pli": len(pli_scores),
                "n_genes_with_loeuf": len(loeuf_scores),
            },
        )

        logger.info(
            f"Loaded gnomAD constraints: {len(gene_ids)} genes, "
            f"{len(pli_scores)} with pLI, {len(loeuf_scores)} with LOEUF"
        )

        return self._constraints

    def load_sfari_genes(
        self,
        csv_path: str,
        gene_col: str = "gene-symbol",
        score_col: str = "gene-score",
        syndromic_col: str = "syndromic",
    ) -> SFARIGenes:
        """
        Load SFARI Gene database.

        Args:
            csv_path: Path to SFARI genes CSV file
            gene_col: Column name for gene symbol
            score_col: Column name for gene score
            syndromic_col: Column name for syndromic flag

        Returns:
            SFARIGenes object
        """
        import csv

        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"SFARI genes file not found: {csv_path}")

        gene_ids = []
        scores: Dict[str, int] = {}
        syndromic: Dict[str, bool] = {}
        evidence: Dict[str, List[str]] = {}
        gene_names: Dict[str, str] = {}

        with open(path, "r") as f:
            reader = csv.DictReader(f)

            # Normalize column names
            fieldnames = reader.fieldnames or []
            col_map = {}
            for col in fieldnames:
                col_lower = col.lower().replace("-", "_").replace(" ", "_")
                col_map[col_lower] = col

            # Find actual column names
            gene_col_actual = col_map.get("gene_symbol") or col_map.get("gene") or gene_col
            score_col_actual = col_map.get("gene_score") or col_map.get("score") or score_col
            syndromic_col_actual = col_map.get("syndromic") or syndromic_col

            for row in reader:
                gene = row.get(gene_col_actual) or row.get(gene_col)
                if not gene:
                    continue

                gene_ids.append(gene)

                # Parse score
                score_str = row.get(score_col_actual) or row.get(score_col, "")
                if score_str:
                    try:
                        # Handle formats like "1", "1S", "Score 1"
                        score_num = int("".join(c for c in score_str if c.isdigit()) or "0")
                        if 1 <= score_num <= 3:
                            scores[gene] = score_num
                    except ValueError:
                        pass

                # Parse syndromic
                syndromic_str = row.get(syndromic_col_actual) or row.get(syndromic_col, "")
                syndromic[gene] = syndromic_str.lower() in ("1", "yes", "true", "y")

                # Parse evidence if available
                evidence_str = row.get("evidence", "") or row.get("genetic-category", "")
                if evidence_str:
                    evidence[gene] = [e.strip() for e in evidence_str.split(",")]

                # Gene name if available
                name = row.get("gene-name") or row.get("gene_name")
                if name:
                    gene_names[gene] = name

        self._sfari = SFARIGenes(
            gene_ids=gene_ids,
            scores=scores,
            syndromic=syndromic,
            evidence=evidence,
            gene_names=gene_names,
            metadata={
                "source": "SFARI",
                "file": str(path),
                "n_high_confidence": len([s for s in scores.values() if s == 1]),
                "n_strong_candidate": len([s for s in scores.values() if s == 2]),
                "n_suggestive": len([s for s in scores.values() if s == 3]),
            },
        )

        logger.info(
            f"Loaded SFARI genes: {len(gene_ids)} total, "
            f"{self._sfari.metadata['n_high_confidence']} high-confidence"
        )

        return self._sfari

    def get_constrained_genes(self, pli_threshold: float = 0.9) -> List[str]:
        """
        Get list of highly constrained genes.

        Args:
            pli_threshold: Minimum pLI score

        Returns:
            List of gene IDs
        """
        if self._constraints is None:
            raise RuntimeError("No constraint data loaded. Call load_gnomad_constraints first.")

        return list(self._constraints.get_constrained_genes(pli_threshold))

    def get_autism_genes(self, max_score: int = 2) -> Set[str]:
        """
        Get autism-associated genes from SFARI.

        Args:
            max_score: Maximum SFARI score to include

        Returns:
            Set of gene IDs
        """
        if self._sfari is None:
            raise RuntimeError("No SFARI data loaded. Call load_sfari_genes first.")

        return self._sfari.get_genes_by_score(max_score)

    def get_constrained_autism_genes(
        self,
        pli_threshold: float = 0.9,
        sfari_max_score: int = 2,
    ) -> Set[str]:
        """
        Get genes that are both constrained and autism-associated.

        Args:
            pli_threshold: Minimum pLI score
            sfari_max_score: Maximum SFARI score

        Returns:
            Set of gene IDs
        """
        if self._constraints is None or self._sfari is None:
            raise RuntimeError("Load both gnomAD and SFARI data first.")

        constrained = self._constraints.get_constrained_genes(pli_threshold)
        autism = self._sfari.get_genes_by_score(sfari_max_score)

        return constrained & autism

    def create_gene_prior_weights(
        self,
        constraint_weight: float = 0.5,
        sfari_weight: float = 0.5,
    ) -> Dict[str, float]:
        """
        Create prior weights for genes based on constraint and autism association.

        Weights are useful for attention mechanisms in GNNs.

        Args:
            constraint_weight: Weight for constraint component
            sfari_weight: Weight for SFARI component

        Returns:
            Dictionary mapping gene ID to weight (0-1)
        """
        if self._constraints is None:
            raise RuntimeError("No constraint data loaded.")

        weights: Dict[str, float] = {}

        all_genes = set(self._constraints.gene_ids)
        if self._sfari is not None:
            all_genes.update(self._sfari.gene_ids)

        for gene in all_genes:
            weight = 0.0

            # Constraint component (normalized pLI)
            pli = self._constraints.pli_scores.get(gene)
            if pli is not None:
                weight += constraint_weight * pli

            # SFARI component
            if self._sfari is not None and gene in self._sfari.scores:
                score = self._sfari.scores[gene]
                # Convert score to weight (1 -> 1.0, 2 -> 0.66, 3 -> 0.33)
                sfari_component = 1.0 - (score - 1) / 3.0
                weight += sfari_weight * sfari_component

            weights[gene] = min(1.0, weight)  # Cap at 1.0

        return weights
