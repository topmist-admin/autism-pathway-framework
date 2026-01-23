"""
Hypothesis ranking for therapeutic candidates.

This module provides the main HypothesisRanker that combines pathway-drug
mapping with evidence scoring to produce ranked therapeutic hypotheses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging

import numpy as np

import sys
from pathlib import Path

# Add module to path for imports
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from pathway_drug_mapping import (
    DrugCandidate,
    DrugTargetDatabase,
    PathwayDrugMapper,
    PathwayDrugMapperConfig,
)
from evidence import (
    EvidenceScore,
    EvidenceScorer,
    EvidenceScorerConfig,
    EvidenceLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class TherapeuticHypothesis:
    """
    A ranked therapeutic hypothesis.

    IMPORTANT: All hypotheses have requires_validation=True. This is
    intentional and cannot be changed. These are research hypotheses
    only, not treatment recommendations.

    Attributes:
        drug_id: Unique drug identifier
        drug_name: Human-readable drug name
        target_pathway: Primary target pathway
        target_genes: Specific genes targeted
        mechanism: Drug mechanism of action
        score: Combined ranking score (0-1)
        evidence: Detailed evidence scoring
        explanation: Human-readable explanation
        confidence: Confidence in the hypothesis (0-1)
        requires_validation: Always True - cannot be changed
        rank: Position in ranked list
        metadata: Additional information
    """

    drug_id: str
    drug_name: str
    target_pathway: str
    target_genes: List[str]
    mechanism: str
    score: float
    evidence: EvidenceScore
    explanation: str
    confidence: float
    requires_validation: bool = field(default=True, init=False)  # Cannot be changed
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate hypothesis and ensure safety flags."""
        # Force requires_validation to True (safety critical)
        object.__setattr__(self, "requires_validation", True)

        if not 0 <= self.score <= 1:
            raise ValueError("score must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "drug_id": self.drug_id,
            "drug_name": self.drug_name,
            "target_pathway": self.target_pathway,
            "target_genes": self.target_genes,
            "mechanism": self.mechanism,
            "score": self.score,
            "evidence": self.evidence.to_dict(),
            "explanation": self.explanation,
            "confidence": self.confidence,
            "requires_validation": True,  # Always True
            "rank": self.rank,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TherapeuticHypothesis":
        """Create from dictionary."""
        evidence = EvidenceScore.from_dict(data.get("evidence", {}))
        hyp = cls(
            drug_id=data["drug_id"],
            drug_name=data.get("drug_name", data["drug_id"]),
            target_pathway=data.get("target_pathway", ""),
            target_genes=data.get("target_genes", []),
            mechanism=data.get("mechanism", ""),
            score=data.get("score", 0),
            evidence=evidence,
            explanation=data.get("explanation", ""),
            confidence=data.get("confidence", 0),
            rank=data.get("rank", 0),
            metadata=data.get("metadata", {}),
        )
        return hyp

    @property
    def is_high_evidence(self) -> bool:
        """Check if hypothesis has high evidence level."""
        return self.evidence.level == EvidenceLevel.HIGH

    @property
    def has_safety_concerns(self) -> bool:
        """Check if hypothesis has safety concerns."""
        return bool(self.evidence.safety_flags)

    def summary(self) -> str:
        """Get brief summary of hypothesis."""
        safety = " (SAFETY FLAGS)" if self.has_safety_concerns else ""
        return (
            f"[{self.rank}] {self.drug_name}: {self.score:.2f} - "
            f"{self.target_pathway} ({self.evidence.level.value}){safety}"
        )


@dataclass
class RankingConfig:
    """Configuration for hypothesis ranking."""

    # Score weights
    weight_evidence: float = 0.4
    weight_pathway_score: float = 0.3
    weight_drug_relevance: float = 0.2
    weight_mechanism_match: float = 0.1

    # Filtering
    min_evidence_score: float = 0.2
    min_pathway_zscore: float = 1.5
    max_hypotheses: int = 50

    # Safety filtering
    exclude_critical_safety: bool = False  # If True, excludes drugs with critical safety flags

    # Diversity
    max_per_pathway: int = 5  # Maximum hypotheses per pathway
    max_per_mechanism: int = 3  # Maximum hypotheses per mechanism type


@dataclass
class RankingResult:
    """
    Result of hypothesis ranking.

    Attributes:
        hypotheses: Ranked list of therapeutic hypotheses
        pathways_analyzed: Pathways that were analyzed
        drugs_considered: Number of drugs considered
        timestamp: When ranking was performed
        config: Configuration used for ranking
        metadata: Additional result information
    """

    hypotheses: List[TherapeuticHypothesis]
    pathways_analyzed: List[str]
    drugs_considered: int
    timestamp: datetime = field(default_factory=datetime.now)
    config: RankingConfig = field(default_factory=RankingConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_hypotheses(self) -> List[TherapeuticHypothesis]:
        """Get top 10 hypotheses."""
        return self.hypotheses[:10]

    @property
    def high_evidence_count(self) -> int:
        """Count of high-evidence hypotheses."""
        return sum(1 for h in self.hypotheses if h.is_high_evidence)

    @property
    def pathways_with_hypotheses(self) -> List[str]:
        """Unique pathways with at least one hypothesis."""
        return list(set(h.target_pathway for h in self.hypotheses))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "pathways_analyzed": self.pathways_analyzed,
            "drugs_considered": self.drugs_considered,
            "timestamp": self.timestamp.isoformat(),
            "high_evidence_count": self.high_evidence_count,
            "pathways_with_hypotheses": self.pathways_with_hypotheses,
        }

    def summary(self) -> str:
        """Get summary of ranking results."""
        lines = [
            f"=== Therapeutic Hypothesis Ranking ===",
            f"Pathways analyzed: {len(self.pathways_analyzed)}",
            f"Drugs considered: {self.drugs_considered}",
            f"Hypotheses generated: {len(self.hypotheses)}",
            f"High evidence: {self.high_evidence_count}",
            f"",
            f"Top Hypotheses:",
        ]
        for h in self.top_hypotheses:
            lines.append(f"  {h.summary()}")
        lines.append("")
        lines.append("NOTE: All hypotheses require clinical validation.")
        return "\n".join(lines)


class HypothesisRanker:
    """
    Ranks therapeutic hypotheses based on pathway disruption and evidence.

    Combines pathway-drug mapping with evidence scoring to produce
    ranked therapeutic hypotheses for investigation.
    """

    def __init__(
        self,
        drug_mapper: Optional[PathwayDrugMapper] = None,
        evidence_scorer: Optional[EvidenceScorer] = None,
        config: Optional[RankingConfig] = None,
    ):
        """
        Initialize hypothesis ranker.

        Args:
            drug_mapper: Pathway-drug mapper (created if not provided)
            evidence_scorer: Evidence scorer (created if not provided)
            config: Ranking configuration
        """
        self.drug_mapper = drug_mapper
        self.evidence_scorer = evidence_scorer or EvidenceScorer()
        self.config = config or RankingConfig()

    def rank(
        self,
        pathway_scores: Dict[str, float],
        pathway_genes: Optional[Dict[str, List[str]]] = None,
        disrupted_genes: Optional[List[str]] = None,
        fired_rules: Optional[List[Any]] = None,
        min_pathway_zscore: Optional[float] = None,
    ) -> RankingResult:
        """
        Rank therapeutic hypotheses for disrupted pathways.

        Args:
            pathway_scores: Pathway ID -> disruption score (z-score)
            pathway_genes: Pathway ID -> list of genes in pathway
            disrupted_genes: List of disrupted genes in individual
            fired_rules: Fired rules from Module 09 (for evidence boost)
            min_pathway_zscore: Minimum z-score to consider (overrides config)

        Returns:
            RankingResult with ranked hypotheses
        """
        min_zscore = min_pathway_zscore or self.config.min_pathway_zscore
        pathway_genes = pathway_genes or {}
        disrupted_genes = disrupted_genes or []

        # Extract rule confidences for evidence boost
        rule_confidences = self._extract_rule_confidences(fired_rules or [])

        # Get candidate drugs for each disrupted pathway
        all_candidates = []
        pathways_analyzed = []
        drugs_seen = set()

        for pathway_id, score in pathway_scores.items():
            if score < min_zscore:
                continue

            pathways_analyzed.append(pathway_id)

            # Get drug candidates for this pathway
            if self.drug_mapper:
                candidates = self.drug_mapper.map(
                    pathway_id=pathway_id,
                    pathway_genes=pathway_genes.get(pathway_id),
                    disrupted_genes=disrupted_genes,
                )
            else:
                candidates = []

            for drug in candidates:
                if drug.drug_id not in drugs_seen:
                    drugs_seen.add(drug.drug_id)
                    all_candidates.append((drug, pathway_id, score))

        # Score and create hypotheses
        hypotheses = []
        for drug, pathway_id, pathway_score in all_candidates:
            # Get rule confidence if available
            rule_conf = rule_confidences.get(drug.drug_id)

            # Score evidence
            evidence = self.evidence_scorer.score(
                drug_id=drug.drug_id,
                drug_name=drug.drug_name,
                target_pathway=pathway_id,
                mechanism=drug.mechanism,
                target_genes=drug.target_genes,
                pathway_genes=pathway_genes.get(pathway_id),
                disrupted_genes=disrupted_genes,
                rule_confidence=rule_conf,
            )

            # Filter by minimum evidence
            if evidence.overall < self.config.min_evidence_score:
                continue

            # Filter critical safety if configured
            if self.config.exclude_critical_safety and evidence.has_critical_safety_flags:
                continue

            # Calculate combined score
            combined_score = self._calculate_combined_score(
                evidence_score=evidence.overall,
                pathway_score=pathway_score,
                drug_relevance=drug.asd_relevance_score,
                mapping_score=drug.metadata.get("mapping_score", 0.5),
            )

            # Calculate confidence
            confidence = self._calculate_confidence(
                evidence=evidence,
                pathway_score=pathway_score,
                rule_confidence=rule_conf,
            )

            # Generate explanation
            explanation = self._generate_explanation(
                drug=drug,
                pathway_id=pathway_id,
                pathway_score=pathway_score,
                evidence=evidence,
            )

            hypothesis = TherapeuticHypothesis(
                drug_id=drug.drug_id,
                drug_name=drug.drug_name,
                target_pathway=pathway_id,
                target_genes=drug.target_genes,
                mechanism=drug.mechanism,
                score=combined_score,
                evidence=evidence,
                explanation=explanation,
                confidence=confidence,
                metadata={
                    "pathway_score": pathway_score,
                    "drug_status": drug.status.value,
                    "rule_confidence": rule_conf,
                },
            )
            hypotheses.append(hypothesis)

        # Apply diversity constraints
        hypotheses = self._apply_diversity(hypotheses)

        # Sort by score (descending)
        hypotheses.sort(key=lambda h: -h.score)

        # Truncate to max
        hypotheses = hypotheses[: self.config.max_hypotheses]

        # Assign ranks
        for i, h in enumerate(hypotheses):
            h.rank = i + 1

        return RankingResult(
            hypotheses=hypotheses,
            pathways_analyzed=pathways_analyzed,
            drugs_considered=len(drugs_seen),
            config=self.config,
            metadata={
                "min_pathway_zscore": min_zscore,
                "disrupted_genes_count": len(disrupted_genes),
                "rules_used": len(rule_confidences),
            },
        )

    def _extract_rule_confidences(
        self,
        fired_rules: List[Any],
    ) -> Dict[str, float]:
        """Extract drug-related confidences from fired rules."""
        confidences = {}

        for rule in fired_rules:
            # Check if it's a therapeutic hypothesis rule
            if not hasattr(rule, "rule"):
                continue

            conclusion = getattr(rule.rule, "conclusion", None)
            if not conclusion:
                continue

            if getattr(conclusion, "type", "") == "therapeutic_hypothesis":
                # Get drug from bindings or attributes
                drug_id = None
                if hasattr(rule, "bindings"):
                    drug_id = rule.bindings.get("D") or rule.bindings.get("drug")
                if not drug_id and hasattr(conclusion, "attributes"):
                    drug_id = conclusion.attributes.get("drug")

                if drug_id and hasattr(rule, "confidence"):
                    confidences[drug_id] = rule.confidence

        return confidences

    def _calculate_combined_score(
        self,
        evidence_score: float,
        pathway_score: float,
        drug_relevance: float,
        mapping_score: float,
    ) -> float:
        """Calculate combined hypothesis score."""
        # Normalize pathway score to 0-1 range (assuming z-scores)
        normalized_pathway = min(pathway_score / 4.0, 1.0)  # z=4 maps to 1.0

        score = (
            self.config.weight_evidence * evidence_score +
            self.config.weight_pathway_score * normalized_pathway +
            self.config.weight_drug_relevance * drug_relevance +
            self.config.weight_mechanism_match * mapping_score
        )

        return min(score, 1.0)

    def _calculate_confidence(
        self,
        evidence: EvidenceScore,
        pathway_score: float,
        rule_confidence: Optional[float],
    ) -> float:
        """Calculate confidence in hypothesis."""
        confidence = evidence.confidence

        # Boost for strong pathway signal
        if pathway_score > 2.5:
            confidence += 0.1

        # Boost for rule support
        if rule_confidence and rule_confidence > 0.7:
            confidence += 0.1

        # Penalty for safety concerns
        if evidence.has_critical_safety_flags:
            confidence -= 0.15

        return max(0.1, min(confidence, 1.0))

    def _apply_diversity(
        self,
        hypotheses: List[TherapeuticHypothesis],
    ) -> List[TherapeuticHypothesis]:
        """Apply diversity constraints to avoid over-representation."""
        # Sort by score first
        hypotheses.sort(key=lambda h: -h.score)

        # Track counts
        pathway_counts: Dict[str, int] = {}
        mechanism_counts: Dict[str, int] = {}

        filtered = []
        for h in hypotheses:
            # Check pathway limit
            pathway_count = pathway_counts.get(h.target_pathway, 0)
            if pathway_count >= self.config.max_per_pathway:
                continue

            # Check mechanism limit
            mech_key = h.mechanism.lower()[:20]  # Normalize
            mech_count = mechanism_counts.get(mech_key, 0)
            if mech_count >= self.config.max_per_mechanism:
                continue

            # Include this hypothesis
            filtered.append(h)
            pathway_counts[h.target_pathway] = pathway_count + 1
            mechanism_counts[mech_key] = mech_count + 1

        return filtered

    def _generate_explanation(
        self,
        drug: DrugCandidate,
        pathway_id: str,
        pathway_score: float,
        evidence: EvidenceScore,
    ) -> str:
        """Generate explanation for hypothesis."""
        parts = []

        parts.append(f"=== Therapeutic Hypothesis: {drug.drug_name} ===\n")

        # Rationale
        parts.append(f"Target pathway: {pathway_id} (disruption z-score: {pathway_score:.2f})")
        parts.append(f"Drug mechanism: {drug.mechanism}")
        parts.append(f"Target genes: {', '.join(drug.target_genes[:5])}")
        if len(drug.target_genes) > 5:
            parts.append(f"  ... and {len(drug.target_genes) - 5} more")

        # Evidence summary
        parts.append(f"\nEvidence level: {evidence.level.value.upper()}")
        parts.append(f"  - Biological plausibility: {evidence.biological_plausibility:.0%}")
        parts.append(f"  - Mechanistic alignment: {evidence.mechanistic_alignment:.0%}")
        parts.append(f"  - Literature support: {evidence.literature_support:.0%}")
        parts.append(f"  - Clinical evidence: {evidence.clinical_evidence:.0%}")

        # Drug status
        parts.append(f"\nDrug status: {drug.status.value}")
        if drug.indications:
            parts.append(f"Known indications: {', '.join(drug.indications[:3])}")

        # Safety considerations
        if evidence.safety_flags:
            parts.append(f"\n*** SAFETY CONSIDERATIONS ***")
            for flag in evidence.safety_flags:
                parts.append(f"  - {flag}")

        # Disclaimer
        parts.append(f"\n" + "=" * 50)
        parts.append("DISCLAIMER: This is a RESEARCH HYPOTHESIS only.")
        parts.append("Requires clinical validation before any therapeutic consideration.")
        parts.append("Consult qualified medical professionals.")

        return "\n".join(parts)


def create_hypothesis_ranker(
    drug_db: Optional[DrugTargetDatabase] = None,
    evidence_scorer: Optional[EvidenceScorer] = None,
    config: Optional[RankingConfig] = None,
) -> HypothesisRanker:
    """
    Factory function to create a configured hypothesis ranker.

    Args:
        drug_db: Drug-target database
        evidence_scorer: Evidence scorer
        config: Ranking configuration

    Returns:
        Configured HypothesisRanker
    """
    # Create drug mapper if database provided
    drug_mapper = None
    if drug_db:
        drug_mapper = PathwayDrugMapper(drug_db)

    return HypothesisRanker(
        drug_mapper=drug_mapper,
        evidence_scorer=evidence_scorer or EvidenceScorer(),
        config=config or RankingConfig(),
    )
