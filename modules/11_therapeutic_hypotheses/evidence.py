"""
Evidence scoring for therapeutic hypotheses.

This module provides evidence scoring functionality that evaluates
therapeutic hypotheses on biological plausibility, mechanistic alignment,
literature support, and safety considerations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


class EvidenceLevel(Enum):
    """Evidence strength levels."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class SafetyFlag(Enum):
    """Safety flag categories."""

    CNS_EFFECTS = "cns_effects"
    DEVELOPMENTAL_CONCERNS = "developmental_concerns"
    DRUG_INTERACTIONS = "drug_interactions"
    CONTRAINDICATED_PEDIATRIC = "contraindicated_pediatric"
    BLACK_BOX_WARNING = "black_box_warning"
    OFF_LABEL_USE = "off_label_use"
    IMMUNOSUPPRESSION = "immunosuppression"
    HEPATOTOXICITY = "hepatotoxicity"
    CARDIOTOXICITY = "cardiotoxicity"
    TERATOGENIC = "teratogenic"
    WITHDRAWAL_RISK = "withdrawal_risk"


@dataclass
class EvidenceScore:
    """
    Evidence score for a therapeutic hypothesis.

    Evaluates hypotheses across multiple criteria with safety considerations.

    Attributes:
        biological_plausibility: How well the drug-pathway link is biologically supported (0-1)
        mechanistic_alignment: How well drug mechanism aligns with disruption (0-1)
        literature_support: Level of published evidence (0-1)
        clinical_evidence: Evidence from clinical trials (0-1)
        safety_flags: List of safety concerns
        overall: Combined weighted score (0-1)
        confidence: Confidence in the overall score (0-1)
        level: Qualitative evidence level
        explanation: Human-readable explanation of the score
        metadata: Additional scoring information
    """

    biological_plausibility: float = 0.0
    mechanistic_alignment: float = 0.0
    literature_support: float = 0.0
    clinical_evidence: float = 0.0
    safety_flags: List[str] = field(default_factory=list)
    overall: float = 0.0
    confidence: float = 0.0
    level: EvidenceLevel = EvidenceLevel.INSUFFICIENT
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate evidence score."""
        for attr in ["biological_plausibility", "mechanistic_alignment",
                     "literature_support", "clinical_evidence", "overall", "confidence"]:
            value = getattr(self, attr)
            if not 0 <= value <= 1:
                raise ValueError(f"{attr} must be between 0 and 1, got {value}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "biological_plausibility": self.biological_plausibility,
            "mechanistic_alignment": self.mechanistic_alignment,
            "literature_support": self.literature_support,
            "clinical_evidence": self.clinical_evidence,
            "safety_flags": self.safety_flags,
            "overall": self.overall,
            "confidence": self.confidence,
            "level": self.level.value,
            "explanation": self.explanation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceScore":
        """Create from dictionary."""
        level = EvidenceLevel(data.get("level", "insufficient"))
        return cls(
            biological_plausibility=data.get("biological_plausibility", 0),
            mechanistic_alignment=data.get("mechanistic_alignment", 0),
            literature_support=data.get("literature_support", 0),
            clinical_evidence=data.get("clinical_evidence", 0),
            safety_flags=data.get("safety_flags", []),
            overall=data.get("overall", 0),
            confidence=data.get("confidence", 0),
            level=level,
            explanation=data.get("explanation", ""),
            metadata=data.get("metadata", {}),
        )

    @property
    def has_critical_safety_flags(self) -> bool:
        """Check if there are critical safety concerns."""
        critical_flags = {
            SafetyFlag.BLACK_BOX_WARNING.value,
            SafetyFlag.CONTRAINDICATED_PEDIATRIC.value,
            SafetyFlag.TERATOGENIC.value,
        }
        return bool(set(self.safety_flags) & critical_flags)

    @property
    def safety_summary(self) -> str:
        """Get summary of safety concerns."""
        if not self.safety_flags:
            return "No specific safety flags identified"
        if self.has_critical_safety_flags:
            return f"CRITICAL SAFETY CONCERNS: {', '.join(self.safety_flags)}"
        return f"Safety considerations: {', '.join(self.safety_flags)}"


@dataclass
class EvidenceScorerConfig:
    """Configuration for evidence scorer."""

    # Component weights (must sum to 1)
    weight_biological: float = 0.3
    weight_mechanistic: float = 0.25
    weight_literature: float = 0.25
    weight_clinical: float = 0.2

    # Thresholds for evidence levels
    threshold_high: float = 0.7
    threshold_moderate: float = 0.5
    threshold_low: float = 0.3

    # Safety penalty
    safety_penalty_per_flag: float = 0.05
    max_safety_penalty: float = 0.3

    # Confidence adjustments
    min_confidence: float = 0.3
    confidence_boost_clinical: float = 0.1
    confidence_penalty_safety: float = 0.1

    def __post_init__(self):
        """Validate configuration."""
        total_weight = (
            self.weight_biological +
            self.weight_mechanistic +
            self.weight_literature +
            self.weight_clinical
        )
        if not math.isclose(total_weight, 1.0, rel_tol=0.01):
            logger.warning(f"Evidence weights sum to {total_weight}, not 1.0")


class EvidenceScorer:
    """
    Scores therapeutic hypotheses on evidence quality.

    Evaluates biological plausibility, mechanistic alignment, literature
    support, and safety considerations.
    """

    def __init__(
        self,
        config: Optional[EvidenceScorerConfig] = None,
        literature_db: Optional[Dict[str, float]] = None,
        clinical_trials_db: Optional[Dict[str, float]] = None,
        safety_db: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize evidence scorer.

        Args:
            config: Scorer configuration
            literature_db: Drug ID -> literature score (mock or real database)
            clinical_trials_db: Drug ID -> clinical evidence score
            safety_db: Drug ID -> list of safety flags
        """
        self.config = config or EvidenceScorerConfig()
        self.literature_db = literature_db or {}
        self.clinical_trials_db = clinical_trials_db or {}
        self.safety_db = safety_db or {}

    def score(
        self,
        drug_id: str,
        drug_name: str,
        target_pathway: str,
        mechanism: str,
        target_genes: List[str],
        pathway_genes: Optional[List[str]] = None,
        disrupted_genes: Optional[List[str]] = None,
        rule_confidence: Optional[float] = None,
    ) -> EvidenceScore:
        """
        Score a therapeutic hypothesis.

        Args:
            drug_id: Drug identifier
            drug_name: Drug name
            target_pathway: Target pathway
            mechanism: Drug mechanism
            target_genes: Genes targeted by drug
            pathway_genes: Genes in the pathway
            disrupted_genes: Genes disrupted in individual
            rule_confidence: Confidence from rule engine (if applicable)

        Returns:
            EvidenceScore with all components
        """
        # Calculate component scores
        bio_plausibility = self._score_biological_plausibility(
            target_genes=target_genes,
            pathway_genes=pathway_genes or [],
            disrupted_genes=disrupted_genes or [],
        )

        mech_alignment = self._score_mechanistic_alignment(
            mechanism=mechanism,
            target_pathway=target_pathway,
        )

        lit_support = self._score_literature_support(drug_id)
        clinical_evidence = self._score_clinical_evidence(drug_id)

        # Get safety flags
        safety_flags = self._get_safety_flags(drug_id, mechanism)

        # Calculate overall score
        overall = self._calculate_overall_score(
            bio_plausibility=bio_plausibility,
            mech_alignment=mech_alignment,
            lit_support=lit_support,
            clinical_evidence=clinical_evidence,
            safety_flags=safety_flags,
        )

        # Adjust for rule confidence if available
        if rule_confidence is not None:
            overall = 0.7 * overall + 0.3 * rule_confidence

        # Calculate confidence
        confidence = self._calculate_confidence(
            bio_plausibility=bio_plausibility,
            mech_alignment=mech_alignment,
            lit_support=lit_support,
            clinical_evidence=clinical_evidence,
            safety_flags=safety_flags,
        )

        # Determine evidence level
        level = self._determine_level(overall)

        # Generate explanation
        explanation = self._generate_explanation(
            drug_name=drug_name,
            target_pathway=target_pathway,
            bio_plausibility=bio_plausibility,
            mech_alignment=mech_alignment,
            lit_support=lit_support,
            clinical_evidence=clinical_evidence,
            safety_flags=safety_flags,
            level=level,
        )

        return EvidenceScore(
            biological_plausibility=bio_plausibility,
            mechanistic_alignment=mech_alignment,
            literature_support=lit_support,
            clinical_evidence=clinical_evidence,
            safety_flags=safety_flags,
            overall=overall,
            confidence=confidence,
            level=level,
            explanation=explanation,
            metadata={
                "drug_id": drug_id,
                "target_pathway": target_pathway,
                "rule_confidence": rule_confidence,
            },
        )

    def _score_biological_plausibility(
        self,
        target_genes: List[str],
        pathway_genes: List[str],
        disrupted_genes: List[str],
    ) -> float:
        """Score biological plausibility of drug-pathway link."""
        if not target_genes:
            return 0.0

        score = 0.3  # Base score for having targets

        # Overlap with pathway genes
        if pathway_genes:
            overlap = set(target_genes) & set(pathway_genes)
            overlap_ratio = len(overlap) / len(pathway_genes)
            score += 0.3 * min(overlap_ratio * 5, 1.0)  # Scale up small overlaps

        # Targeting disrupted genes (strong signal)
        if disrupted_genes:
            disrupted_overlap = set(target_genes) & set(disrupted_genes)
            if disrupted_overlap:
                score += 0.4 * (len(disrupted_overlap) / len(disrupted_genes))

        return min(score, 1.0)

    def _score_mechanistic_alignment(
        self,
        mechanism: str,
        target_pathway: str,
    ) -> float:
        """Score how well drug mechanism aligns with pathway disruption."""
        if not mechanism:
            return 0.3  # Unknown mechanism gets base score

        mechanism_lower = mechanism.lower()
        pathway_lower = target_pathway.lower()

        score = 0.4  # Base score

        # Check for mechanistic keywords
        positive_keywords = {
            "synaptic": ["agonist", "modulator", "enhancer"],
            "chromatin": ["inhibitor", "modulator"],
            "mtor": ["inhibitor"],
            "gaba": ["agonist", "modulator"],
            "glutamate": ["antagonist", "modulator"],
            "serotonin": ["inhibitor", "modulator"],
        }

        for pathway_key, good_mechanisms in positive_keywords.items():
            if pathway_key in pathway_lower:
                for good_mech in good_mechanisms:
                    if good_mech in mechanism_lower:
                        score += 0.3
                        break

        # Bonus for specific mechanism descriptions
        if "receptor" in mechanism_lower or "transporter" in mechanism_lower:
            score += 0.1

        return min(score, 1.0)

    def _score_literature_support(self, drug_id: str) -> float:
        """Score based on literature evidence."""
        # Check database
        if drug_id in self.literature_db:
            return self.literature_db[drug_id]

        # Default: assume some basic literature exists
        return 0.3

    def _score_clinical_evidence(self, drug_id: str) -> float:
        """Score based on clinical trial evidence."""
        # Check database
        if drug_id in self.clinical_trials_db:
            return self.clinical_trials_db[drug_id]

        # Default: no clinical evidence
        return 0.0

    def _get_safety_flags(
        self,
        drug_id: str,
        mechanism: str,
    ) -> List[str]:
        """Get safety flags for a drug."""
        flags = []

        # Check safety database
        if drug_id in self.safety_db:
            flags.extend(self.safety_db[drug_id])

        # Infer flags from mechanism
        mechanism_lower = mechanism.lower()

        if "immunosuppres" in mechanism_lower:
            flags.append(SafetyFlag.IMMUNOSUPPRESSION.value)

        if "mtor" in mechanism_lower or "hdac" in mechanism_lower:
            flags.append(SafetyFlag.DEVELOPMENTAL_CONCERNS.value)

        if any(term in mechanism_lower for term in ["ssri", "serotonin", "dopamine"]):
            flags.append(SafetyFlag.CNS_EFFECTS.value)
            flags.append(SafetyFlag.WITHDRAWAL_RISK.value)

        return list(set(flags))  # Remove duplicates

    def _calculate_overall_score(
        self,
        bio_plausibility: float,
        mech_alignment: float,
        lit_support: float,
        clinical_evidence: float,
        safety_flags: List[str],
    ) -> float:
        """Calculate weighted overall score."""
        # Weighted sum of components
        score = (
            self.config.weight_biological * bio_plausibility +
            self.config.weight_mechanistic * mech_alignment +
            self.config.weight_literature * lit_support +
            self.config.weight_clinical * clinical_evidence
        )

        # Apply safety penalty
        safety_penalty = min(
            len(safety_flags) * self.config.safety_penalty_per_flag,
            self.config.max_safety_penalty,
        )
        score = max(0, score - safety_penalty)

        return min(score, 1.0)

    def _calculate_confidence(
        self,
        bio_plausibility: float,
        mech_alignment: float,
        lit_support: float,
        clinical_evidence: float,
        safety_flags: List[str],
    ) -> float:
        """Calculate confidence in the evidence score."""
        # Start with base confidence from data completeness
        confidence = self.config.min_confidence

        # Boost for strong individual components
        if bio_plausibility > 0.6:
            confidence += 0.15
        if mech_alignment > 0.6:
            confidence += 0.1
        if lit_support > 0.5:
            confidence += 0.15

        # Major boost for clinical evidence
        if clinical_evidence > 0.3:
            confidence += self.config.confidence_boost_clinical

        # Agreement between components increases confidence
        scores = [bio_plausibility, mech_alignment, lit_support]
        if np.std(scores) < 0.2:  # Components agree
            confidence += 0.1

        # Penalty for safety concerns
        if safety_flags:
            confidence -= self.config.confidence_penalty_safety

        return max(self.config.min_confidence, min(confidence, 1.0))

    def _determine_level(self, overall: float) -> EvidenceLevel:
        """Determine qualitative evidence level."""
        if overall >= self.config.threshold_high:
            return EvidenceLevel.HIGH
        elif overall >= self.config.threshold_moderate:
            return EvidenceLevel.MODERATE
        elif overall >= self.config.threshold_low:
            return EvidenceLevel.LOW
        else:
            return EvidenceLevel.INSUFFICIENT

    def _generate_explanation(
        self,
        drug_name: str,
        target_pathway: str,
        bio_plausibility: float,
        mech_alignment: float,
        lit_support: float,
        clinical_evidence: float,
        safety_flags: List[str],
        level: EvidenceLevel,
    ) -> str:
        """Generate human-readable explanation."""
        parts = []

        parts.append(f"{drug_name} targeting {target_pathway}:")

        # Evidence level summary
        level_desc = {
            EvidenceLevel.HIGH: "Strong evidence supports this hypothesis.",
            EvidenceLevel.MODERATE: "Moderate evidence supports this hypothesis.",
            EvidenceLevel.LOW: "Limited evidence supports this hypothesis.",
            EvidenceLevel.INSUFFICIENT: "Insufficient evidence for this hypothesis.",
        }
        parts.append(level_desc[level])

        # Component breakdown
        parts.append("\nEvidence breakdown:")
        parts.append(f"  - Biological plausibility: {bio_plausibility:.0%}")
        parts.append(f"  - Mechanistic alignment: {mech_alignment:.0%}")
        parts.append(f"  - Literature support: {lit_support:.0%}")
        parts.append(f"  - Clinical evidence: {clinical_evidence:.0%}")

        # Safety considerations
        if safety_flags:
            parts.append(f"\nSafety considerations: {', '.join(safety_flags)}")
        else:
            parts.append("\nNo specific safety flags identified.")

        # Disclaimer
        parts.append("\nNOTE: This is a HYPOTHESIS requiring clinical validation.")

        return "\n".join(parts)

    def add_literature_evidence(
        self,
        drug_id: str,
        score: float,
    ) -> None:
        """Add literature evidence score for a drug."""
        self.literature_db[drug_id] = max(0, min(1, score))

    def add_clinical_evidence(
        self,
        drug_id: str,
        score: float,
    ) -> None:
        """Add clinical trial evidence score for a drug."""
        self.clinical_trials_db[drug_id] = max(0, min(1, score))

    def add_safety_flags(
        self,
        drug_id: str,
        flags: List[str],
    ) -> None:
        """Add safety flags for a drug."""
        if drug_id not in self.safety_db:
            self.safety_db[drug_id] = []
        self.safety_db[drug_id].extend(flags)
        self.safety_db[drug_id] = list(set(self.safety_db[drug_id]))


def create_sample_evidence_databases() -> Tuple[Dict[str, float], Dict[str, float], Dict[str, List[str]]]:
    """
    Create sample evidence databases for testing/demonstration.

    Returns:
        Tuple of (literature_db, clinical_trials_db, safety_db)
    """
    # Literature evidence scores (based on publication count/quality)
    literature_db = {
        "DB00334": 0.7,  # Memantine - well studied
        "DB01104": 0.6,  # Arbaclofen - some studies
        "DB06603": 0.3,  # Panobinostat - limited ASD literature
        "DB00877": 0.8,  # Sirolimus - well studied for TSC
        "DB01590": 0.8,  # Everolimus - FDA approved for TSC
        "DB00107": 0.5,  # Oxytocin - mixed results
        "DB00196": 0.4,  # Fluoxetine - some ASD studies
        "EXP001": 0.5,  # IGF-1 - investigational
    }

    # Clinical trial evidence
    clinical_trials_db = {
        "DB00334": 0.4,  # Some trials
        "DB01104": 0.3,  # Phase 2 trials
        "DB00877": 0.6,  # TSC trials
        "DB01590": 0.7,  # FDA approved
        "DB00107": 0.3,  # Mixed results
        "EXP001": 0.2,  # Early phase
    }

    # Safety flags
    safety_db = {
        "DB00334": [SafetyFlag.CNS_EFFECTS.value],
        "DB06603": [
            SafetyFlag.DEVELOPMENTAL_CONCERNS.value,
            SafetyFlag.BLACK_BOX_WARNING.value,
        ],
        "DB00877": [
            SafetyFlag.IMMUNOSUPPRESSION.value,
            SafetyFlag.DRUG_INTERACTIONS.value,
        ],
        "DB01590": [
            SafetyFlag.IMMUNOSUPPRESSION.value,
            SafetyFlag.DRUG_INTERACTIONS.value,
        ],
        "DB00196": [
            SafetyFlag.CNS_EFFECTS.value,
            SafetyFlag.WITHDRAWAL_RISK.value,
            SafetyFlag.BLACK_BOX_WARNING.value,  # Pediatric
        ],
    }

    return literature_db, clinical_trials_db, safety_db
