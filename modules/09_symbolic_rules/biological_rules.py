"""
Curated biological rules (R1-R6) for autism genetics inference.

These rules encode domain knowledge about autism-relevant genetic mechanisms,
backed by published research and biological databases.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import sys
from pathlib import Path

# Add module to path for imports
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))


class ConclusionType(Enum):
    """Types of conclusions that rules can produce."""

    PATHWAY_DISRUPTION = "pathway_disruption"
    PATHWAY_CONVERGENCE = "pathway_convergence"
    SUBTYPE_INDICATOR = "subtype_indicator"
    EFFECT_MODIFIER = "effect_modifier"
    THERAPEUTIC_HYPOTHESIS = "therapeutic_hypothesis"


@dataclass
class Conclusion:
    """
    The conclusion of a fired rule.

    Attributes:
        type: Category of conclusion
        attributes: Detailed information about the conclusion
        confidence_modifier: Multiplier for rule's base confidence
    """
    type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence_modifier: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "attributes": self.attributes,
            "confidence_modifier": self.confidence_modifier,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conclusion":
        """Create from dictionary."""
        return cls(
            type=data["type"],
            attributes=data.get("attributes", {}),
            confidence_modifier=data.get("confidence_modifier", 1.0),
        )


@dataclass
class Rule:
    """
    A biological inference rule.

    Rules consist of conditions that must be satisfied and a conclusion
    that is drawn when conditions are met.

    Attributes:
        id: Unique rule identifier
        name: Human-readable rule name
        description: Detailed description of the rule
        conditions: List of conditions to evaluate
        conclusion: Conclusion when rule fires
        base_confidence: Base confidence score (0-1)
        evidence_sources: Literature/database sources backing this rule
        logic: How to combine conditions ("AND" or "OR")
        priority: Rule priority for conflict resolution (higher = more important)
    """
    id: str
    name: str
    description: str
    conditions: List["Condition"]  # Forward reference
    conclusion: Conclusion
    base_confidence: float = 0.8
    evidence_sources: List[str] = field(default_factory=list)
    logic: str = "AND"  # "AND" or "OR"
    priority: int = 0

    def __post_init__(self):
        """Validate rule after initialization."""
        if not self.id:
            raise ValueError("Rule ID cannot be empty")
        if not self.conditions:
            raise ValueError("Rule must have at least one condition")
        if not 0 <= self.base_confidence <= 1:
            raise ValueError("Base confidence must be between 0 and 1")
        if self.logic not in ("AND", "OR"):
            raise ValueError("Logic must be 'AND' or 'OR'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "conditions": [c.to_dict() for c in self.conditions],
            "conclusion": self.conclusion.to_dict(),
            "base_confidence": self.base_confidence,
            "evidence_sources": self.evidence_sources,
            "logic": self.logic,
            "priority": self.priority,
        }


# Import Condition here to avoid circular import
from conditions import Condition


class BiologicalRules:
    """
    Factory class for curated autism-specific biological rules.

    Rules R1-R6 encode key patterns in autism genetics:
    - R1: Constrained LoF in developing cortex
    - R2: Pathway convergence (multiple hits)
    - R3: CHD8 chromatin cascade
    - R4: Synaptic excitatory disruption
    - R5: Paralog compensation
    - R6: Therapeutic pathway targeting
    """

    @staticmethod
    def R1_constrained_lof_developing_cortex() -> Rule:
        """
        R1: LoF in constrained gene expressed in developing cortex
            → High-confidence pathway disruption

        Biological rationale:
        - Loss-of-function variants in constrained genes are more likely pathogenic
        - Prenatal cortical expression indicates developmental brain relevance
        - This combination is strongly associated with neurodevelopmental disorders

        Evidence:
        - Samocha et al. 2014 (constraint scores)
        - Willsey et al. 2013 (prenatal cortical expression)
        - Sanders et al. 2015 (autism de novo variants)
        """
        return Rule(
            id="R1",
            name="Constrained LoF in Developing Cortex",
            description=(
                "Loss-of-function variant in evolutionarily constrained gene "
                "with prenatal cortical expression indicates high-confidence "
                "pathway disruption via haploinsufficiency."
            ),
            conditions=[
                Condition(
                    predicate="has_lof_variant",
                    arguments={},
                ),
                Condition(
                    predicate="is_constrained",
                    arguments={"pli_threshold": 0.9},
                ),
                Condition(
                    predicate="prenatally_expressed",
                    arguments={"threshold": 1.0},
                ),
            ],
            conclusion=Conclusion(
                type=ConclusionType.PATHWAY_DISRUPTION.value,
                attributes={
                    "confidence": "high",
                    "mechanism": "haploinsufficiency",
                    "developmental_timing": "prenatal",
                    "tissue": "cortex",
                },
                confidence_modifier=1.0,
            ),
            base_confidence=0.90,
            evidence_sources=[
                "gnomAD constraint metrics",
                "BrainSpan developmental expression",
                "Samocha et al. 2014",
                "Willsey et al. 2013",
            ],
            logic="AND",
            priority=10,  # High priority rule
        )

    @staticmethod
    def R2_pathway_convergence() -> Rule:
        """
        R2: Multiple hits in same pathway (≥2 genes)
            → Pathway-level convergence signal

        Biological rationale:
        - Independent hits in the same pathway suggest pathway-level dysfunction
        - This is a key pattern for identifying causal pathways in ASD
        - Strengthens evidence beyond single-gene findings

        Evidence:
        - Voineagu et al. 2011 (transcriptomic convergence)
        - Parikshak et al. 2013 (network convergence)
        - De Rubeis et al. 2014 (pathway enrichment)
        """
        return Rule(
            id="R2",
            name="Pathway Convergence",
            description=(
                "Multiple independent damaging variants converging on the "
                "same biological pathway indicates strong pathway-level "
                "dysfunction signal."
            ),
            conditions=[
                Condition(
                    predicate="has_multiple_hits",
                    arguments={"min_genes": 2, "pathway": "P"},
                ),
                Condition(
                    predicate="hits_are_independent",
                    arguments={"pathway": "P"},
                ),
            ],
            conclusion=Conclusion(
                type=ConclusionType.PATHWAY_CONVERGENCE.value,
                attributes={
                    "strength": "strong",
                    "evidence_type": "multi-gene",
                },
                confidence_modifier=1.0,
            ),
            base_confidence=0.85,
            evidence_sources=[
                "Pathway membership databases",
                "Voineagu et al. 2011",
                "Parikshak et al. 2013",
            ],
            logic="AND",
            priority=8,
        )

    @staticmethod
    def R3_chd8_cascade() -> Rule:
        """
        R3: Disruption in CHD8 or its regulatory targets
            → Chromatin regulation cascade

        Biological rationale:
        - CHD8 is a master regulator of chromatin remodeling
        - CHD8 mutations are among the highest-confidence ASD risk factors
        - CHD8 targets show coordinated dysregulation in ASD

        Evidence:
        - Cotney et al. 2015 (CHD8 ChIP-seq targets)
        - Sugathan et al. 2014 (CHD8 knockdown transcriptomics)
        - Bernier et al. 2014 (CHD8 clinical phenotype)
        """
        return Rule(
            id="R3",
            name="CHD8 Chromatin Cascade",
            description=(
                "Disruption in CHD8 or its regulatory target genes indicates "
                "chromatin remodeling cascade dysfunction, a well-established "
                "autism mechanism."
            ),
            conditions=[
                # Either CHD8 itself OR a CHD8 target is disrupted
                Condition(
                    predicate="has_damaging_variant",
                    arguments={"gene": "CHD8"},
                ),
            ],
            conclusion=Conclusion(
                type=ConclusionType.PATHWAY_DISRUPTION.value,
                attributes={
                    "pathway": "chromatin_regulation",
                    "mechanism": "CHD8_cascade",
                    "subtype_indicator": "chromatin_remodeling",
                    "master_regulator": "CHD8",
                },
                confidence_modifier=1.0,
            ),
            base_confidence=0.92,
            evidence_sources=[
                "Cotney et al. 2015",
                "Sugathan et al. 2014",
                "Bernier et al. 2014",
            ],
            logic="AND",
            priority=9,
        )

    @staticmethod
    def R3b_chd8_target_cascade() -> Rule:
        """
        R3b: Disruption in CHD8 regulatory target
             → Chromatin regulation cascade (indirect)

        Separate rule for CHD8 targets to allow different confidence levels.
        """
        return Rule(
            id="R3b",
            name="CHD8 Target Disruption",
            description=(
                "Disruption in a gene regulated by CHD8 indicates potential "
                "chromatin cascade dysfunction via indirect mechanism."
            ),
            conditions=[
                Condition(
                    predicate="has_damaging_variant",
                    arguments={},  # Gene will be bound during evaluation
                ),
                Condition(
                    predicate="is_chd8_target",
                    arguments={},
                ),
            ],
            conclusion=Conclusion(
                type=ConclusionType.PATHWAY_DISRUPTION.value,
                attributes={
                    "pathway": "chromatin_regulation",
                    "mechanism": "CHD8_target_disruption",
                    "subtype_indicator": "chromatin_remodeling",
                    "indirect": True,
                },
                confidence_modifier=0.8,  # Lower confidence for indirect
            ),
            base_confidence=0.75,
            evidence_sources=[
                "Cotney et al. 2015",
                "Sugathan et al. 2014",
            ],
            logic="AND",
            priority=7,
        )

    @staticmethod
    def R4_synaptic_excitatory() -> Rule:
        """
        R4: Synaptic gene hit + expression in excitatory neurons
            → Synaptic subtype indicator

        Biological rationale:
        - Synaptic genes are enriched among ASD risk genes
        - Excitatory/inhibitory imbalance is a key ASD hypothesis
        - Cell-type specific expression adds mechanistic precision

        Evidence:
        - Koopmans et al. 2019 (SynGO database)
        - Satterstrom et al. 2020 (cell-type expression)
        - Rubenstein & Merzenich 2003 (E/I imbalance)
        """
        return Rule(
            id="R4",
            name="Synaptic Excitatory Disruption",
            description=(
                "Disruption of synaptic gene with preferential expression in "
                "excitatory neurons indicates synaptic dysfunction subtype "
                "with potential E/I imbalance."
            ),
            conditions=[
                Condition(
                    predicate="has_damaging_variant",
                    arguments={},
                ),
                Condition(
                    predicate="is_synaptic_gene",
                    arguments={"ontology": "SynGO"},
                ),
                Condition(
                    predicate="cell_type_expression",
                    arguments={
                        "cell_type": "excitatory_neuron",
                        "enriched": True,
                    },
                ),
            ],
            conclusion=Conclusion(
                type=ConclusionType.SUBTYPE_INDICATOR.value,
                attributes={
                    "subtype": "synaptic_dysfunction",
                    "cell_type": "excitatory",
                    "mechanism": "synaptic_transmission",
                    "implication": "E_I_imbalance",
                },
                confidence_modifier=1.0,
            ),
            base_confidence=0.82,
            evidence_sources=[
                "SynGO (Koopmans et al. 2019)",
                "Single-cell expression atlas",
                "Satterstrom et al. 2020",
            ],
            logic="AND",
            priority=7,
        )

    @staticmethod
    def R4b_synaptic_inhibitory() -> Rule:
        """
        R4b: Synaptic gene hit + expression in inhibitory neurons
             → Synaptic subtype indicator (inhibitory)

        Similar to R4 but for inhibitory neuron expression.
        """
        return Rule(
            id="R4b",
            name="Synaptic Inhibitory Disruption",
            description=(
                "Disruption of synaptic gene with preferential expression in "
                "inhibitory neurons indicates synaptic dysfunction with "
                "potential GABAergic involvement."
            ),
            conditions=[
                Condition(
                    predicate="has_damaging_variant",
                    arguments={},
                ),
                Condition(
                    predicate="is_synaptic_gene",
                    arguments={"ontology": "SynGO"},
                ),
                Condition(
                    predicate="cell_type_expression",
                    arguments={
                        "cell_type": "inhibitory_neuron",
                        "enriched": True,
                    },
                ),
            ],
            conclusion=Conclusion(
                type=ConclusionType.SUBTYPE_INDICATOR.value,
                attributes={
                    "subtype": "synaptic_dysfunction",
                    "cell_type": "inhibitory",
                    "mechanism": "GABAergic_signaling",
                    "implication": "E_I_imbalance",
                },
                confidence_modifier=1.0,
            ),
            base_confidence=0.80,
            evidence_sources=[
                "SynGO (Koopmans et al. 2019)",
                "Single-cell expression atlas",
            ],
            logic="AND",
            priority=6,
        )

    @staticmethod
    def R5_compensatory_paralog() -> Rule:
        """
        R5: Paralog intact + expressed
            → Potential compensation (reduced penetrance)

        Biological rationale:
        - Gene duplicates can provide functional redundancy
        - Expressed paralogs may compensate for loss-of-function
        - This is a modifier of penetrance, not a direct cause

        Evidence:
        - Diss & Bhatt 2020 (paralog compensation)
        - Kuzmin et al. 2020 (genetic interactions)
        """
        return Rule(
            id="R5",
            name="Paralog Compensation",
            description=(
                "Intact and expressed paralog may compensate for disrupted "
                "gene, potentially reducing penetrance of the variant. "
                "This is a modifier effect."
            ),
            conditions=[
                Condition(
                    predicate="has_lof_variant",
                    arguments={},
                ),
                Condition(
                    predicate="has_paralog",
                    arguments={},
                ),
                # Paralog is NOT disrupted
                Condition(
                    predicate="has_damaging_variant",
                    arguments={},
                    negated=True,
                ),
                Condition(
                    predicate="expressed_in",
                    arguments={"tissue": "brain", "level": "high"},
                ),
            ],
            conclusion=Conclusion(
                type=ConclusionType.EFFECT_MODIFIER.value,
                attributes={
                    "modifier_type": "compensation",
                    "confidence_reduction": 0.3,
                    "mechanism": "paralog_redundancy",
                    "clinical_implication": "variable_penetrance",
                },
                confidence_modifier=0.7,  # Reduces confidence of pathogenicity
            ),
            base_confidence=0.70,
            evidence_sources=[
                "Paralog databases (Ensembl)",
                "Expression data",
                "Diss & Bhatt 2020",
            ],
            logic="AND",
            priority=5,  # Lower priority - modifier rule
        )

    @staticmethod
    def R6_drug_pathway_target() -> Rule:
        """
        R6: Drug targets disrupted pathway
            → Therapeutic hypothesis candidate

        Biological rationale:
        - If a pathway is disrupted, drugs targeting that pathway may help
        - This generates hypotheses for therapeutic investigation
        - Requires validation - not a treatment recommendation

        Evidence:
        - DrugBank database
        - Pathway-drug mappings
        - Mechanism annotations
        """
        return Rule(
            id="R6",
            name="Therapeutic Pathway Target",
            description=(
                "Drug targeting a gene within the disrupted pathway represents "
                "a therapeutic hypothesis candidate. REQUIRES VALIDATION - "
                "this is a research hypothesis, not a treatment recommendation."
            ),
            conditions=[
                Condition(
                    predicate="pathway_disrupted",
                    arguments={"score_threshold": 2.0},
                ),
                Condition(
                    predicate="drug_targets",
                    arguments={},
                ),
                Condition(
                    predicate="gene_in_pathway",
                    arguments={},
                ),
                Condition(
                    predicate="mechanism_alignment",
                    arguments={},
                ),
            ],
            conclusion=Conclusion(
                type=ConclusionType.THERAPEUTIC_HYPOTHESIS.value,
                attributes={
                    "requires_validation": True,
                    "hypothesis_type": "pathway_targeting",
                    "evidence_level": "computational",
                    "disclaimer": (
                        "This is a research hypothesis for investigation, "
                        "not a treatment recommendation."
                    ),
                },
                confidence_modifier=1.0,
            ),
            base_confidence=0.60,  # Lower base confidence - hypothesis only
            evidence_sources=[
                "DrugBank",
                "Pathway databases",
                "Mechanism annotations",
            ],
            logic="AND",
            priority=4,  # Lower priority - hypothesis generation
        )

    @staticmethod
    def R7_sfari_high_confidence() -> Rule:
        """
        R7: High-confidence SFARI gene with damaging variant
            → Strong autism association

        Additional rule for SFARI gene prioritization.
        """
        return Rule(
            id="R7",
            name="High-Confidence SFARI Gene Hit",
            description=(
                "Damaging variant in high-confidence SFARI gene (score 1) "
                "indicates strong autism genetic association."
            ),
            conditions=[
                Condition(
                    predicate="has_damaging_variant",
                    arguments={},
                ),
                Condition(
                    predicate="is_high_confidence_sfari",
                    arguments={},
                ),
            ],
            conclusion=Conclusion(
                type=ConclusionType.PATHWAY_DISRUPTION.value,
                attributes={
                    "confidence": "high",
                    "evidence_type": "SFARI_curated",
                    "sfari_score": 1,
                },
                confidence_modifier=1.0,
            ),
            base_confidence=0.88,
            evidence_sources=[
                "SFARI Gene database",
                "Curated autism genetics literature",
            ],
            logic="AND",
            priority=9,
        )

    @staticmethod
    def get_all_rules() -> List[Rule]:
        """
        Return all curated biological rules.

        Returns:
            List of Rule objects for R1-R7 (plus variants)
        """
        return [
            BiologicalRules.R1_constrained_lof_developing_cortex(),
            BiologicalRules.R2_pathway_convergence(),
            BiologicalRules.R3_chd8_cascade(),
            BiologicalRules.R3b_chd8_target_cascade(),
            BiologicalRules.R4_synaptic_excitatory(),
            BiologicalRules.R4b_synaptic_inhibitory(),
            BiologicalRules.R5_compensatory_paralog(),
            BiologicalRules.R6_drug_pathway_target(),
            BiologicalRules.R7_sfari_high_confidence(),
        ]

    @staticmethod
    def get_core_rules() -> List[Rule]:
        """
        Return core rules (R1-R6) without variants.

        Returns:
            List of core Rule objects
        """
        return [
            BiologicalRules.R1_constrained_lof_developing_cortex(),
            BiologicalRules.R2_pathway_convergence(),
            BiologicalRules.R3_chd8_cascade(),
            BiologicalRules.R4_synaptic_excitatory(),
            BiologicalRules.R5_compensatory_paralog(),
            BiologicalRules.R6_drug_pathway_target(),
        ]

    @staticmethod
    def get_rules_by_conclusion_type(
        conclusion_type: ConclusionType,
    ) -> List[Rule]:
        """
        Get rules that produce a specific type of conclusion.

        Args:
            conclusion_type: The type of conclusion to filter by

        Returns:
            List of rules producing that conclusion type
        """
        all_rules = BiologicalRules.get_all_rules()
        return [
            r for r in all_rules
            if r.conclusion.type == conclusion_type.value
        ]

    @staticmethod
    def get_rule_by_id(rule_id: str) -> Optional[Rule]:
        """
        Get a specific rule by its ID.

        Args:
            rule_id: The rule identifier (e.g., "R1", "R3b")

        Returns:
            The Rule object or None if not found
        """
        for rule in BiologicalRules.get_all_rules():
            if rule.id == rule_id:
                return rule
        return None
