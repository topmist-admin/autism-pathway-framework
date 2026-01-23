"""
Reasoning chain and explanation generation for symbolic rules.

This module provides facilities for generating human-readable explanations
of rule-based inferences, including reasoning chains that connect
genetic variants to biological conclusions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
import json
import sys
from pathlib import Path

# Add module to path for imports
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from biological_rules import Rule, Conclusion, ConclusionType
from rule_engine import FiredRule


@dataclass
class ReasoningStep:
    """
    A single step in a reasoning chain.

    Attributes:
        step_number: Position in the chain
        description: What was determined
        evidence: Supporting evidence
        rule_id: Rule that produced this step (if any)
        confidence: Confidence in this step
    """
    step_number: int
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    rule_id: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step_number,
            "description": self.description,
            "evidence": self.evidence,
            "rule_id": self.rule_id,
            "confidence": self.confidence,
        }


@dataclass
class ReasoningChain:
    """
    Chain of reasoning from variants to conclusions.

    This represents the complete reasoning process for an individual,
    showing how genetic variants lead to biological conclusions.

    Attributes:
        individual_id: Sample identifier
        fired_rules: Rules that fired
        pathway_conclusions: Pathway -> confidence mapping
        subtype_indicators: List of subtype indicators identified
        therapeutic_hypotheses: Therapeutic hypothesis candidates
        explanation_text: Full human-readable explanation
        steps: Individual reasoning steps
        timestamp: When the chain was generated
    """
    individual_id: str
    fired_rules: List[FiredRule] = field(default_factory=list)
    pathway_conclusions: Dict[str, float] = field(default_factory=dict)
    subtype_indicators: List[str] = field(default_factory=list)
    therapeutic_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    explanation_text: str = ""
    steps: List[ReasoningStep] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "individual_id": self.individual_id,
            "fired_rules": [fr.to_dict() for fr in self.fired_rules],
            "pathway_conclusions": self.pathway_conclusions,
            "subtype_indicators": self.subtype_indicators,
            "therapeutic_hypotheses": self.therapeutic_hypotheses,
            "explanation_text": self.explanation_text,
            "steps": [s.to_dict() for s in self.steps],
            "timestamp": self.timestamp.isoformat(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def n_rules_fired(self) -> int:
        """Number of rules that fired."""
        return len(self.fired_rules)

    @property
    def average_confidence(self) -> float:
        """Average confidence across fired rules."""
        if not self.fired_rules:
            return 0.0
        return sum(fr.confidence for fr in self.fired_rules) / len(self.fired_rules)

    @property
    def genes_affected(self) -> Set[str]:
        """Set of genes affected by fired rules."""
        genes = set()
        for fr in self.fired_rules:
            gene = fr.bindings.get("G") or fr.bindings.get("gene")
            if gene:
                genes.add(gene)
        return genes

    def get_rules_by_type(self, conclusion_type: str) -> List[FiredRule]:
        """Get fired rules by conclusion type."""
        return [
            fr for fr in self.fired_rules
            if fr.rule.conclusion.type == conclusion_type
        ]


class ExplanationGenerator:
    """
    Generates human-readable explanations for rule-based inferences.

    This class provides methods for converting fired rules into
    comprehensive reasoning chains with explanations.
    """

    def __init__(
        self,
        include_evidence: bool = True,
        include_disclaimers: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize explanation generator.

        Args:
            include_evidence: Include evidence details in explanations
            include_disclaimers: Include research disclaimers
            verbose: Generate verbose explanations
        """
        self.include_evidence = include_evidence
        self.include_disclaimers = include_disclaimers
        self.verbose = verbose

    def generate_reasoning_chain(
        self,
        individual_id: str,
        fired_rules: List[FiredRule],
    ) -> ReasoningChain:
        """
        Generate a complete reasoning chain from fired rules.

        Args:
            individual_id: Sample identifier
            fired_rules: List of rules that fired

        Returns:
            Complete ReasoningChain with explanations
        """
        # Build the chain
        chain = ReasoningChain(
            individual_id=individual_id,
            fired_rules=fired_rules,
        )

        # Extract conclusions by type
        chain.pathway_conclusions = self._extract_pathway_conclusions(fired_rules)
        chain.subtype_indicators = self._extract_subtype_indicators(fired_rules)
        chain.therapeutic_hypotheses = self._extract_therapeutic_hypotheses(fired_rules)

        # Generate reasoning steps
        chain.steps = self._generate_steps(fired_rules)

        # Generate full explanation text
        chain.explanation_text = self._generate_explanation_text(chain)

        return chain

    def _extract_pathway_conclusions(
        self,
        fired_rules: List[FiredRule],
    ) -> Dict[str, float]:
        """Extract pathway disruption conclusions."""
        pathways = {}
        for fr in fired_rules:
            if fr.rule.conclusion.type in (
                ConclusionType.PATHWAY_DISRUPTION.value,
                ConclusionType.PATHWAY_CONVERGENCE.value,
            ):
                # Get pathway from evidence or attributes
                pathway = (
                    fr.evidence.get("pathway") or
                    fr.rule.conclusion.attributes.get("pathway") or
                    "unknown_pathway"
                )
                # Use max confidence for same pathway
                if pathway not in pathways or fr.confidence > pathways[pathway]:
                    pathways[pathway] = fr.confidence

        return pathways

    def _extract_subtype_indicators(
        self,
        fired_rules: List[FiredRule],
    ) -> List[str]:
        """Extract subtype indicator conclusions."""
        indicators = set()
        for fr in fired_rules:
            if fr.rule.conclusion.type == ConclusionType.SUBTYPE_INDICATOR.value:
                subtype = fr.rule.conclusion.attributes.get("subtype")
                if subtype:
                    indicators.add(subtype)
            # Also check for subtype_indicator in attributes
            subtype = fr.rule.conclusion.attributes.get("subtype_indicator")
            if subtype:
                indicators.add(subtype)

        return list(indicators)

    def _extract_therapeutic_hypotheses(
        self,
        fired_rules: List[FiredRule],
    ) -> List[Dict[str, Any]]:
        """Extract therapeutic hypothesis conclusions."""
        hypotheses = []
        for fr in fired_rules:
            if fr.rule.conclusion.type == ConclusionType.THERAPEUTIC_HYPOTHESIS.value:
                hypothesis = {
                    "rule_id": fr.rule.id,
                    "confidence": fr.confidence,
                    "attributes": fr.rule.conclusion.attributes.copy(),
                    "evidence": fr.evidence.copy(),
                }
                hypotheses.append(hypothesis)

        return hypotheses

    def _generate_steps(
        self,
        fired_rules: List[FiredRule],
    ) -> List[ReasoningStep]:
        """Generate reasoning steps from fired rules."""
        steps = []
        step_num = 1

        # Group rules by gene for cleaner presentation
        rules_by_gene: Dict[str, List[FiredRule]] = {}
        general_rules: List[FiredRule] = []

        for fr in fired_rules:
            gene = fr.bindings.get("G") or fr.bindings.get("gene")
            if gene:
                if gene not in rules_by_gene:
                    rules_by_gene[gene] = []
                rules_by_gene[gene].append(fr)
            else:
                general_rules.append(fr)

        # Step 1: Variant identification
        if rules_by_gene:
            genes = list(rules_by_gene.keys())
            steps.append(ReasoningStep(
                step_number=step_num,
                description=f"Identified damaging variants in {len(genes)} gene(s): {', '.join(genes[:5])}{'...' if len(genes) > 5 else ''}",
                evidence={"genes": genes},
            ))
            step_num += 1

        # Step 2-N: Gene-specific rule firing
        for gene, gene_rules in rules_by_gene.items():
            for fr in gene_rules:
                steps.append(ReasoningStep(
                    step_number=step_num,
                    description=self._describe_rule_firing(fr, gene),
                    evidence=fr.evidence,
                    rule_id=fr.rule.id,
                    confidence=fr.confidence,
                ))
                step_num += 1

        # General rules
        for fr in general_rules:
            steps.append(ReasoningStep(
                step_number=step_num,
                description=self._describe_rule_firing(fr, None),
                evidence=fr.evidence,
                rule_id=fr.rule.id,
                confidence=fr.confidence,
            ))
            step_num += 1

        return steps

    def _describe_rule_firing(
        self,
        fired_rule: FiredRule,
        gene: Optional[str],
    ) -> str:
        """Generate description for a rule firing."""
        rule = fired_rule.rule

        if rule.id == "R1":
            return (
                f"Gene {gene} meets high-confidence criteria: "
                f"constrained (pLI ≥ 0.9), prenatally expressed in cortex, "
                f"with loss-of-function variant → pathway disruption"
            )

        elif rule.id == "R2":
            hit_genes = fired_rule.evidence.get("hit_genes", [])
            pathway = fired_rule.evidence.get("pathway", "pathway")
            return (
                f"Pathway convergence detected: {len(hit_genes)} independent genes "
                f"({', '.join(hit_genes[:3])}) hit in {pathway}"
            )

        elif rule.id == "R3":
            return (
                f"CHD8 disruption detected in {gene} → "
                f"chromatin remodeling cascade affected"
            )

        elif rule.id == "R3b":
            return (
                f"CHD8 regulatory target {gene} disrupted → "
                f"potential chromatin cascade effect"
            )

        elif rule.id in ("R4", "R4b"):
            cell_type = rule.conclusion.attributes.get("cell_type", "neuron")
            return (
                f"Synaptic gene {gene} disrupted with {cell_type} "
                f"neuron enrichment → synaptic dysfunction subtype indicator"
            )

        elif rule.id == "R5":
            paralog = fired_rule.bindings.get("P", "paralog")
            return (
                f"Paralog {paralog} intact and expressed for disrupted gene {gene} → "
                f"potential compensation (reduced penetrance modifier)"
            )

        elif rule.id == "R6":
            drug = fired_rule.evidence.get("drug", "drug")
            pathway = fired_rule.evidence.get("pathway", "pathway")
            return (
                f"Drug {drug} targets disrupted pathway {pathway} → "
                f"therapeutic hypothesis (requires validation)"
            )

        elif rule.id == "R7":
            return (
                f"High-confidence SFARI gene {gene} with damaging variant → "
                f"strong autism association"
            )

        # Default description
        return f"Rule {rule.id} ({rule.name}) fired for {gene or 'individual'}"

    def _generate_explanation_text(
        self,
        chain: ReasoningChain,
    ) -> str:
        """Generate full explanation text for a reasoning chain."""
        parts = []

        # Header
        parts.append(f"=== Reasoning Chain for {chain.individual_id} ===\n")
        parts.append(f"Generated: {chain.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        parts.append(f"Rules fired: {chain.n_rules_fired}\n")
        parts.append(f"Average confidence: {chain.average_confidence:.1%}\n")

        # Summary
        parts.append("\n--- Summary ---\n")

        if chain.genes_affected:
            genes = list(chain.genes_affected)
            parts.append(f"Genes affected: {', '.join(genes[:10])}")
            if len(genes) > 10:
                parts.append(f"... and {len(genes) - 10} more")
            parts.append("\n")

        if chain.pathway_conclusions:
            parts.append(f"\nPathway conclusions:\n")
            for pathway, conf in sorted(
                chain.pathway_conclusions.items(),
                key=lambda x: -x[1]
            )[:5]:
                parts.append(f"  - {pathway}: {conf:.1%} confidence\n")

        if chain.subtype_indicators:
            parts.append(f"\nSubtype indicators: {', '.join(chain.subtype_indicators)}\n")

        if chain.therapeutic_hypotheses:
            parts.append(f"\nTherapeutic hypotheses: {len(chain.therapeutic_hypotheses)}\n")

        # Detailed steps
        if self.verbose and chain.steps:
            parts.append("\n--- Reasoning Steps ---\n")
            for step in chain.steps:
                parts.append(f"\nStep {step.step_number}: {step.description}\n")
                if step.rule_id:
                    parts.append(f"  Rule: {step.rule_id}\n")
                    parts.append(f"  Confidence: {step.confidence:.1%}\n")
                if self.include_evidence and step.evidence:
                    for key, value in step.evidence.items():
                        if key not in ("gene", "genes"):  # Avoid redundancy
                            parts.append(f"  {key}: {value}\n")

        # Disclaimers
        if self.include_disclaimers:
            parts.append("\n--- Disclaimers ---\n")
            parts.append(
                "This analysis is for research purposes only. "
                "Findings should be validated independently. "
                "Therapeutic hypotheses require clinical validation "
                "and are not treatment recommendations.\n"
            )

        return "".join(parts)

    def generate_clinical_summary(
        self,
        chain: ReasoningChain,
    ) -> str:
        """
        Generate a concise clinical-style summary.

        Args:
            chain: Reasoning chain to summarize

        Returns:
            Brief clinical summary
        """
        parts = []

        parts.append(f"Sample: {chain.individual_id}\n\n")

        # Key findings
        parts.append("KEY FINDINGS:\n")

        # High-confidence pathway disruptions
        high_conf_pathways = [
            (p, c) for p, c in chain.pathway_conclusions.items()
            if c >= 0.8
        ]
        if high_conf_pathways:
            parts.append("\nHigh-confidence pathway disruptions:\n")
            for pathway, conf in sorted(high_conf_pathways, key=lambda x: -x[1]):
                parts.append(f"  • {pathway} ({conf:.0%})\n")

        # Subtype indicators
        if chain.subtype_indicators:
            parts.append(f"\nSubtype indicators: {', '.join(chain.subtype_indicators)}\n")

        # Genes of interest
        genes = chain.genes_affected
        if genes:
            # Highlight high-confidence genes
            high_conf_genes = set()
            for fr in chain.fired_rules:
                if fr.confidence >= 0.85:
                    gene = fr.bindings.get("G") or fr.bindings.get("gene")
                    if gene:
                        high_conf_genes.add(gene)

            if high_conf_genes:
                parts.append(f"\nHigh-confidence gene hits: {', '.join(high_conf_genes)}\n")

        # Therapeutic hypotheses
        if chain.therapeutic_hypotheses:
            parts.append(f"\nTherapeutic hypotheses for investigation: {len(chain.therapeutic_hypotheses)}\n")
            parts.append("  (Require validation - not treatment recommendations)\n")

        return "".join(parts)

    def compare_individuals(
        self,
        chains: List[ReasoningChain],
    ) -> str:
        """
        Generate comparison of multiple individuals.

        Args:
            chains: List of reasoning chains to compare

        Returns:
            Comparison summary
        """
        parts = []
        parts.append(f"=== Comparison of {len(chains)} Individuals ===\n\n")

        # Collect statistics
        all_pathways: Dict[str, int] = {}
        all_subtypes: Dict[str, int] = {}
        all_genes: Dict[str, int] = {}

        for chain in chains:
            for pathway in chain.pathway_conclusions:
                all_pathways[pathway] = all_pathways.get(pathway, 0) + 1
            for subtype in chain.subtype_indicators:
                all_subtypes[subtype] = all_subtypes.get(subtype, 0) + 1
            for gene in chain.genes_affected:
                all_genes[gene] = all_genes.get(gene, 0) + 1

        # Report shared pathways
        if all_pathways:
            parts.append("Pathways disrupted (count across individuals):\n")
            for pathway, count in sorted(all_pathways.items(), key=lambda x: -x[1])[:10]:
                pct = count / len(chains) * 100
                parts.append(f"  {pathway}: {count}/{len(chains)} ({pct:.0f}%)\n")

        # Report shared subtypes
        if all_subtypes:
            parts.append("\nSubtype indicators:\n")
            for subtype, count in sorted(all_subtypes.items(), key=lambda x: -x[1]):
                pct = count / len(chains) * 100
                parts.append(f"  {subtype}: {count}/{len(chains)} ({pct:.0f}%)\n")

        # Report recurrent genes
        recurrent_genes = {g: c for g, c in all_genes.items() if c > 1}
        if recurrent_genes:
            parts.append("\nRecurrent gene hits:\n")
            for gene, count in sorted(recurrent_genes.items(), key=lambda x: -x[1])[:15]:
                parts.append(f"  {gene}: {count} individuals\n")

        return "".join(parts)


def format_fired_rule(fired_rule: FiredRule, verbose: bool = False) -> str:
    """
    Format a single fired rule for display.

    Args:
        fired_rule: The fired rule to format
        verbose: Include detailed evidence

    Returns:
        Formatted string
    """
    rule = fired_rule.rule
    parts = [
        f"[{rule.id}] {rule.name}",
        f"  Confidence: {fired_rule.confidence:.1%}",
    ]

    gene = fired_rule.bindings.get("G") or fired_rule.bindings.get("gene")
    if gene:
        parts.append(f"  Gene: {gene}")

    parts.append(f"  Conclusion: {rule.conclusion.type}")

    if verbose:
        parts.append(f"  Description: {rule.description}")
        if fired_rule.evidence:
            parts.append("  Evidence:")
            for key, value in fired_rule.evidence.items():
                parts.append(f"    {key}: {value}")

    return "\n".join(parts)


def format_rule_summary(fired_rules: List[FiredRule]) -> str:
    """
    Format a summary of multiple fired rules.

    Args:
        fired_rules: List of fired rules

    Returns:
        Summary string
    """
    if not fired_rules:
        return "No rules fired."

    parts = [f"Rules fired: {len(fired_rules)}\n"]

    # Group by conclusion type
    by_type: Dict[str, List[FiredRule]] = {}
    for fr in fired_rules:
        ctype = fr.rule.conclusion.type
        if ctype not in by_type:
            by_type[ctype] = []
        by_type[ctype].append(fr)

    for ctype, rules in by_type.items():
        parts.append(f"\n{ctype} ({len(rules)}):")
        for fr in rules[:5]:  # Limit display
            gene = fr.bindings.get("G") or fr.bindings.get("gene") or ""
            parts.append(f"  - {fr.rule.id}: {fr.rule.name}")
            if gene:
                parts.append(f" [{gene}]")
            parts.append(f" ({fr.confidence:.0%})\n")

        if len(rules) > 5:
            parts.append(f"  ... and {len(rules) - 5} more\n")

    return "".join(parts)
