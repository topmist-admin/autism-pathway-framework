"""
Condition data structures and evaluators for symbolic rule-based inference.

This module defines the building blocks for biological rules:
- Condition: A single testable predicate
- ConditionEvaluator: Evaluates conditions against biological context
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum


class ConditionType(Enum):
    """Types of biological conditions that can be evaluated."""

    # Variant-level conditions
    HAS_VARIANT = "has_variant"
    HAS_LOF_VARIANT = "has_lof_variant"
    HAS_MISSENSE_VARIANT = "has_missense_variant"
    HAS_DAMAGING_VARIANT = "has_damaging_variant"
    HAS_MULTIPLE_HITS = "has_multiple_hits"

    # Gene-level conditions
    IS_CONSTRAINED = "is_constrained"
    IS_SFARI_GENE = "is_sfari_gene"
    IS_HIGH_CONFIDENCE_SFARI = "is_high_confidence_sfari"
    HAS_PARALOG = "has_paralog"
    IS_CHD8_TARGET = "is_chd8_target"
    IS_SYNAPTIC_GENE = "is_synaptic_gene"

    # Expression conditions
    EXPRESSED_IN = "expressed_in"
    PRENATALLY_EXPRESSED = "prenatally_expressed"
    CELL_TYPE_ENRICHED = "cell_type_enriched"

    # Pathway conditions
    GENE_IN_PATHWAY = "gene_in_pathway"
    PATHWAY_DISRUPTED = "pathway_disrupted"
    HITS_ARE_INDEPENDENT = "hits_are_independent"

    # Drug/therapeutic conditions
    DRUG_TARGETS_GENE = "drug_targets_gene"
    MECHANISM_ALIGNMENT = "mechanism_alignment"


@dataclass
class Condition:
    """
    A single condition in a biological rule.

    Conditions are predicates that can be evaluated against biological
    data to determine if they are satisfied.

    Attributes:
        predicate: The type of condition (e.g., "has_lof_variant", "is_constrained")
        arguments: Parameters for the condition evaluation
        negated: If True, the condition is satisfied when the predicate is False
    """
    predicate: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    negated: bool = False

    def __post_init__(self):
        """Validate condition after initialization."""
        if not self.predicate:
            raise ValueError("Condition predicate cannot be empty")

    def __str__(self) -> str:
        """Human-readable representation."""
        neg = "NOT " if self.negated else ""
        args = ", ".join(f"{k}={v}" for k, v in self.arguments.items())
        return f"{neg}{self.predicate}({args})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "predicate": self.predicate,
            "arguments": self.arguments,
            "negated": self.negated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Condition":
        """Create from dictionary."""
        return cls(
            predicate=data["predicate"],
            arguments=data.get("arguments", {}),
            negated=data.get("negated", False),
        )


@dataclass
class ConditionResult:
    """
    Result of evaluating a condition.

    Attributes:
        satisfied: Whether the condition was met
        evidence: Supporting data for the result
        bound_variables: Variables that were bound during evaluation
        explanation: Human-readable explanation
    """
    satisfied: bool
    evidence: Dict[str, Any] = field(default_factory=dict)
    bound_variables: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


class ConditionEvaluator:
    """
    Evaluates conditions against biological context and individual data.

    This class provides the evaluation logic for all supported condition types.
    """

    def __init__(self, biological_context: "BiologicalContext"):
        """
        Initialize evaluator with biological reference data.

        Args:
            biological_context: Reference data for evaluation (constraints, expression, etc.)
        """
        self.context = biological_context

        # Register condition evaluators
        self._evaluators = {
            # Variant conditions
            "has_variant": self._eval_has_variant,
            "has_lof_variant": self._eval_has_lof_variant,
            "has_missense_variant": self._eval_has_missense_variant,
            "has_damaging_variant": self._eval_has_damaging_variant,
            "has_multiple_hits": self._eval_has_multiple_hits,

            # Gene conditions
            "is_constrained": self._eval_is_constrained,
            "gene_constraint": self._eval_gene_constraint,  # Alias
            "is_sfari_gene": self._eval_is_sfari_gene,
            "is_high_confidence_sfari": self._eval_is_high_confidence_sfari,
            "has_paralog": self._eval_has_paralog,
            "is_chd8_target": self._eval_is_chd8_target,
            "is_synaptic_gene": self._eval_is_synaptic_gene,

            # Expression conditions
            "expressed_in": self._eval_expressed_in,
            "prenatally_expressed": self._eval_prenatally_expressed,
            "cell_type_expression": self._eval_cell_type_expression,
            "cell_type_enriched": self._eval_cell_type_enriched,

            # Pathway conditions
            "gene_in_pathway": self._eval_gene_in_pathway,
            "pathway_disrupted": self._eval_pathway_disrupted,
            "hits_are_independent": self._eval_hits_are_independent,

            # Drug conditions
            "drug_targets": self._eval_drug_targets,
            "mechanism_alignment": self._eval_mechanism_alignment,
        }

    def evaluate(
        self,
        condition: Condition,
        individual_data: "IndividualData",
        bindings: Optional[Dict[str, Any]] = None,
    ) -> ConditionResult:
        """
        Evaluate a condition against individual data.

        Args:
            condition: The condition to evaluate
            individual_data: Data for the individual being evaluated
            bindings: Pre-existing variable bindings (for chained conditions)

        Returns:
            ConditionResult with satisfaction status and evidence
        """
        bindings = bindings or {}

        # Resolve any variable references in arguments
        resolved_args = self._resolve_arguments(condition.arguments, bindings)

        # Get the appropriate evaluator
        evaluator = self._evaluators.get(condition.predicate)
        if evaluator is None:
            return ConditionResult(
                satisfied=False,
                explanation=f"Unknown predicate: {condition.predicate}",
            )

        # Evaluate the condition
        result = evaluator(resolved_args, individual_data, bindings)

        # Handle negation
        if condition.negated:
            result.satisfied = not result.satisfied
            result.explanation = f"NOT ({result.explanation})"

        return result

    def _resolve_arguments(
        self,
        arguments: Dict[str, Any],
        bindings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve variable references in arguments."""
        resolved = {}
        for key, value in arguments.items():
            if isinstance(value, str) and value in bindings:
                resolved[key] = bindings[value]
            else:
                resolved[key] = value
        return resolved

    # ==================== Variant Condition Evaluators ====================

    def _eval_has_variant(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if individual has a variant in specified gene."""
        gene = args.get("gene")
        variant_type = args.get("variant_type")

        # Get variants for this individual
        variants = individual.get_gene_variants(gene)

        if not variants:
            return ConditionResult(
                satisfied=False,
                explanation=f"No variants found in gene {gene}",
            )

        # Filter by variant type if specified
        if variant_type:
            if variant_type == "loss_of_function":
                variants = [v for v in variants if v.is_lof]
            elif variant_type == "damaging":
                variants = [v for v in variants if v.is_damaging]
            elif variant_type == "missense":
                variants = [v for v in variants if v.is_missense]

        if variants:
            return ConditionResult(
                satisfied=True,
                evidence={"variants": [str(v) for v in variants[:5]]},  # Limit for readability
                bound_variables={"G": gene} if "G" in str(args) else {},
                explanation=f"Found {len(variants)} variant(s) in {gene}",
            )

        return ConditionResult(
            satisfied=False,
            explanation=f"No {variant_type or 'matching'} variants in {gene}",
        )

    def _eval_has_lof_variant(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if individual has loss-of-function variant."""
        gene = args.get("gene")

        if gene:
            variants = individual.get_gene_variants(gene)
            lof_variants = [v for v in variants if v.is_lof]
        else:
            lof_variants = [v for v in individual.variants if v.is_lof]

        if lof_variants:
            affected_genes = list(set(v.gene_id for v in lof_variants if v.gene_id))
            return ConditionResult(
                satisfied=True,
                evidence={
                    "lof_count": len(lof_variants),
                    "affected_genes": affected_genes[:10],
                },
                bound_variables={"lof_genes": affected_genes},
                explanation=f"Found {len(lof_variants)} LoF variant(s) in {len(affected_genes)} gene(s)",
            )

        return ConditionResult(
            satisfied=False,
            explanation="No loss-of-function variants found",
        )

    def _eval_has_missense_variant(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if individual has missense variant."""
        gene = args.get("gene")
        cadd_threshold = args.get("cadd_threshold", 20.0)

        if gene:
            variants = individual.get_gene_variants(gene)
        else:
            variants = individual.variants

        missense = [v for v in variants if v.is_missense]

        # Filter by CADD if threshold specified
        if cadd_threshold:
            missense = [
                v for v in missense
                if v.cadd_phred and v.cadd_phred >= cadd_threshold
            ]

        if missense:
            return ConditionResult(
                satisfied=True,
                evidence={"missense_count": len(missense)},
                explanation=f"Found {len(missense)} damaging missense variant(s)",
            )

        return ConditionResult(
            satisfied=False,
            explanation="No damaging missense variants found",
        )

    def _eval_has_damaging_variant(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if individual has damaging variant (LoF or damaging missense)."""
        gene = args.get("gene")

        if gene:
            variants = individual.get_gene_variants(gene)
        else:
            variants = individual.variants

        damaging = [v for v in variants if v.is_damaging]

        if damaging:
            return ConditionResult(
                satisfied=True,
                evidence={
                    "damaging_count": len(damaging),
                    "genes": list(set(v.gene_id for v in damaging if v.gene_id))[:10],
                },
                explanation=f"Found {len(damaging)} damaging variant(s)",
            )

        return ConditionResult(
            satisfied=False,
            explanation="No damaging variants found",
        )

    def _eval_has_multiple_hits(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if individual has multiple damaging hits in a pathway."""
        min_genes = args.get("min_genes", 2)
        pathway = args.get("pathway")

        if not pathway:
            return ConditionResult(
                satisfied=False,
                explanation="No pathway specified for multiple hits check",
            )

        # Get genes in pathway
        pathway_genes = self.context.get_pathway_genes(pathway)
        if not pathway_genes:
            return ConditionResult(
                satisfied=False,
                explanation=f"Pathway {pathway} not found or empty",
            )

        # Count damaging variants in pathway genes
        hit_genes = set()
        for gene in pathway_genes:
            variants = individual.get_gene_variants(gene)
            if any(v.is_damaging for v in variants):
                hit_genes.add(gene)

        if len(hit_genes) >= min_genes:
            return ConditionResult(
                satisfied=True,
                evidence={
                    "hit_genes": list(hit_genes),
                    "hit_count": len(hit_genes),
                    "pathway": pathway,
                },
                bound_variables={"hit_genes": list(hit_genes)},
                explanation=f"Found {len(hit_genes)} gene(s) with damaging variants in {pathway}",
            )

        return ConditionResult(
            satisfied=False,
            explanation=f"Only {len(hit_genes)} gene(s) hit in pathway (need {min_genes})",
        )

    # ==================== Gene Condition Evaluators ====================

    def _eval_is_constrained(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if gene is evolutionarily constrained."""
        gene = args.get("gene")
        pli_threshold = args.get("pli_threshold", 0.9)

        if not gene:
            return ConditionResult(
                satisfied=False,
                explanation="No gene specified for constraint check",
            )

        is_constrained = self.context.is_gene_constrained(gene, pli_threshold)
        pli = self.context.get_pli_score(gene)

        if is_constrained:
            return ConditionResult(
                satisfied=True,
                evidence={"gene": gene, "pLI": pli, "threshold": pli_threshold},
                explanation=f"{gene} is constrained (pLI={pli:.3f} >= {pli_threshold})",
            )

        return ConditionResult(
            satisfied=False,
            explanation=f"{gene} is not constrained (pLI={pli:.3f} < {pli_threshold})" if pli else f"{gene} has no constraint data",
        )

    def _eval_gene_constraint(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Alias for is_constrained."""
        return self._eval_is_constrained(args, individual, bindings)

    def _eval_is_sfari_gene(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if gene is in SFARI database."""
        gene = args.get("gene")
        max_score = args.get("max_score", 3)  # 1=high, 2=strong, 3=suggestive

        if not gene:
            return ConditionResult(
                satisfied=False,
                explanation="No gene specified for SFARI check",
            )

        is_sfari = self.context.is_sfari_gene(gene, max_score)
        score = self.context.get_sfari_score(gene)

        if is_sfari:
            return ConditionResult(
                satisfied=True,
                evidence={"gene": gene, "sfari_score": score},
                explanation=f"{gene} is SFARI gene (score={score})",
            )

        return ConditionResult(
            satisfied=False,
            explanation=f"{gene} is not a SFARI gene (score <= {max_score})",
        )

    def _eval_is_high_confidence_sfari(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if gene is high-confidence SFARI gene."""
        args["max_score"] = 1
        return self._eval_is_sfari_gene(args, individual, bindings)

    def _eval_has_paralog(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if gene has a paralog."""
        gene = args.get("gene")
        paralog = args.get("paralog")  # Specific paralog to check

        if not gene:
            return ConditionResult(
                satisfied=False,
                explanation="No gene specified for paralog check",
            )

        paralogs = self.context.get_paralogs(gene)

        if paralog:
            # Check specific paralog
            if paralog in paralogs:
                return ConditionResult(
                    satisfied=True,
                    evidence={"gene": gene, "paralog": paralog},
                    bound_variables={"P": paralog},
                    explanation=f"{gene} has paralog {paralog}",
                )
            return ConditionResult(
                satisfied=False,
                explanation=f"{paralog} is not a paralog of {gene}",
            )

        if paralogs:
            return ConditionResult(
                satisfied=True,
                evidence={"gene": gene, "paralogs": list(paralogs)[:5]},
                bound_variables={"P": list(paralogs)[0] if paralogs else None},
                explanation=f"{gene} has {len(paralogs)} paralog(s)",
            )

        return ConditionResult(
            satisfied=False,
            explanation=f"{gene} has no known paralogs",
        )

    def _eval_is_chd8_target(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if gene is a CHD8 regulatory target."""
        gene = args.get("gene")

        if not gene:
            return ConditionResult(
                satisfied=False,
                explanation="No gene specified for CHD8 target check",
            )

        is_target = self.context.is_chd8_target(gene)

        if is_target:
            return ConditionResult(
                satisfied=True,
                evidence={"gene": gene, "is_chd8_target": True},
                explanation=f"{gene} is a CHD8 regulatory target",
            )

        return ConditionResult(
            satisfied=False,
            explanation=f"{gene} is not a known CHD8 target",
        )

    def _eval_is_synaptic_gene(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if gene is annotated as synaptic (SynGO)."""
        gene = args.get("gene")
        ontology = args.get("ontology", "SynGO")

        if not gene:
            return ConditionResult(
                satisfied=False,
                explanation="No gene specified for synaptic check",
            )

        is_synaptic = self.context.is_synaptic_gene(gene)

        if is_synaptic:
            return ConditionResult(
                satisfied=True,
                evidence={"gene": gene, "ontology": ontology},
                explanation=f"{gene} is a synaptic gene ({ontology})",
            )

        return ConditionResult(
            satisfied=False,
            explanation=f"{gene} is not annotated as synaptic",
        )

    # ==================== Expression Condition Evaluators ====================

    def _eval_expressed_in(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if gene is expressed in specified tissue/stage."""
        gene = args.get("gene")
        tissue = args.get("tissue")
        stage = args.get("stage")
        level = args.get("level", "any")  # "any", "high", "moderate"
        threshold = args.get("threshold", 1.0)

        if not gene:
            return ConditionResult(
                satisfied=False,
                explanation="No gene specified for expression check",
            )

        expression = self.context.get_expression(gene, tissue, stage)

        if expression is not None and expression >= threshold:
            return ConditionResult(
                satisfied=True,
                evidence={
                    "gene": gene,
                    "tissue": tissue,
                    "stage": stage,
                    "expression": float(expression),
                },
                explanation=f"{gene} is expressed in {tissue}/{stage} (level={expression:.2f})",
            )

        return ConditionResult(
            satisfied=False,
            explanation=f"{gene} not expressed in {tissue}/{stage} (threshold={threshold})",
        )

    def _eval_prenatally_expressed(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if gene is expressed prenatally in cortex."""
        gene = args.get("gene")
        threshold = args.get("threshold", 1.0)

        if not gene:
            return ConditionResult(
                satisfied=False,
                explanation="No gene specified for prenatal expression check",
            )

        is_prenatal = self.context.is_prenatally_expressed(gene, threshold)
        expression = self.context.get_prenatal_expression(gene)

        if is_prenatal:
            return ConditionResult(
                satisfied=True,
                evidence={
                    "gene": gene,
                    "prenatal_expression": float(expression) if expression else None,
                },
                explanation=f"{gene} is prenatally expressed in cortex",
            )

        return ConditionResult(
            satisfied=False,
            explanation=f"{gene} is not prenatally expressed",
        )

    def _eval_cell_type_expression(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if gene is expressed in specific cell type."""
        gene = args.get("gene")
        cell_type = args.get("cell_type")
        enriched = args.get("enriched", False)
        threshold = args.get("fold_change", 2.0)

        if not gene or not cell_type:
            return ConditionResult(
                satisfied=False,
                explanation="Gene and cell_type required for cell type expression check",
            )

        if enriched:
            is_enriched = self.context.is_enriched_in_cell_type(gene, cell_type, threshold)
            if is_enriched:
                return ConditionResult(
                    satisfied=True,
                    evidence={"gene": gene, "cell_type": cell_type, "enriched": True},
                    explanation=f"{gene} is enriched in {cell_type}",
                )
            return ConditionResult(
                satisfied=False,
                explanation=f"{gene} is not enriched in {cell_type}",
            )

        # Just check if expressed
        expression = self.context.get_cell_type_expression(gene, cell_type)
        if expression and expression > 0:
            return ConditionResult(
                satisfied=True,
                evidence={"gene": gene, "cell_type": cell_type, "expression": expression},
                explanation=f"{gene} is expressed in {cell_type}",
            )

        return ConditionResult(
            satisfied=False,
            explanation=f"{gene} is not expressed in {cell_type}",
        )

    def _eval_cell_type_enriched(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if gene is enriched in specific cell type."""
        args["enriched"] = True
        return self._eval_cell_type_expression(args, individual, bindings)

    # ==================== Pathway Condition Evaluators ====================

    def _eval_gene_in_pathway(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if gene is member of pathway."""
        gene = args.get("gene")
        pathway = args.get("pathway")

        if not gene or not pathway:
            return ConditionResult(
                satisfied=False,
                explanation="Gene and pathway required",
            )

        is_member = self.context.is_gene_in_pathway(gene, pathway)

        if is_member:
            return ConditionResult(
                satisfied=True,
                evidence={"gene": gene, "pathway": pathway},
                explanation=f"{gene} is member of pathway {pathway}",
            )

        return ConditionResult(
            satisfied=False,
            explanation=f"{gene} is not in pathway {pathway}",
        )

    def _eval_pathway_disrupted(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if pathway is disrupted (high score) in individual."""
        pathway = args.get("pathway")
        score_threshold = args.get("score_threshold", 2.0)

        if not pathway:
            return ConditionResult(
                satisfied=False,
                explanation="No pathway specified",
            )

        score = individual.get_pathway_score(pathway)

        if score is not None and score >= score_threshold:
            return ConditionResult(
                satisfied=True,
                evidence={
                    "pathway": pathway,
                    "score": float(score),
                    "threshold": score_threshold,
                },
                explanation=f"Pathway {pathway} is disrupted (score={score:.2f})",
            )

        score_str = f"{score:.2f}" if score is not None else "N/A"
        return ConditionResult(
            satisfied=False,
            explanation=f"Pathway {pathway} not significantly disrupted (score={score_str})",
        )

    def _eval_hits_are_independent(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if multiple hits are in distinct genes (not compound het)."""
        pathway = args.get("pathway")
        hit_genes = bindings.get("hit_genes", [])

        if len(hit_genes) < 2:
            return ConditionResult(
                satisfied=False,
                explanation="Need at least 2 hit genes to check independence",
            )

        # Check that variants are in different genes
        # (This is automatically true if we counted distinct genes)
        return ConditionResult(
            satisfied=True,
            evidence={"independent_genes": hit_genes},
            explanation=f"Hits are in {len(hit_genes)} independent genes",
        )

    # ==================== Drug Condition Evaluators ====================

    def _eval_drug_targets(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if drug targets a gene."""
        drug = args.get("drug")
        target = args.get("target")

        if not drug:
            return ConditionResult(
                satisfied=False,
                explanation="No drug specified",
            )

        targets = self.context.get_drug_targets(drug)

        if target:
            if target in targets:
                return ConditionResult(
                    satisfied=True,
                    evidence={"drug": drug, "target": target},
                    explanation=f"{drug} targets {target}",
                )
            return ConditionResult(
                satisfied=False,
                explanation=f"{drug} does not target {target}",
            )

        if targets:
            return ConditionResult(
                satisfied=True,
                evidence={"drug": drug, "targets": list(targets)[:10]},
                bound_variables={"T": list(targets)[0]},
                explanation=f"{drug} has {len(targets)} known targets",
            )

        return ConditionResult(
            satisfied=False,
            explanation=f"No known targets for {drug}",
        )

    def _eval_mechanism_alignment(
        self,
        args: Dict[str, Any],
        individual: "IndividualData",
        bindings: Dict[str, Any],
    ) -> ConditionResult:
        """Check if drug mechanism aligns with pathway biology."""
        drug = args.get("drug")
        pathway = args.get("pathway")

        if not drug or not pathway:
            return ConditionResult(
                satisfied=False,
                explanation="Drug and pathway required for mechanism alignment",
            )

        # Get drug mechanism and pathway function
        drug_mechanism = self.context.get_drug_mechanism(drug)
        pathway_function = self.context.get_pathway_function(pathway)

        # Check alignment (simplified - real implementation would be more sophisticated)
        if drug_mechanism and pathway_function:
            # Check if mechanism keywords overlap with pathway function
            aligned = self.context.check_mechanism_pathway_alignment(drug, pathway)
            if aligned:
                return ConditionResult(
                    satisfied=True,
                    evidence={
                        "drug": drug,
                        "pathway": pathway,
                        "mechanism": drug_mechanism,
                    },
                    explanation=f"{drug} mechanism aligns with {pathway}",
                )

        return ConditionResult(
            satisfied=False,
            explanation=f"No clear mechanism alignment between {drug} and {pathway}",
        )
