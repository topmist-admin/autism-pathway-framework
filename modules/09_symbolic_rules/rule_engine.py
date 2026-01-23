"""
Rule evaluation engine for biological inference.

This module provides the RuleEngine class that evaluates biological rules
against individual genetic data and produces fired rules with explanations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
from datetime import datetime
import sys
from pathlib import Path

# Add module to path for imports
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from conditions import Condition, ConditionEvaluator, ConditionResult
from biological_rules import Rule, Conclusion, ConclusionType


logger = logging.getLogger(__name__)


@dataclass
class FiredRule:
    """
    A rule that has been evaluated and satisfied.

    Attributes:
        rule: The rule that fired
        bindings: Variable assignments that satisfied conditions
        confidence: Final confidence after modifiers
        explanation: Human-readable explanation
        evidence: Supporting data for this firing
        timestamp: When the rule fired
    """
    rule: Rule
    bindings: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    explanation: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Calculate confidence if not set."""
        if self.confidence == 0.0:
            self.confidence = (
                self.rule.base_confidence *
                self.rule.conclusion.confidence_modifier
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_id": self.rule.id,
            "rule_name": self.rule.name,
            "bindings": self.bindings,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "evidence": self.evidence,
            "conclusion": self.rule.conclusion.to_dict(),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class IndividualData:
    """
    Data for an individual to be evaluated against rules.

    This is the input data structure for rule evaluation.

    Attributes:
        sample_id: Unique identifier for the individual
        variants: Annotated variants for this individual
        gene_burdens: Gene burden scores (gene -> score)
        pathway_scores: Pathway scores (pathway -> score)
        metadata: Additional metadata
    """
    sample_id: str
    variants: List[Any] = field(default_factory=list)  # List[AnnotatedVariant]
    gene_burdens: Dict[str, float] = field(default_factory=dict)
    pathway_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_gene_variants(self, gene_id: str) -> List[Any]:
        """Get variants in a specific gene."""
        return [v for v in self.variants if getattr(v, 'gene_id', None) == gene_id]

    def get_pathway_score(self, pathway_id: str) -> Optional[float]:
        """Get score for a specific pathway."""
        return self.pathway_scores.get(pathway_id)

    def get_gene_burden(self, gene_id: str) -> float:
        """Get burden score for a specific gene."""
        return self.gene_burdens.get(gene_id, 0.0)

    def get_genes_with_variants(self) -> Set[str]:
        """Get set of genes with any variants."""
        return set(
            v.gene_id for v in self.variants
            if hasattr(v, 'gene_id') and v.gene_id
        )

    def get_genes_with_damaging_variants(self) -> Set[str]:
        """Get set of genes with damaging variants."""
        return set(
            v.gene_id for v in self.variants
            if hasattr(v, 'gene_id') and v.gene_id and
            hasattr(v, 'is_damaging') and v.is_damaging
        )


@dataclass
class BiologicalContext:
    """
    Biological reference data needed for rule evaluation.

    This provides the biological context against which conditions
    are evaluated.

    Attributes:
        gene_constraints: Gene constraint scores (pLI, LOEUF)
        developmental_expression: BrainSpan expression data
        single_cell_atlas: Cell-type expression data
        sfari_genes: SFARI autism gene annotations
        pathway_db: Pathway database
        paralog_map: Gene -> paralogs mapping
        chd8_targets: CHD8 regulatory target genes
        syngo_genes: SynGO synaptic genes
        drug_targets: Drug -> target genes mapping
        drug_mechanisms: Drug mechanism annotations
    """
    gene_constraints: Optional[Any] = None  # GeneConstraints
    developmental_expression: Optional[Any] = None  # DevelopmentalExpression
    single_cell_atlas: Optional[Any] = None  # SingleCellAtlas
    sfari_genes: Optional[Any] = None  # SFARIGenes
    pathway_db: Optional[Any] = None  # PathwayDatabase
    paralog_map: Dict[str, List[str]] = field(default_factory=dict)
    chd8_targets: Set[str] = field(default_factory=set)
    syngo_genes: Set[str] = field(default_factory=set)
    drug_targets: Dict[str, Set[str]] = field(default_factory=dict)
    drug_mechanisms: Dict[str, str] = field(default_factory=dict)
    pathway_functions: Dict[str, str] = field(default_factory=dict)

    # ==================== Constraint Methods ====================

    def is_gene_constrained(self, gene_id: str, pli_threshold: float = 0.9) -> bool:
        """Check if gene is evolutionarily constrained."""
        if self.gene_constraints is None:
            return False
        if hasattr(self.gene_constraints, 'is_constrained'):
            return self.gene_constraints.is_constrained(gene_id, pli_threshold)
        pli = self.get_pli_score(gene_id)
        return pli is not None and pli >= pli_threshold

    def get_pli_score(self, gene_id: str) -> Optional[float]:
        """Get pLI score for a gene."""
        if self.gene_constraints is None:
            return None
        if hasattr(self.gene_constraints, 'get_pli'):
            return self.gene_constraints.get_pli(gene_id)
        if hasattr(self.gene_constraints, 'pli_scores'):
            return self.gene_constraints.pli_scores.get(gene_id)
        return None

    # ==================== SFARI Methods ====================

    def is_sfari_gene(self, gene_id: str, max_score: int = 3) -> bool:
        """Check if gene is in SFARI database."""
        if self.sfari_genes is None:
            return False
        if hasattr(self.sfari_genes, 'is_sfari_gene'):
            return self.sfari_genes.is_sfari_gene(gene_id)
        score = self.get_sfari_score(gene_id)
        return score is not None and score <= max_score

    def get_sfari_score(self, gene_id: str) -> Optional[int]:
        """Get SFARI score for a gene."""
        if self.sfari_genes is None:
            return None
        if hasattr(self.sfari_genes, 'get_score'):
            return self.sfari_genes.get_score(gene_id)
        if hasattr(self.sfari_genes, 'scores'):
            return self.sfari_genes.scores.get(gene_id)
        return None

    def is_high_confidence_sfari(self, gene_id: str) -> bool:
        """Check if gene is high-confidence SFARI gene (score 1)."""
        if self.sfari_genes is None:
            return False
        if hasattr(self.sfari_genes, 'is_high_confidence'):
            return self.sfari_genes.is_high_confidence(gene_id)
        score = self.get_sfari_score(gene_id)
        return score == 1

    # ==================== Expression Methods ====================

    def is_prenatally_expressed(
        self,
        gene_id: str,
        threshold: float = 1.0,
    ) -> bool:
        """Check if gene is prenatally expressed in cortex."""
        if self.developmental_expression is None:
            return False
        if hasattr(self.developmental_expression, 'is_prenatally_expressed'):
            return self.developmental_expression.is_prenatally_expressed(
                gene_id, threshold
            )
        expr = self.get_prenatal_expression(gene_id)
        return expr is not None and expr >= threshold

    def get_prenatal_expression(self, gene_id: str) -> Optional[float]:
        """Get prenatal cortical expression for a gene."""
        if self.developmental_expression is None:
            return None
        if hasattr(self.developmental_expression, 'get_prenatal_expression'):
            expr = self.developmental_expression.get_prenatal_expression(gene_id)
            if expr is not None:
                return float(expr.mean()) if hasattr(expr, 'mean') else float(expr)
        return None

    def get_expression(
        self,
        gene_id: str,
        tissue: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Optional[float]:
        """Get expression level for a gene."""
        if self.developmental_expression is None:
            return None
        if hasattr(self.developmental_expression, 'get_expression'):
            return self.developmental_expression.get_expression(
                gene_id, stage, tissue
            )
        return None

    # ==================== Cell Type Methods ====================

    def get_cell_type_expression(
        self,
        gene_id: str,
        cell_type: str,
    ) -> Optional[float]:
        """Get expression in specific cell type."""
        if self.single_cell_atlas is None:
            return None
        if hasattr(self.single_cell_atlas, 'get_expression'):
            expr = self.single_cell_atlas.get_expression(gene_id, cell_type)
            if expr is not None:
                return float(expr)
        return None

    def is_enriched_in_cell_type(
        self,
        gene_id: str,
        cell_type: str,
        fold_change: float = 2.0,
    ) -> bool:
        """Check if gene is enriched in specific cell type."""
        if self.single_cell_atlas is None:
            return False
        if hasattr(self.single_cell_atlas, 'is_enriched_in'):
            return self.single_cell_atlas.is_enriched_in(
                gene_id, cell_type, fold_change
            )
        return False

    # ==================== Pathway Methods ====================

    def get_pathway_genes(self, pathway_id: str) -> Set[str]:
        """Get genes in a pathway."""
        if self.pathway_db is None:
            return set()
        if hasattr(self.pathway_db, 'get_pathway_genes'):
            return self.pathway_db.get_pathway_genes(pathway_id)
        if hasattr(self.pathway_db, 'pathways'):
            return self.pathway_db.pathways.get(pathway_id, set())
        return set()

    def is_gene_in_pathway(self, gene_id: str, pathway_id: str) -> bool:
        """Check if gene is in pathway."""
        return gene_id in self.get_pathway_genes(pathway_id)

    def get_pathway_function(self, pathway_id: str) -> Optional[str]:
        """Get functional description of pathway."""
        return self.pathway_functions.get(pathway_id)

    # ==================== Gene Annotation Methods ====================

    def get_paralogs(self, gene_id: str) -> Set[str]:
        """Get paralogs of a gene."""
        return set(self.paralog_map.get(gene_id, []))

    def is_chd8_target(self, gene_id: str) -> bool:
        """Check if gene is CHD8 target."""
        return gene_id in self.chd8_targets

    def is_synaptic_gene(self, gene_id: str) -> bool:
        """Check if gene is in SynGO."""
        return gene_id in self.syngo_genes

    # ==================== Drug Methods ====================

    def get_drug_targets(self, drug_id: str) -> Set[str]:
        """Get target genes for a drug."""
        return self.drug_targets.get(drug_id, set())

    def get_drug_mechanism(self, drug_id: str) -> Optional[str]:
        """Get mechanism for a drug."""
        return self.drug_mechanisms.get(drug_id)

    def check_mechanism_pathway_alignment(
        self,
        drug_id: str,
        pathway_id: str,
    ) -> bool:
        """Check if drug mechanism aligns with pathway function."""
        mechanism = self.get_drug_mechanism(drug_id)
        function = self.get_pathway_function(pathway_id)
        if not mechanism or not function:
            return False
        # Simple keyword overlap check (could be more sophisticated)
        mech_words = set(mechanism.lower().split())
        func_words = set(function.lower().split())
        return len(mech_words & func_words) > 0


class RuleEngine:
    """
    Engine for evaluating biological rules against individual data.

    The RuleEngine takes a set of rules and a biological context, then
    evaluates the rules against individual genetic data to produce
    fired rules with explanations.
    """

    def __init__(
        self,
        rules: List[Rule],
        biological_context: BiologicalContext,
    ):
        """
        Initialize rule engine.

        Args:
            rules: List of rules to evaluate
            biological_context: Reference data for condition evaluation
        """
        self.rules = sorted(rules, key=lambda r: -r.priority)  # High priority first
        self.context = biological_context
        self.condition_evaluator = ConditionEvaluator(biological_context)

    def evaluate(
        self,
        individual_data: IndividualData,
        rule_ids: Optional[List[str]] = None,
    ) -> List[FiredRule]:
        """
        Evaluate rules against individual data.

        Args:
            individual_data: Data for the individual
            rule_ids: Optional list of specific rules to evaluate

        Returns:
            List of FiredRule objects for rules that fired
        """
        fired_rules = []

        # Filter rules if specific IDs provided
        rules_to_evaluate = self.rules
        if rule_ids:
            rules_to_evaluate = [r for r in self.rules if r.id in rule_ids]

        for rule in rules_to_evaluate:
            result = self._evaluate_rule(rule, individual_data)
            if result is not None:
                fired_rules.append(result)

        return fired_rules

    def evaluate_batch(
        self,
        cohort_data: List[IndividualData],
        rule_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[FiredRule]]:
        """
        Evaluate rules against a cohort of individuals.

        Args:
            cohort_data: List of IndividualData for each individual
            rule_ids: Optional list of specific rules to evaluate

        Returns:
            Dictionary mapping sample_id to list of fired rules
        """
        results = {}
        for individual in cohort_data:
            fired = self.evaluate(individual, rule_ids)
            results[individual.sample_id] = fired
        return results

    def _evaluate_rule(
        self,
        rule: Rule,
        individual: IndividualData,
    ) -> Optional[FiredRule]:
        """
        Evaluate a single rule against individual data.

        Args:
            rule: The rule to evaluate
            individual: Individual's data

        Returns:
            FiredRule if rule fires, None otherwise
        """
        bindings: Dict[str, Any] = {}
        evidence: Dict[str, Any] = {}
        condition_results: List[ConditionResult] = []

        # Special handling for rules that iterate over genes
        genes_to_check = self._get_genes_to_check(rule, individual)

        if genes_to_check:
            # Evaluate rule for each candidate gene
            for gene in genes_to_check:
                gene_bindings = {"G": gene, "gene": gene}
                result = self._evaluate_conditions_for_gene(
                    rule, individual, gene, gene_bindings
                )
                if result is not None:
                    return result
        else:
            # Standard rule evaluation
            return self._evaluate_conditions(rule, individual, bindings)

        return None

    def _get_genes_to_check(
        self,
        rule: Rule,
        individual: IndividualData,
    ) -> List[str]:
        """Determine which genes to check for a rule."""
        # Check if rule has gene-specific conditions without a bound gene
        needs_gene_iteration = any(
            c.predicate in (
                "has_damaging_variant", "has_lof_variant",
                "is_constrained", "is_sfari_gene", "is_synaptic_gene",
                "prenatally_expressed", "is_chd8_target",
            ) and not c.arguments.get("gene")
            for c in rule.conditions
        )

        if needs_gene_iteration:
            # Start with genes that have damaging variants
            return list(individual.get_genes_with_damaging_variants())

        return []

    def _evaluate_conditions_for_gene(
        self,
        rule: Rule,
        individual: IndividualData,
        gene: str,
        bindings: Dict[str, Any],
    ) -> Optional[FiredRule]:
        """Evaluate rule conditions for a specific gene."""
        evidence = {"gene": gene}
        condition_explanations = []

        if rule.logic == "AND":
            # All conditions must be satisfied
            for condition in rule.conditions:
                # Inject gene into arguments if needed
                args = condition.arguments.copy()
                if "gene" not in args and condition.predicate in (
                    "has_damaging_variant", "has_lof_variant", "has_missense_variant",
                    "is_constrained", "gene_constraint", "is_sfari_gene",
                    "is_high_confidence_sfari", "is_synaptic_gene", "is_chd8_target",
                    "prenatally_expressed", "expressed_in", "cell_type_expression",
                    "has_paralog",
                ):
                    args["gene"] = gene

                modified_condition = Condition(
                    predicate=condition.predicate,
                    arguments=args,
                    negated=condition.negated,
                )

                result = self.condition_evaluator.evaluate(
                    modified_condition, individual, bindings
                )

                if not result.satisfied:
                    return None

                bindings.update(result.bound_variables)
                evidence.update(result.evidence)
                condition_explanations.append(result.explanation)

        elif rule.logic == "OR":
            # At least one condition must be satisfied
            any_satisfied = False
            for condition in rule.conditions:
                args = condition.arguments.copy()
                if "gene" not in args:
                    args["gene"] = gene

                modified_condition = Condition(
                    predicate=condition.predicate,
                    arguments=args,
                    negated=condition.negated,
                )

                result = self.condition_evaluator.evaluate(
                    modified_condition, individual, bindings
                )

                if result.satisfied:
                    any_satisfied = True
                    bindings.update(result.bound_variables)
                    evidence.update(result.evidence)
                    condition_explanations.append(result.explanation)
                    break

            if not any_satisfied:
                return None

        # Calculate confidence
        confidence = rule.base_confidence * rule.conclusion.confidence_modifier

        # Generate explanation
        explanation = self._generate_explanation(
            rule, bindings, evidence, condition_explanations
        )

        return FiredRule(
            rule=rule,
            bindings=bindings,
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
        )

    def _evaluate_conditions(
        self,
        rule: Rule,
        individual: IndividualData,
        bindings: Dict[str, Any],
    ) -> Optional[FiredRule]:
        """Evaluate rule conditions without gene iteration."""
        evidence = {}
        condition_explanations = []

        if rule.logic == "AND":
            for condition in rule.conditions:
                result = self.condition_evaluator.evaluate(
                    condition, individual, bindings
                )
                if not result.satisfied:
                    return None
                bindings.update(result.bound_variables)
                evidence.update(result.evidence)
                condition_explanations.append(result.explanation)

        elif rule.logic == "OR":
            any_satisfied = False
            for condition in rule.conditions:
                result = self.condition_evaluator.evaluate(
                    condition, individual, bindings
                )
                if result.satisfied:
                    any_satisfied = True
                    bindings.update(result.bound_variables)
                    evidence.update(result.evidence)
                    condition_explanations.append(result.explanation)
                    break

            if not any_satisfied:
                return None

        confidence = rule.base_confidence * rule.conclusion.confidence_modifier
        explanation = self._generate_explanation(
            rule, bindings, evidence, condition_explanations
        )

        return FiredRule(
            rule=rule,
            bindings=bindings,
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
        )

    def _generate_explanation(
        self,
        rule: Rule,
        bindings: Dict[str, Any],
        evidence: Dict[str, Any],
        condition_explanations: List[str],
    ) -> str:
        """Generate human-readable explanation for fired rule."""
        gene = bindings.get("G") or bindings.get("gene") or evidence.get("gene")

        parts = [f"Rule {rule.id} ({rule.name}) fired"]

        if gene:
            parts.append(f"for gene {gene}")

        parts.append(":")

        # Add condition explanations
        if condition_explanations:
            parts.append("\n  Conditions met:")
            for expl in condition_explanations:
                parts.append(f"\n    - {expl}")

        # Add conclusion
        parts.append(f"\n  Conclusion: {rule.conclusion.type}")
        for key, value in rule.conclusion.attributes.items():
            if key != "disclaimer":
                parts.append(f"\n    - {key}: {value}")

        return "".join(parts)

    def get_rules_by_type(
        self,
        conclusion_type: ConclusionType,
    ) -> List[Rule]:
        """Get rules that produce a specific conclusion type."""
        return [
            r for r in self.rules
            if r.conclusion.type == conclusion_type.value
        ]

    def explain(self, fired_rule: FiredRule) -> str:
        """
        Generate detailed explanation for a fired rule.

        Args:
            fired_rule: The fired rule to explain

        Returns:
            Detailed human-readable explanation
        """
        rule = fired_rule.rule
        parts = [
            f"=== {rule.name} (Rule {rule.id}) ===\n",
            f"Description: {rule.description}\n\n",
            f"Evidence supporting this inference:\n",
        ]

        for key, value in fired_rule.evidence.items():
            parts.append(f"  - {key}: {value}\n")

        parts.append(f"\nConclusion ({fired_rule.confidence:.0%} confidence):\n")
        parts.append(f"  Type: {rule.conclusion.type}\n")

        for key, value in rule.conclusion.attributes.items():
            parts.append(f"  {key}: {value}\n")

        if rule.evidence_sources:
            parts.append(f"\nEvidence sources:\n")
            for source in rule.evidence_sources:
                parts.append(f"  - {source}\n")

        return "".join(parts)

    def get_summary(
        self,
        fired_rules: List[FiredRule],
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for fired rules.

        Args:
            fired_rules: List of fired rules

        Returns:
            Summary dictionary
        """
        by_type = {}
        by_rule = {}
        genes_affected = set()
        total_confidence = 0

        for fr in fired_rules:
            # By conclusion type
            ctype = fr.rule.conclusion.type
            by_type[ctype] = by_type.get(ctype, 0) + 1

            # By rule ID
            by_rule[fr.rule.id] = by_rule.get(fr.rule.id, 0) + 1

            # Genes
            gene = fr.bindings.get("G") or fr.bindings.get("gene")
            if gene:
                genes_affected.add(gene)

            total_confidence += fr.confidence

        return {
            "total_fired": len(fired_rules),
            "by_conclusion_type": by_type,
            "by_rule": by_rule,
            "genes_affected": list(genes_affected),
            "average_confidence": total_confidence / len(fired_rules) if fired_rules else 0,
        }
