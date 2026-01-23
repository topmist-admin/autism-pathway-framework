"""
Counterfactual reasoning engine.

Enables queries like:
- "Would phenotype differ if pathway X were intact?"
- "What if this individual had a different variant?"

Implements the three-step counterfactual algorithm:
1. Abduction: Infer exogenous variables from factual evidence
2. Action: Apply counterfactual intervention
3. Prediction: Compute query variable under modified model
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

from causal_graph import StructuralCausalModel, CausalEdgeType
from do_calculus import DoCalculusEngine, Distribution

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualResult:
    """Result of a counterfactual query."""
    factual_value: float  # Observed/factual value
    counterfactual_value: float  # Value under counterfactual
    difference: float  # counterfactual - factual
    confidence: float  # Confidence in the estimate [0, 1]
    exogenous_values: Dict[str, float]  # Inferred exogenous variables
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "factual_value": self.factual_value,
            "counterfactual_value": self.counterfactual_value,
            "difference": self.difference,
            "confidence": self.confidence,
            "exogenous_values": self.exogenous_values,
            "explanation": self.explanation,
        }


class CounterfactualEngine:
    """
    Enables counterfactual reasoning.

    Queries like:
    - "Would phenotype differ if pathway X were intact?"
    - "What if this individual had a different variant?"
    """

    def __init__(
        self,
        scm: StructuralCausalModel,
        do_engine: Optional[DoCalculusEngine] = None
    ):
        self.scm = scm
        self.do_engine = do_engine or DoCalculusEngine(scm)

    def counterfactual(
        self,
        factual_evidence: Dict[str, float],
        counterfactual_intervention: Dict[str, float],
        query_variable: str
    ) -> CounterfactualResult:
        """
        Three-step counterfactual computation:
        1. Abduction: Infer exogenous variables from factual evidence
        2. Action: Apply counterfactual intervention
        3. Prediction: Compute query variable under modified model

        Example:
            # For an individual with SHANK3 mutation and ASD diagnosis,
            # what would phenotype be if SHANK3 were intact?
            engine.counterfactual(
                factual_evidence={"SHANK3_function": 0, "asd_phenotype": 1},
                counterfactual_intervention={"SHANK3_function": 1},
                query_variable="asd_phenotype"
            )
        """
        if query_variable not in self.scm.nodes:
            raise ValueError(f"Query variable {query_variable} not found")

        # Step 1: Abduction - infer exogenous variables
        exogenous = self._abduction(factual_evidence)

        # Step 2: Action - apply counterfactual intervention
        modified_scm = self._action(exogenous, counterfactual_intervention)

        # Step 3: Prediction - compute query under modified model
        counterfactual_value = self._prediction(modified_scm, query_variable, exogenous)

        # Get factual value
        factual_value = factual_evidence.get(query_variable, 0.5)

        # Compute confidence based on evidence quality
        confidence = self._compute_confidence(factual_evidence, counterfactual_intervention)

        # Generate explanation
        explanation = self._generate_explanation(
            factual_evidence,
            counterfactual_intervention,
            query_variable,
            factual_value,
            counterfactual_value
        )

        return CounterfactualResult(
            factual_value=factual_value,
            counterfactual_value=counterfactual_value,
            difference=counterfactual_value - factual_value,
            confidence=confidence,
            exogenous_values=exogenous,
            explanation=explanation,
        )

    def _abduction(self, evidence: Dict[str, float]) -> Dict[str, float]:
        """
        Abduction step: Infer exogenous (noise) variables from observed evidence.

        For each observed node, infer what the exogenous contribution must be
        to explain the observed value given the structural equations.
        """
        exogenous = {}

        # Process nodes in topological order
        order = self._topological_sort()

        for node_id in order:
            if node_id in evidence:
                observed = evidence[node_id]

                # Compute expected value from parents
                expected = self._compute_expected_value(node_id, evidence)

                # Exogenous = observed - expected (the residual)
                exogenous[node_id] = observed - expected
            else:
                # For unobserved nodes, assume exogenous = 0
                exogenous[node_id] = 0.0

        return exogenous

    def _action(
        self,
        exogenous: Dict[str, float],
        intervention: Dict[str, float]
    ) -> StructuralCausalModel:
        """
        Action step: Apply counterfactual intervention.

        Create a modified SCM where:
        1. Intervention nodes have their values fixed
        2. Exogenous variables are preserved from factual world
        """
        # Create a copy of the SCM
        modified = self.scm.copy()

        # Apply intervention (remove incoming edges, fix values)
        for node_id, value in intervention.items():
            if node_id not in modified.nodes:
                raise ValueError(f"Intervention node {node_id} not found")

            # Remove incoming edges
            edges_to_remove = [
                (e.source, e.target) for e in modified.edges
                if e.target == node_id
            ]
            for source, target in edges_to_remove:
                modified.remove_edge(source, target)

            # Fix the value
            modified.nodes[node_id].value = value

        return modified

    def _prediction(
        self,
        modified_scm: StructuralCausalModel,
        query_variable: str,
        exogenous: Dict[str, float]
    ) -> float:
        """
        Prediction step: Compute query variable under modified model.

        Uses the exogenous variables from abduction and the modified
        structural equations from action.
        """
        values = {}
        order = self._topological_sort(modified_scm)

        for node_id in order:
            node = modified_scm.nodes[node_id]

            # If value is fixed (from intervention), use it
            if node.value is not None:
                values[node_id] = node.value
                continue

            # Compute from parents + exogenous
            expected = self._compute_expected_value(node_id, values, modified_scm)
            exogenous_contribution = exogenous.get(node_id, 0.0)

            # Counterfactual value = expected + exogenous
            value = expected + exogenous_contribution

            # Clamp to [0, 1]
            values[node_id] = max(0.0, min(1.0, value))

        return values.get(query_variable, 0.5)

    def _compute_expected_value(
        self,
        node_id: str,
        values: Dict[str, float],
        scm: Optional[StructuralCausalModel] = None
    ) -> float:
        """Compute expected value of a node given parent values."""
        scm = scm or self.scm
        parents = scm.get_parents(node_id)

        if not parents:
            return 0.5  # Root node default

        # Check for structural equation
        if node_id in scm.structural_equations:
            parent_values = {p: values.get(p, 0.5) for p in parents}
            return scm.structural_equations[node_id](**parent_values)

        # Default: weighted average of parent effects
        total_effect = 0.0
        total_weight = 0.0

        for parent in parents:
            edge = scm.get_edge(parent, node_id)
            if edge:
                parent_value = values.get(parent, 0.5)
                effect = parent_value * edge.strength
                total_effect += effect
                total_weight += edge.strength

        if total_weight > 0:
            return total_effect / total_weight

        return 0.5

    def _topological_sort(
        self,
        scm: Optional[StructuralCausalModel] = None
    ) -> List[str]:
        """Get nodes in topological order."""
        scm = scm or self.scm
        in_degree = {node_id: 0 for node_id in scm.nodes}

        for edge in scm.edges:
            in_degree[edge.target] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in scm.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # Add any remaining nodes (for cycles)
        remaining = [n for n in scm.nodes if n not in order]
        order.extend(remaining)

        return order

    def _compute_confidence(
        self,
        evidence: Dict[str, float],
        intervention: Dict[str, float]
    ) -> float:
        """
        Compute confidence in counterfactual estimate.

        Higher confidence when:
        - More evidence is provided
        - Evidence is close to intervention targets
        - Shorter causal paths
        """
        # Base confidence
        confidence = 0.5

        # Boost for more evidence
        evidence_boost = min(0.2, len(evidence) * 0.05)
        confidence += evidence_boost

        # Check if intervention targets are related to evidence
        for int_node in intervention:
            for ev_node in evidence:
                # Check if there's a path
                if int_node in self.scm.get_ancestors(ev_node):
                    confidence += 0.1
                    break
                if ev_node in self.scm.get_ancestors(int_node):
                    confidence += 0.1
                    break

        return min(0.95, confidence)

    def _generate_explanation(
        self,
        evidence: Dict[str, float],
        intervention: Dict[str, float],
        query: str,
        factual: float,
        counterfactual: float
    ) -> str:
        """Generate human-readable explanation."""
        int_desc = ", ".join(f"{k}={v:.2f}" for k, v in intervention.items())
        ev_desc = ", ".join(f"{k}={v:.2f}" for k, v in evidence.items())

        diff = counterfactual - factual
        direction = "increase" if diff > 0 else "decrease" if diff < 0 else "remain unchanged"

        return (
            f"Given factual evidence ({ev_desc}), if we had set {int_desc}, "
            f"then {query} would {direction} from {factual:.2f} to {counterfactual:.2f} "
            f"(difference: {diff:+.2f})."
        )

    def probability_of_necessity(
        self,
        treatment: str,
        outcome: str,
        factual: Dict[str, float]
    ) -> float:
        """
        Probability of Necessity (PN).

        P(Y_0 = 0 | T = 1, Y = 1)
        "Given that treatment happened and outcome occurred,
         would outcome not have occurred without treatment?"

        This is the probability that the treatment was necessary for the outcome.
        """
        # Validate that factual shows treatment=1 and outcome=1
        treatment_value = factual.get(treatment, 0)
        outcome_value = factual.get(outcome, 0)

        if treatment_value < 0.5 or outcome_value < 0.5:
            logger.warning(
                "PN typically assumes T=1 and Y=1 in factual. "
                f"Got T={treatment_value}, Y={outcome_value}"
            )

        # Compute counterfactual: what would outcome be if treatment=0?
        cf_result = self.counterfactual(
            factual_evidence=factual,
            counterfactual_intervention={treatment: 0.0},
            query_variable=outcome
        )

        # PN = P(Y_0 = 0) = 1 - P(Y_0 = 1)
        # Y_0 is the counterfactual outcome under no treatment
        pn = 1.0 - cf_result.counterfactual_value

        return max(0.0, min(1.0, pn))

    def probability_of_sufficiency(
        self,
        treatment: str,
        outcome: str,
        factual: Dict[str, float]
    ) -> float:
        """
        Probability of Sufficiency (PS).

        P(Y_1 = 1 | T = 0, Y = 0)
        "Given that treatment didn't happen and outcome didn't occur,
         would outcome have occurred with treatment?"

        This is the probability that the treatment would be sufficient
        to cause the outcome.
        """
        # Validate that factual shows treatment=0 and outcome=0
        treatment_value = factual.get(treatment, 1)
        outcome_value = factual.get(outcome, 1)

        if treatment_value > 0.5 or outcome_value > 0.5:
            logger.warning(
                "PS typically assumes T=0 and Y=0 in factual. "
                f"Got T={treatment_value}, Y={outcome_value}"
            )

        # Compute counterfactual: what would outcome be if treatment=1?
        cf_result = self.counterfactual(
            factual_evidence=factual,
            counterfactual_intervention={treatment: 1.0},
            query_variable=outcome
        )

        # PS = P(Y_1 = 1)
        ps = cf_result.counterfactual_value

        return max(0.0, min(1.0, ps))

    def probability_of_necessity_and_sufficiency(
        self,
        treatment: str,
        outcome: str
    ) -> float:
        """
        Probability of Necessity and Sufficiency (PNS).

        P(Y_1 = 1, Y_0 = 0)
        "Would the outcome occur with treatment AND not occur without treatment?"

        This is the probability that the treatment is both necessary and sufficient.
        """
        # Compute using bounds from PN and PS
        # In general SCMs: PNS = P(Y_1=1) - P(Y_0=1) (when certain monotonicity holds)

        # Compute E[Y | do(T=1)]
        dist_treated = self.do_engine.query(
            outcome=outcome,
            intervention={treatment: 1.0}
        )

        # Compute E[Y | do(T=0)]
        dist_control = self.do_engine.query(
            outcome=outcome,
            intervention={treatment: 0.0}
        )

        # PNS approximation (under monotonicity)
        pns = dist_treated.mean - dist_control.mean

        return max(0.0, min(1.0, pns))

    def what_if_analysis(
        self,
        factual_evidence: Dict[str, float],
        interventions: List[Dict[str, float]],
        query_variable: str
    ) -> List[CounterfactualResult]:
        """
        Compare multiple counterfactual scenarios.

        Useful for asking "which intervention would have the biggest effect?"
        """
        results = []

        for intervention in interventions:
            result = self.counterfactual(
                factual_evidence=factual_evidence,
                counterfactual_intervention=intervention,
                query_variable=query_variable
            )
            results.append(result)

        # Sort by absolute difference (largest effect first)
        results.sort(key=lambda r: abs(r.difference), reverse=True)

        return results

    def individual_treatment_effect(
        self,
        treatment: str,
        outcome: str,
        individual_evidence: Dict[str, float]
    ) -> float:
        """
        Compute Individual Treatment Effect (ITE) for a specific individual.

        ITE = Y_1 - Y_0
        where Y_1 is outcome with treatment, Y_0 is outcome without treatment,
        both computed for this specific individual.
        """
        # Counterfactual: outcome if treated
        cf_treated = self.counterfactual(
            factual_evidence=individual_evidence,
            counterfactual_intervention={treatment: 1.0},
            query_variable=outcome
        )

        # Counterfactual: outcome if not treated
        cf_untreated = self.counterfactual(
            factual_evidence=individual_evidence,
            counterfactual_intervention={treatment: 0.0},
            query_variable=outcome
        )

        ite = cf_treated.counterfactual_value - cf_untreated.counterfactual_value

        return ite
