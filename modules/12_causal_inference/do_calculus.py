"""
Pearl's do-calculus for intervention reasoning.

Enables queries like:
- P(phenotype | do(gene_disrupted))
- P(phenotype | do(pathway_targeted))
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import logging
import math

from causal_graph import StructuralCausalModel, CausalNode, CausalEdge, CausalEdgeType

logger = logging.getLogger(__name__)


@dataclass
class Distribution:
    """Represents a probability distribution over outcomes."""
    mean: float
    variance: float = 0.0
    samples: List[float] = field(default_factory=list)
    confidence_interval: Tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        # Clamp mean to [0, 1] for probability interpretation
        self.mean = max(0.0, min(1.0, self.mean))

    @property
    def std(self) -> float:
        """Standard deviation."""
        return math.sqrt(self.variance) if self.variance > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "mean": self.mean,
            "variance": self.variance,
            "std": self.std,
            "confidence_interval": self.confidence_interval,
        }


class IntervenedModel:
    """
    A model with interventions applied.

    When do(X=x) is applied:
    1. Set X to value x
    2. Remove all incoming edges to X
    3. Propagate effects through the graph
    """

    def __init__(
        self,
        original_scm: StructuralCausalModel,
        interventions: Dict[str, float]
    ):
        self.original_scm = original_scm
        self.interventions = interventions
        self.modified_scm = self._apply_interventions()

    def _apply_interventions(self) -> StructuralCausalModel:
        """Apply do-interventions to create modified SCM."""
        # Create a copy of the original model
        modified = self.original_scm.copy()

        # For each intervention, remove incoming edges and set value
        for node_id, value in self.interventions.items():
            if node_id not in modified.nodes:
                raise ValueError(f"Intervention node {node_id} not found")

            # Remove all incoming edges to the intervention node
            edges_to_remove = []
            for edge in modified.edges:
                if edge.target == node_id:
                    edges_to_remove.append((edge.source, edge.target))

            for source, target in edges_to_remove:
                modified.remove_edge(source, target)

            # Set the node value
            modified.nodes[node_id].value = value

        return modified

    def get_scm(self) -> StructuralCausalModel:
        """Get the modified SCM."""
        return self.modified_scm


class DoCalculusEngine:
    """
    Implements Pearl's do-calculus for intervention reasoning.

    Enables queries like:
    - P(phenotype | do(gene_disrupted))
    - P(phenotype | do(pathway_targeted))
    """

    def __init__(
        self,
        scm: StructuralCausalModel,
        default_noise_std: float = 0.1
    ):
        self.scm = scm
        self.default_noise_std = default_noise_std

    def do(self, intervention: Dict[str, float]) -> IntervenedModel:
        """
        Apply do-operator: set node values and remove incoming edges.

        Example:
            engine.do({"SHANK3_function": 0})  # Simulate SHANK3 knockout
        """
        return IntervenedModel(self.scm, intervention)

    def query(
        self,
        outcome: str,
        intervention: Dict[str, float],
        evidence: Optional[Dict[str, float]] = None,
        n_samples: int = 1000
    ) -> Distribution:
        """
        Compute P(outcome | do(intervention), evidence).

        Example:
            # What's the probability of ASD phenotype if we disrupt synaptic pathway?
            engine.query(
                outcome="asd_phenotype",
                intervention={"synaptic_pathway": 0}
            )
        """
        if outcome not in self.scm.nodes:
            raise ValueError(f"Outcome node {outcome} not found")

        # Apply intervention
        intervened = self.do(intervention)
        modified_scm = intervened.get_scm()

        # Propagate effects through the graph
        # Use topological order to compute values
        node_values = self._propagate_values(modified_scm, evidence)

        # Get outcome value
        outcome_value = node_values.get(outcome, 0.5)

        # Compute distribution with uncertainty
        variance = self._estimate_variance(modified_scm, outcome, node_values)

        return Distribution(
            mean=outcome_value,
            variance=variance,
            confidence_interval=(
                max(0, outcome_value - 1.96 * math.sqrt(variance)),
                min(1, outcome_value + 1.96 * math.sqrt(variance))
            )
        )

    def _propagate_values(
        self,
        scm: StructuralCausalModel,
        evidence: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Propagate values through the causal graph in topological order.

        Uses structural equations if available, otherwise uses linear
        combination of parent effects weighted by edge strengths.
        """
        values = {}
        evidence = evidence or {}

        # Get topological order
        order = self._topological_sort(scm)

        for node_id in order:
            node = scm.nodes[node_id]

            # Check if node has a fixed value (from intervention)
            if node.value is not None:
                values[node_id] = node.value
                continue

            # Check if we have evidence for this node
            if node_id in evidence:
                values[node_id] = evidence[node_id]
                continue

            # Use structural equation if available
            if node_id in scm.structural_equations:
                parent_values = {
                    parent: values.get(parent, 0.5)
                    for parent in scm.get_parents(node_id)
                }
                values[node_id] = scm.structural_equations[node_id](**parent_values)
                continue

            # Default: linear combination of parent effects
            parents = scm.get_parents(node_id)
            if not parents:
                # Root node with no value - assume baseline
                values[node_id] = 0.5
            else:
                # Combine parent effects weighted by edge strengths
                total_effect = 0.0
                total_weight = 0.0

                for parent in parents:
                    edge = scm.get_edge(parent, node_id)
                    if edge:
                        parent_value = values.get(parent, 0.5)
                        # Effect depends on parent value and edge strength
                        # Higher parent disruption (lower value) -> higher effect
                        if edge.edge_type == CausalEdgeType.CAUSES:
                            effect = parent_value * edge.strength
                        elif edge.edge_type == CausalEdgeType.CONFOUNDS:
                            effect = parent_value * edge.strength * 0.5  # Weaker
                        else:
                            effect = parent_value * edge.strength * 0.7
                        total_effect += effect
                        total_weight += edge.strength

                if total_weight > 0:
                    values[node_id] = total_effect / total_weight
                else:
                    values[node_id] = 0.5

        return values

    def _topological_sort(self, scm: StructuralCausalModel) -> List[str]:
        """Get nodes in topological order."""
        in_degree = {node_id: 0 for node_id in scm.nodes}

        for edge in scm.edges:
            in_degree[edge.target] += 1

        # Start with nodes that have no incoming edges
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            for child in scm.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # Handle cycles (shouldn't happen in DAG, but be safe)
        remaining = [n for n in scm.nodes if n not in order]
        order.extend(remaining)

        return order

    def _estimate_variance(
        self,
        scm: StructuralCausalModel,
        outcome: str,
        node_values: Dict[str, float]
    ) -> float:
        """
        Estimate variance in outcome due to model uncertainty.

        Uses path-based variance propagation.
        """
        base_variance = self.default_noise_std ** 2

        # Add variance from paths to outcome
        ancestors = scm.get_ancestors(outcome)

        for ancestor in ancestors:
            # Variance increases with path length and decreases with edge strength
            paths = scm._find_all_paths(ancestor, outcome, directed=True)
            for path in paths:
                path_strength = 1.0
                for i in range(len(path) - 1):
                    edge = scm.get_edge(path[i], path[i + 1])
                    if edge:
                        path_strength *= edge.strength

                # Lower strength -> more variance
                base_variance += (1 - path_strength) * 0.01

        return min(base_variance, 0.25)  # Cap variance

    def average_treatment_effect(
        self,
        treatment: str,
        outcome: str,
        treatment_values: Tuple[float, float] = (0.0, 1.0)
    ) -> float:
        """
        Compute Average Treatment Effect (ATE).

        ATE = E[Y | do(T=1)] - E[Y | do(T=0)]

        This measures the average causal effect of the treatment on the outcome.
        """
        if treatment not in self.scm.nodes:
            raise ValueError(f"Treatment node {treatment} not found")
        if outcome not in self.scm.nodes:
            raise ValueError(f"Outcome node {outcome} not found")

        # E[Y | do(T=t1)]
        dist_treated = self.query(
            outcome=outcome,
            intervention={treatment: treatment_values[1]}
        )

        # E[Y | do(T=t0)]
        dist_control = self.query(
            outcome=outcome,
            intervention={treatment: treatment_values[0]}
        )

        ate = dist_treated.mean - dist_control.mean

        logger.debug(
            f"ATE({treatment} -> {outcome}): "
            f"E[Y|do(T={treatment_values[1]})]={dist_treated.mean:.3f} - "
            f"E[Y|do(T={treatment_values[0]})]={dist_control.mean:.3f} = {ate:.3f}"
        )

        return ate

    def conditional_average_treatment_effect(
        self,
        treatment: str,
        outcome: str,
        subgroup: Dict[str, float],
        treatment_values: Tuple[float, float] = (0.0, 1.0)
    ) -> float:
        """
        Compute Conditional Average Treatment Effect (CATE) for a subgroup.

        CATE(subgroup) = E[Y | do(T=1), subgroup] - E[Y | do(T=0), subgroup]

        This measures the average causal effect within a specific subgroup.
        """
        if treatment not in self.scm.nodes:
            raise ValueError(f"Treatment node {treatment} not found")
        if outcome not in self.scm.nodes:
            raise ValueError(f"Outcome node {outcome} not found")

        # E[Y | do(T=t1), subgroup]
        dist_treated = self.query(
            outcome=outcome,
            intervention={treatment: treatment_values[1]},
            evidence=subgroup
        )

        # E[Y | do(T=t0), subgroup]
        dist_control = self.query(
            outcome=outcome,
            intervention={treatment: treatment_values[0]},
            evidence=subgroup
        )

        cate = dist_treated.mean - dist_control.mean

        logger.debug(
            f"CATE({treatment} -> {outcome} | subgroup): {cate:.3f}"
        )

        return cate

    def intervention_effect_on_path(
        self,
        intervention_node: str,
        outcome: str,
        intervention_value: float
    ) -> Dict[str, float]:
        """
        Compute the effect of an intervention on all nodes along paths to outcome.

        Returns a dictionary mapping each node to its expected value under intervention.
        """
        # Apply intervention
        intervened = self.do({intervention_node: intervention_value})
        modified_scm = intervened.get_scm()

        # Get values for all nodes
        values = self._propagate_values(modified_scm)

        # Filter to nodes on paths from intervention to outcome
        paths = modified_scm._find_all_paths(intervention_node, outcome, directed=True)
        nodes_on_paths = set()
        for path in paths:
            nodes_on_paths.update(path)

        return {node: values.get(node, 0.5) for node in nodes_on_paths}

    def identify_effect(
        self,
        treatment: str,
        outcome: str
    ) -> Optional[str]:
        """
        Check if the causal effect is identifiable and return adjustment formula.

        Returns a string describing how to identify the effect, or None if
        not identifiable.
        """
        # Check for backdoor paths
        backdoor_paths = self.scm.get_backdoor_paths(treatment, outcome)

        if not backdoor_paths:
            return f"P({outcome} | do({treatment})) = P({outcome} | {treatment})"

        # Check for valid adjustment sets
        adjustment_sets = self.scm.get_valid_adjustment_sets(treatment, outcome)

        if adjustment_sets:
            adj_set = adjustment_sets[0]
            if adj_set:
                adj_str = ", ".join(sorted(adj_set))
                return (
                    f"P({outcome} | do({treatment})) = "
                    f"Σ_{{adj}} P({outcome} | {treatment}, {adj_str}) P({adj_str})"
                )
            else:
                return f"P({outcome} | do({treatment})) = P({outcome} | {treatment})"

        # Check if treatment is directly connected to outcome
        if self.scm.get_edge(treatment, outcome):
            return f"Direct effect: P({outcome} | do({treatment})) identifiable"

        return None  # Effect not identifiable

    def sensitivity_analysis(
        self,
        treatment: str,
        outcome: str,
        confounding_strength_range: Tuple[float, float] = (0.0, 0.5)
    ) -> Dict[str, float]:
        """
        Perform sensitivity analysis for unmeasured confounding.

        Returns ATE estimates under different confounding assumptions.
        """
        results = {}
        base_ate = self.average_treatment_effect(treatment, outcome)
        results["base_ate"] = base_ate

        # Test different confounding strengths
        for strength in [0.1, 0.2, 0.3, 0.4, 0.5]:
            if confounding_strength_range[0] <= strength <= confounding_strength_range[1]:
                # Adjust ATE for potential unmeasured confounding
                adjusted_ate = base_ate * (1 - strength)
                results[f"ate_confounding_{strength}"] = adjusted_ate

        return results
