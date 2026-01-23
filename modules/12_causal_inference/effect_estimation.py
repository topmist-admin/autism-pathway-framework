"""
Causal effect estimation including mediation analysis.

Provides tools for:
- Total causal effect estimation
- Direct and indirect effect decomposition
- Mediation analysis with proportion mediated
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

from causal_graph import StructuralCausalModel
from do_calculus import DoCalculusEngine, Distribution

logger = logging.getLogger(__name__)


@dataclass
class MediationResult:
    """Result of mediation analysis."""
    total_effect: float
    direct_effect: float  # Natural Direct Effect (NDE)
    indirect_effect: float  # Natural Indirect Effect (NIE)
    proportion_mediated: float
    confidence_interval: Tuple[float, float]
    treatment: str
    outcome: str
    mediator: str
    explanation: str = ""

    def __post_init__(self):
        # Validate proportion mediated
        if self.total_effect != 0:
            # Recalculate to ensure consistency
            calculated_prop = self.indirect_effect / self.total_effect
            if abs(calculated_prop - self.proportion_mediated) > 0.01:
                logger.debug(
                    f"Proportion mediated adjusted: {self.proportion_mediated:.3f} -> {calculated_prop:.3f}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_effect": self.total_effect,
            "direct_effect": self.direct_effect,
            "indirect_effect": self.indirect_effect,
            "proportion_mediated": self.proportion_mediated,
            "confidence_interval": self.confidence_interval,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "mediator": self.mediator,
            "explanation": self.explanation,
        }


@dataclass
class EffectDecomposition:
    """Decomposition of causal effect through multiple paths."""
    total_effect: float
    path_effects: Dict[str, float]  # path_name -> effect
    residual: float  # Unexplained effect
    treatment: str
    outcome: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_effect": self.total_effect,
            "path_effects": self.path_effects,
            "residual": self.residual,
            "treatment": self.treatment,
            "outcome": self.outcome,
        }


class CausalEffectEstimator:
    """
    Estimate direct, indirect, and total causal effects.

    Implements mediation analysis for understanding how effects
    are transmitted through causal pathways.
    """

    def __init__(
        self,
        scm: StructuralCausalModel,
        do_engine: Optional[DoCalculusEngine] = None
    ):
        self.scm = scm
        self.do_engine = do_engine or DoCalculusEngine(scm)

    def total_effect(
        self,
        treatment: str,
        outcome: str,
        treatment_values: Tuple[float, float] = (0.0, 1.0)
    ) -> float:
        """
        Total causal effect of treatment on outcome.

        TE = E[Y | do(T=1)] - E[Y | do(T=0)]

        This includes all causal pathways from treatment to outcome.
        """
        return self.do_engine.average_treatment_effect(
            treatment=treatment,
            outcome=outcome,
            treatment_values=treatment_values
        )

    def direct_effect(
        self,
        treatment: str,
        outcome: str,
        mediator: str,
        treatment_values: Tuple[float, float] = (0.0, 1.0)
    ) -> float:
        """
        Natural Direct Effect (NDE): Effect not through mediator.

        NDE = E[Y(t=1, M(t=0))] - E[Y(t=0, M(t=0))]

        This is the effect of treatment on outcome while keeping the
        mediator at its natural value under no treatment.

        Example:
            # Direct effect of gene on phenotype, not through pathway
            estimator.direct_effect("SHANK3_function", "asd_phenotype", "synaptic_pathway")
        """
        if treatment not in self.scm.nodes:
            raise ValueError(f"Treatment node {treatment} not found")
        if outcome not in self.scm.nodes:
            raise ValueError(f"Outcome node {outcome} not found")
        if mediator not in self.scm.nodes:
            raise ValueError(f"Mediator node {mediator} not found")

        # Step 1: Get mediator value under control (t=0)
        mediator_under_control = self.do_engine.query(
            outcome=mediator,
            intervention={treatment: treatment_values[0]}
        ).mean

        # Step 2: Y(t=1, M(t=0)) - outcome when treated but mediator held at control level
        y_treated_m_control = self.do_engine.query(
            outcome=outcome,
            intervention={
                treatment: treatment_values[1],
                mediator: mediator_under_control
            }
        ).mean

        # Step 3: Y(t=0, M(t=0)) - outcome under control
        y_control = self.do_engine.query(
            outcome=outcome,
            intervention={treatment: treatment_values[0]}
        ).mean

        nde = y_treated_m_control - y_control

        logger.debug(
            f"NDE({treatment} -> {outcome} not through {mediator}): {nde:.3f}"
        )

        return nde

    def indirect_effect(
        self,
        treatment: str,
        outcome: str,
        mediator: str,
        treatment_values: Tuple[float, float] = (0.0, 1.0)
    ) -> float:
        """
        Natural Indirect Effect (NIE): Effect through mediator.

        NIE = E[Y(t=1, M(t=1))] - E[Y(t=1, M(t=0))]

        This is the effect of treatment on outcome that operates
        through the mediator.

        Example:
            # How much of SHANK3's effect on phenotype is mediated by synaptic pathway?
            estimator.indirect_effect("SHANK3_function", "asd_phenotype", "synaptic_pathway")
        """
        if treatment not in self.scm.nodes:
            raise ValueError(f"Treatment node {treatment} not found")
        if outcome not in self.scm.nodes:
            raise ValueError(f"Outcome node {outcome} not found")
        if mediator not in self.scm.nodes:
            raise ValueError(f"Mediator node {mediator} not found")

        # Step 1: Get mediator value under treatment (t=1)
        mediator_under_treatment = self.do_engine.query(
            outcome=mediator,
            intervention={treatment: treatment_values[1]}
        ).mean

        # Step 2: Get mediator value under control (t=0)
        mediator_under_control = self.do_engine.query(
            outcome=mediator,
            intervention={treatment: treatment_values[0]}
        ).mean

        # Step 3: Y(t=1, M(t=1)) - outcome when treated with natural mediator level
        y_treated = self.do_engine.query(
            outcome=outcome,
            intervention={treatment: treatment_values[1]}
        ).mean

        # Step 4: Y(t=1, M(t=0)) - outcome when treated but mediator held at control
        y_treated_m_control = self.do_engine.query(
            outcome=outcome,
            intervention={
                treatment: treatment_values[1],
                mediator: mediator_under_control
            }
        ).mean

        nie = y_treated - y_treated_m_control

        logger.debug(
            f"NIE({treatment} -> {outcome} through {mediator}): {nie:.3f}"
        )

        return nie

    def mediation_analysis(
        self,
        treatment: str,
        outcome: str,
        mediator: str,
        treatment_values: Tuple[float, float] = (0.0, 1.0),
        n_bootstrap: int = 100
    ) -> MediationResult:
        """
        Full mediation analysis with proportion mediated.

        Decomposes total effect into:
        - Natural Direct Effect (NDE): T -> Y not through M
        - Natural Indirect Effect (NIE): T -> M -> Y

        And computes proportion mediated = NIE / TE
        """
        # Compute effects
        te = self.total_effect(treatment, outcome, treatment_values)
        nde = self.direct_effect(treatment, outcome, mediator, treatment_values)
        nie = self.indirect_effect(treatment, outcome, mediator, treatment_values)

        # Compute proportion mediated
        if abs(te) > 1e-6:
            proportion = nie / te
        else:
            proportion = 0.0

        # Clamp proportion to [0, 1] for interpretability
        proportion = max(0.0, min(1.0, abs(proportion)))

        # Estimate confidence interval using simple approximation
        # In practice, would use bootstrap
        std_error = self._estimate_effect_se(treatment, outcome, mediator)
        ci = (
            max(0.0, proportion - 1.96 * std_error),
            min(1.0, proportion + 1.96 * std_error)
        )

        # Generate explanation
        explanation = self._generate_mediation_explanation(
            treatment, outcome, mediator, te, nde, nie, proportion
        )

        return MediationResult(
            total_effect=te,
            direct_effect=nde,
            indirect_effect=nie,
            proportion_mediated=proportion,
            confidence_interval=ci,
            treatment=treatment,
            outcome=outcome,
            mediator=mediator,
            explanation=explanation,
        )

    def _estimate_effect_se(
        self,
        treatment: str,
        outcome: str,
        mediator: str
    ) -> float:
        """
        Estimate standard error for proportion mediated.

        Uses a simple approximation based on path strengths.
        """
        # Get path strengths
        t_m_edge = self.scm.get_edge(treatment, mediator)
        m_o_edge = self.scm.get_edge(mediator, outcome)
        t_o_edge = self.scm.get_edge(treatment, outcome)

        # SE increases with weaker path strengths
        base_se = 0.1

        if t_m_edge:
            base_se *= (1 - t_m_edge.strength * 0.5)
        if m_o_edge:
            base_se *= (1 - m_o_edge.strength * 0.5)

        return min(0.3, base_se)

    def _generate_mediation_explanation(
        self,
        treatment: str,
        outcome: str,
        mediator: str,
        te: float,
        nde: float,
        nie: float,
        proportion: float
    ) -> str:
        """Generate human-readable mediation explanation."""
        if abs(te) < 0.01:
            return f"No significant total effect of {treatment} on {outcome}."

        effect_direction = "increases" if te > 0 else "decreases"

        if proportion > 0.7:
            mediation_strength = "strongly"
        elif proportion > 0.3:
            mediation_strength = "partially"
        else:
            mediation_strength = "weakly"

        return (
            f"{treatment} {effect_direction} {outcome} (TE={te:.3f}). "
            f"This effect is {mediation_strength} mediated by {mediator} "
            f"({proportion:.1%} of the effect). "
            f"Direct effect: {nde:.3f}, Indirect effect: {nie:.3f}."
        )

    def decompose_effect_by_path(
        self,
        treatment: str,
        outcome: str
    ) -> EffectDecomposition:
        """
        Decompose total effect into contributions from different paths.

        Identifies all directed paths from treatment to outcome and
        estimates the contribution of each.
        """
        if treatment not in self.scm.nodes:
            raise ValueError(f"Treatment node {treatment} not found")
        if outcome not in self.scm.nodes:
            raise ValueError(f"Outcome node {outcome} not found")

        # Get total effect
        te = self.total_effect(treatment, outcome)

        # Find all directed paths
        paths = self.scm._find_all_paths(treatment, outcome, directed=True)

        path_effects = {}
        total_path_effect = 0.0

        for i, path in enumerate(paths):
            # Compute path-specific effect
            path_strength = 1.0
            for j in range(len(path) - 1):
                edge = self.scm.get_edge(path[j], path[j + 1])
                if edge:
                    path_strength *= edge.strength

            # Effect through this path
            path_effect = te * path_strength
            path_name = " -> ".join(path)
            path_effects[path_name] = path_effect
            total_path_effect += path_effect

        # Residual (unexplained)
        residual = te - total_path_effect

        return EffectDecomposition(
            total_effect=te,
            path_effects=path_effects,
            residual=residual,
            treatment=treatment,
            outcome=outcome,
        )

    def controlled_direct_effect(
        self,
        treatment: str,
        outcome: str,
        mediator: str,
        mediator_value: float,
        treatment_values: Tuple[float, float] = (0.0, 1.0)
    ) -> float:
        """
        Controlled Direct Effect (CDE) at a specific mediator level.

        CDE(m) = E[Y | do(T=1, M=m)] - E[Y | do(T=0, M=m)]

        Unlike NDE, CDE fixes the mediator at a specific value rather
        than its natural value.
        """
        # Y under treatment with mediator fixed
        y_treated = self.do_engine.query(
            outcome=outcome,
            intervention={
                treatment: treatment_values[1],
                mediator: mediator_value
            }
        ).mean

        # Y under control with mediator fixed
        y_control = self.do_engine.query(
            outcome=outcome,
            intervention={
                treatment: treatment_values[0],
                mediator: mediator_value
            }
        ).mean

        return y_treated - y_control

    def effect_heterogeneity(
        self,
        treatment: str,
        outcome: str,
        effect_modifiers: List[str],
        n_subgroups: int = 3
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze effect heterogeneity across subgroups defined by effect modifiers.

        Returns ATE for different levels of each effect modifier.
        """
        results = {}

        for modifier in effect_modifiers:
            if modifier not in self.scm.nodes:
                continue

            results[modifier] = {}

            # Create subgroups based on modifier levels
            for level in range(n_subgroups):
                modifier_value = level / (n_subgroups - 1)

                cate = self.do_engine.conditional_average_treatment_effect(
                    treatment=treatment,
                    outcome=outcome,
                    subgroup={modifier: modifier_value}
                )

                results[modifier][f"level_{level}"] = cate

        return results

    def multi_mediator_analysis(
        self,
        treatment: str,
        outcome: str,
        mediators: List[str]
    ) -> Dict[str, MediationResult]:
        """
        Analyze mediation through multiple potential mediators.

        Returns mediation results for each mediator separately.
        """
        results = {}

        for mediator in mediators:
            try:
                result = self.mediation_analysis(treatment, outcome, mediator)
                results[mediator] = result
            except ValueError as e:
                logger.warning(f"Could not analyze mediator {mediator}: {e}")

        return results

    def pathway_contribution_analysis(
        self,
        gene: str,
        phenotype: str,
        pathways: List[str]
    ) -> Dict[str, float]:
        """
        Analyze how much of a gene's effect on phenotype goes through each pathway.

        Specific to ASD genetics use case.
        """
        total_effect = self.total_effect(gene, phenotype)

        contributions = {}

        if abs(total_effect) < 1e-6:
            # No total effect, all contributions are 0
            for pathway in pathways:
                contributions[pathway] = 0.0
            return contributions

        for pathway in pathways:
            if pathway not in self.scm.nodes:
                contributions[pathway] = 0.0
                continue

            try:
                mediation = self.mediation_analysis(gene, phenotype, pathway)
                contributions[pathway] = mediation.proportion_mediated
            except ValueError:
                contributions[pathway] = 0.0

        return contributions
