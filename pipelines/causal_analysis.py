"""
Causal Analysis Pipeline (Session 18)

Standalone causal reasoning pipeline for individual case analysis.

This pipeline provides:
- Structural causal model construction from genetic data
- Intervention queries (do-calculus)
- Counterfactual reasoning
- Mediation analysis for pathway effects

Example Usage:
    from pipelines import CausalAnalysisPipeline, CausalAnalysisConfig

    # Configure pipeline
    config = CausalAnalysisConfig(
        sample_id="PATIENT_001",
        variant_genes=["SHANK3", "CHD8"],
        disrupted_pathways=["synaptic_transmission", "chromatin_remodeling"],
    )

    pipeline = CausalAnalysisPipeline(config)
    result = pipeline.run()

    # View causal analysis
    print(result.summary)

    # Query intervention effects
    effect = pipeline.query_intervention(
        treatment="synaptic_transmission",
        outcome="asd_phenotype"
    )
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# Dynamic Module Imports
# =============================================================================

_project_root = Path(__file__).parent.parent
_modules_dir = _project_root / "modules"


def _import_module_from_path(module_name: str, dir_name: str):
    """Import a module from a directory with numeric prefix."""
    module_path = _modules_dir / dir_name / "__init__.py"
    if not module_path.exists():
        raise ImportError(f"Module not found: {module_path}")

    module_dir = str(_modules_dir / dir_name)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Module 12: Causal Inference
_mod_12 = _import_module_from_path("causal_inference", "12_causal_inference")
CausalNodeType = _mod_12.CausalNodeType
CausalEdgeType = _mod_12.CausalEdgeType
CausalNode = _mod_12.CausalNode
CausalEdge = _mod_12.CausalEdge
CausalQuery = _mod_12.CausalQuery
CausalQueryBuilder = _mod_12.CausalQueryBuilder
StructuralCausalModel = _mod_12.StructuralCausalModel
IntervenedModel = _mod_12.IntervenedModel
DoCalculusEngine = _mod_12.DoCalculusEngine
Distribution = _mod_12.Distribution
CounterfactualEngine = _mod_12.CounterfactualEngine
CounterfactualResult = _mod_12.CounterfactualResult
CausalEffectEstimator = _mod_12.CausalEffectEstimator
MediationResult = _mod_12.MediationResult
EffectDecomposition = _mod_12.EffectDecomposition
create_sample_asd_scm = _mod_12.create_sample_asd_scm

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CausalAnalysisConfig:
    """Configuration for causal analysis pipeline."""

    # Individual identification
    sample_id: str = "individual"

    # Genetic data
    variant_genes: List[str] = field(default_factory=list)
    disrupted_pathways: List[str] = field(default_factory=list)
    gene_effects: Dict[str, float] = field(default_factory=dict)  # Gene -> effect size
    pathway_effects: Dict[str, float] = field(default_factory=dict)  # Pathway -> effect size

    # Causal model
    causal_model_path: Optional[str] = None
    use_sample_model: bool = True
    build_from_data: bool = False

    # Analysis options
    run_intervention_analysis: bool = True
    run_counterfactual_analysis: bool = True
    run_mediation_analysis: bool = True

    # Intervention queries
    default_interventions: List[Dict[str, Any]] = field(default_factory=list)

    # Counterfactual queries
    counterfactual_scenarios: List[Dict[str, Any]] = field(default_factory=list)

    # Output
    verbose: bool = True


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class InterventionAnalysis:
    """Results of intervention (do-calculus) analysis."""
    interventions: Dict[str, float]  # Variable -> intervened value
    outcome: str
    effect: float
    distribution: Optional[Distribution] = None
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "interventions": self.interventions,
            "outcome": self.outcome,
            "effect": self.effect,
            "distribution": self.distribution.to_dict() if self.distribution else None,
            "explanation": self.explanation,
        }


@dataclass
class CounterfactualAnalysis:
    """Results of counterfactual analysis."""
    factual: Dict[str, float]  # Observed values
    counterfactual: Dict[str, float]  # What-if values
    outcome_variable: str
    factual_outcome: float
    counterfactual_outcome: float
    change: float
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "factual": self.factual,
            "counterfactual": self.counterfactual,
            "outcome_variable": self.outcome_variable,
            "factual_outcome": self.factual_outcome,
            "counterfactual_outcome": self.counterfactual_outcome,
            "change": self.change,
            "explanation": self.explanation,
        }


@dataclass
class PathwayMediationAnalysis:
    """Mediation analysis for a specific pathway."""
    gene: str
    pathway: str
    phenotype: str
    total_effect: float
    direct_effect: float
    indirect_effect: float
    proportion_mediated: float
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "gene": self.gene,
            "pathway": self.pathway,
            "phenotype": self.phenotype,
            "total_effect": self.total_effect,
            "direct_effect": self.direct_effect,
            "indirect_effect": self.indirect_effect,
            "proportion_mediated": self.proportion_mediated,
            "confidence_interval": self.confidence_interval,
            "explanation": self.explanation,
        }


@dataclass
class CausalAnalysisResult:
    """Complete results from causal analysis pipeline."""

    # Individual info
    sample_id: str
    variant_genes: List[str]
    disrupted_pathways: List[str]

    # Causal model info
    causal_model_nodes: List[str]
    causal_model_edges: int

    # Analysis results
    intervention_analyses: List[InterventionAnalysis]
    counterfactual_analyses: List[CounterfactualAnalysis]
    mediation_analyses: List[PathwayMediationAnalysis]

    # Computed insights
    key_causal_drivers: List[str]
    intervention_opportunities: List[Dict[str, Any]]

    # Metadata
    config: Optional[CausalAnalysisConfig] = None
    runtime_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def n_interventions(self) -> int:
        return len(self.intervention_analyses)

    @property
    def n_counterfactuals(self) -> int:
        return len(self.counterfactual_analyses)

    @property
    def n_mediations(self) -> int:
        return len(self.mediation_analyses)

    @property
    def summary(self) -> str:
        """Generate comprehensive summary."""
        lines = [
            "=" * 70,
            "CAUSAL ANALYSIS PIPELINE RESULTS",
            "=" * 70,
            f"Sample ID: {self.sample_id}",
            f"Timestamp: {self.timestamp}",
            f"Runtime: {self.runtime_seconds:.1f} seconds",
            "",
            "GENETIC DATA:",
            f"  Variant genes: {', '.join(self.variant_genes) if self.variant_genes else 'None'}",
            f"  Disrupted pathways: {', '.join(self.disrupted_pathways) if self.disrupted_pathways else 'None'}",
            "",
            "CAUSAL MODEL:",
            f"  Nodes: {len(self.causal_model_nodes)}",
            f"  Edges: {self.causal_model_edges}",
            "",
            "ANALYSIS RESULTS:",
            f"  Intervention analyses: {self.n_interventions}",
            f"  Counterfactual analyses: {self.n_counterfactuals}",
            f"  Mediation analyses: {self.n_mediations}",
        ]

        if self.key_causal_drivers:
            lines.extend([
                "",
                "KEY CAUSAL DRIVERS:",
            ])
            for driver in self.key_causal_drivers[:5]:
                lines.append(f"  - {driver}")

        if self.intervention_opportunities:
            lines.extend([
                "",
                "INTERVENTION OPPORTUNITIES:",
            ])
            for opp in self.intervention_opportunities[:3]:
                lines.append(f"  - {opp.get('description', 'Unknown')}")

        if self.mediation_analyses:
            lines.extend([
                "",
                "MEDIATION ANALYSIS:",
            ])
            for med in self.mediation_analyses[:3]:
                lines.append(
                    f"  - {med.gene} -> {med.pathway} -> {med.phenotype}: "
                    f"{med.proportion_mediated:.1%} mediated"
                )

        lines.extend([
            "",
            "=" * 70,
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sample_id": self.sample_id,
            "variant_genes": self.variant_genes,
            "disrupted_pathways": self.disrupted_pathways,
            "causal_model_nodes": self.causal_model_nodes,
            "causal_model_edges": self.causal_model_edges,
            "intervention_analyses": [a.to_dict() for a in self.intervention_analyses],
            "counterfactual_analyses": [a.to_dict() for a in self.counterfactual_analyses],
            "mediation_analyses": [a.to_dict() for a in self.mediation_analyses],
            "key_causal_drivers": self.key_causal_drivers,
            "intervention_opportunities": self.intervention_opportunities,
            "runtime_seconds": self.runtime_seconds,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Main Pipeline
# =============================================================================

class CausalAnalysisPipeline:
    """
    Standalone causal analysis pipeline for individual case analysis.

    Provides intervention queries, counterfactual reasoning, and
    mediation analysis for understanding genetic mechanisms.
    """

    def __init__(self, config: CausalAnalysisConfig):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self._setup_logging()

        # Lazy-loaded components
        self._scm: Optional[StructuralCausalModel] = None
        self._do_engine: Optional[DoCalculusEngine] = None
        self._cf_engine: Optional[CounterfactualEngine] = None
        self._effect_estimator: Optional[CausalEffectEstimator] = None

    def _setup_logging(self) -> None:
        """Configure logging."""
        level = logging.INFO if self.config.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    @property
    def scm(self) -> StructuralCausalModel:
        """Get or create the structural causal model."""
        if self._scm is None:
            self._scm = self._load_or_build_model()
        return self._scm

    @property
    def do_engine(self) -> DoCalculusEngine:
        """Get or create the do-calculus engine."""
        if self._do_engine is None:
            self._do_engine = DoCalculusEngine(self.scm)
        return self._do_engine

    @property
    def counterfactual_engine(self) -> CounterfactualEngine:
        """Get or create the counterfactual engine."""
        if self._cf_engine is None:
            self._cf_engine = CounterfactualEngine(self.scm)
        return self._cf_engine

    @property
    def effect_estimator(self) -> CausalEffectEstimator:
        """Get or create the effect estimator."""
        if self._effect_estimator is None:
            self._effect_estimator = CausalEffectEstimator(self.scm, self.do_engine)
        return self._effect_estimator

    def run(self) -> CausalAnalysisResult:
        """
        Execute the complete causal analysis pipeline.

        Returns:
            CausalAnalysisResult with all analysis outputs
        """
        start_time = datetime.now()
        logger.info(f"Starting causal analysis for {self.config.sample_id}")

        # Initialize model
        _ = self.scm  # Force model loading

        # Step 1: Intervention analysis
        intervention_analyses = []
        if self.config.run_intervention_analysis:
            logger.info("Step 1: Running intervention analysis")
            intervention_analyses = self._run_intervention_analysis()

        # Step 2: Counterfactual analysis
        counterfactual_analyses = []
        if self.config.run_counterfactual_analysis:
            logger.info("Step 2: Running counterfactual analysis")
            counterfactual_analyses = self._run_counterfactual_analysis()

        # Step 3: Mediation analysis
        mediation_analyses = []
        if self.config.run_mediation_analysis:
            logger.info("Step 3: Running mediation analysis")
            mediation_analyses = self._run_mediation_analysis()

        # Step 4: Identify key causal drivers
        logger.info("Step 4: Identifying key causal drivers")
        key_drivers = self._identify_key_drivers(
            intervention_analyses, mediation_analyses
        )

        # Step 5: Identify intervention opportunities
        logger.info("Step 5: Identifying intervention opportunities")
        intervention_opportunities = self._identify_intervention_opportunities(
            intervention_analyses, mediation_analyses
        )

        runtime = (datetime.now() - start_time).total_seconds()
        logger.info(f"Causal analysis completed in {runtime:.1f} seconds")

        return CausalAnalysisResult(
            sample_id=self.config.sample_id,
            variant_genes=self.config.variant_genes,
            disrupted_pathways=self.config.disrupted_pathways,
            causal_model_nodes=list(self.scm.nodes.keys()),
            causal_model_edges=len(self.scm.edges),
            intervention_analyses=intervention_analyses,
            counterfactual_analyses=counterfactual_analyses,
            mediation_analyses=mediation_analyses,
            key_causal_drivers=key_drivers,
            intervention_opportunities=intervention_opportunities,
            config=self.config,
            runtime_seconds=runtime,
        )

    # =========================================================================
    # Model Loading
    # =========================================================================

    def _load_or_build_model(self) -> StructuralCausalModel:
        """Load or build the structural causal model."""
        if self.config.causal_model_path:
            logger.info(f"Loading causal model from {self.config.causal_model_path}")
            return StructuralCausalModel.load(self.config.causal_model_path)

        if self.config.build_from_data:
            logger.info("Building causal model from genetic data")
            return self._build_model_from_data()

        if self.config.use_sample_model:
            logger.info("Using sample ASD causal model")
            scm = create_sample_asd_scm()
            # Add individual-specific nodes
            self._add_individual_nodes(scm)
            return scm

        raise ValueError(
            "No causal model configured. Set causal_model_path, "
            "build_from_data=True, or use_sample_model=True"
        )

    def _build_model_from_data(self) -> StructuralCausalModel:
        """Build causal model from individual's genetic data."""
        scm = StructuralCausalModel()

        # Add phenotype node
        scm.add_node(CausalNode(
            id="asd_phenotype",
            node_type=CausalNodeType.PHENOTYPE,
            observed=True,
            value=1.0,  # Individual has ASD
        ))

        # Add pathway nodes
        for pathway in self.config.disrupted_pathways:
            effect = self.config.pathway_effects.get(pathway, 0.5)
            scm.add_node(CausalNode(
                id=pathway,
                node_type=CausalNodeType.PATHWAY,
                observed=True,
                value=effect,
            ))
            # Connect pathway to phenotype
            scm.add_edge(CausalEdge(
                source=pathway,
                target="asd_phenotype",
                edge_type=CausalEdgeType.CAUSES,
                strength=effect,
                mechanism=f"{pathway} disruption contributes to ASD phenotype",
            ))

        # Add gene nodes
        for gene in self.config.variant_genes:
            effect = self.config.gene_effects.get(gene, 0.7)
            scm.add_node(CausalNode(
                id=f"{gene}_function",
                node_type=CausalNodeType.GENE_FUNCTION,
                observed=True,
                value=1.0 - effect,  # Disrupted function
            ))

            # Connect gene to relevant pathways
            for pathway in self.config.disrupted_pathways:
                scm.add_edge(CausalEdge(
                    source=f"{gene}_function",
                    target=pathway,
                    edge_type=CausalEdgeType.CAUSES,
                    strength=0.5,
                    mechanism=f"{gene} affects {pathway}",
                ))

        # Add confounders
        scm.add_node(CausalNode(
            id="ancestry",
            node_type=CausalNodeType.CONFOUNDER,
            observed=False,
        ))

        return scm

    def _add_individual_nodes(self, scm: StructuralCausalModel) -> None:
        """Add individual-specific nodes to the causal model."""
        # Add gene function nodes for variant genes
        for gene in self.config.variant_genes:
            node_id = f"{gene}_function"
            if node_id not in scm.nodes:
                effect = self.config.gene_effects.get(gene, 0.7)
                scm.add_node(CausalNode(
                    id=node_id,
                    node_type=CausalNodeType.GENE_FUNCTION,
                    observed=True,
                    value=1.0 - effect,
                ))

        # Add pathway nodes for disrupted pathways
        for pathway in self.config.disrupted_pathways:
            if pathway not in scm.nodes:
                effect = self.config.pathway_effects.get(pathway, 0.5)
                scm.add_node(CausalNode(
                    id=pathway,
                    node_type=CausalNodeType.PATHWAY,
                    observed=True,
                    value=effect,
                ))

    # =========================================================================
    # Intervention Analysis
    # =========================================================================

    def _run_intervention_analysis(self) -> List[InterventionAnalysis]:
        """Run intervention (do-calculus) analysis."""
        analyses = []

        # Default interventions based on disrupted pathways
        if not self.config.default_interventions:
            interventions = self._generate_default_interventions()
        else:
            interventions = self.config.default_interventions

        for intervention in interventions:
            try:
                analysis = self._analyze_intervention(
                    interventions=intervention.get("set", {}),
                    outcome=intervention.get("outcome", "asd_phenotype"),
                )
                analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Intervention analysis failed: {e}")

        return analyses

    def _generate_default_interventions(self) -> List[Dict[str, Any]]:
        """Generate default intervention queries."""
        interventions = []

        # For each disrupted pathway, query effect of normalizing it
        for pathway in self.config.disrupted_pathways:
            interventions.append({
                "set": {pathway: 0.0},  # Normalize pathway
                "outcome": "asd_phenotype",
                "description": f"Effect of normalizing {pathway}",
            })

        # For each variant gene, query effect of restoring function
        for gene in self.config.variant_genes:
            gene_node = f"{gene}_function"
            if gene_node in self.scm.nodes:
                interventions.append({
                    "set": {gene_node: 1.0},  # Restore full function
                    "outcome": "asd_phenotype",
                    "description": f"Effect of restoring {gene} function",
                })

        return interventions

    def _analyze_intervention(
        self,
        interventions: Dict[str, float],
        outcome: str,
    ) -> InterventionAnalysis:
        """Analyze a single intervention."""
        # Query the distribution under intervention
        distribution = self.do_engine.query(
            outcome=outcome,
            intervention=interventions,
        )

        # Calculate effect (compared to observational)
        observational = self.do_engine.query(
            outcome=outcome,
            intervention={},
        )

        effect = distribution.mean - observational.mean

        # Generate explanation
        intervention_str = ", ".join(
            f"do({var}={val})" for var, val in interventions.items()
        )
        explanation = (
            f"Under intervention {intervention_str}:\n"
            f"  Expected {outcome}: {distribution.mean:.3f}\n"
            f"  Observational {outcome}: {observational.mean:.3f}\n"
            f"  Effect: {effect:.3f}"
        )

        return InterventionAnalysis(
            interventions=interventions,
            outcome=outcome,
            effect=effect,
            distribution=distribution,
            explanation=explanation,
        )

    def query_intervention(
        self,
        treatment: str,
        outcome: str,
        treatment_value: float = 1.0,
    ) -> float:
        """
        Query the effect of a single intervention.

        Args:
            treatment: Variable to intervene on
            outcome: Outcome variable
            treatment_value: Value to set treatment to

        Returns:
            Average treatment effect
        """
        return self.do_engine.average_treatment_effect(
            treatment=treatment,
            outcome=outcome,
            treatment_values=(0.0, treatment_value),
        )

    # =========================================================================
    # Counterfactual Analysis
    # =========================================================================

    def _run_counterfactual_analysis(self) -> List[CounterfactualAnalysis]:
        """Run counterfactual analysis."""
        analyses = []

        # Default counterfactuals based on genetic data
        if not self.config.counterfactual_scenarios:
            scenarios = self._generate_default_counterfactuals()
        else:
            scenarios = self.config.counterfactual_scenarios

        for scenario in scenarios:
            try:
                analysis = self._analyze_counterfactual(
                    factual=scenario.get("factual", {}),
                    counterfactual=scenario.get("counterfactual", {}),
                    outcome=scenario.get("outcome", "asd_phenotype"),
                )
                analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Counterfactual analysis failed: {e}")

        return analyses

    def _generate_default_counterfactuals(self) -> List[Dict[str, Any]]:
        """Generate default counterfactual scenarios."""
        scenarios = []

        # "What if this gene wasn't mutated?"
        for gene in self.config.variant_genes:
            gene_node = f"{gene}_function"
            if gene_node in self.scm.nodes:
                scenarios.append({
                    "factual": {gene_node: 0.3},  # Current disrupted state
                    "counterfactual": {gene_node: 1.0},  # Full function
                    "outcome": "asd_phenotype",
                    "description": f"What if {gene} wasn't disrupted?",
                })

        return scenarios

    def _analyze_counterfactual(
        self,
        factual: Dict[str, float],
        counterfactual: Dict[str, float],
        outcome: str,
    ) -> CounterfactualAnalysis:
        """Analyze a single counterfactual scenario."""
        # Query counterfactual
        cf_result = self.counterfactual_engine.query(
            factual=factual,
            counterfactual=counterfactual,
            outcome=outcome,
        )

        # Generate explanation
        factual_str = ", ".join(f"{k}={v}" for k, v in factual.items())
        cf_str = ", ".join(f"{k}={v}" for k, v in counterfactual.items())
        explanation = (
            f"Factual world: {factual_str}\n"
            f"Counterfactual world: {cf_str}\n"
            f"Factual {outcome}: {cf_result.factual_outcome:.3f}\n"
            f"Counterfactual {outcome}: {cf_result.counterfactual_outcome:.3f}\n"
            f"Change: {cf_result.counterfactual_outcome - cf_result.factual_outcome:.3f}"
        )

        return CounterfactualAnalysis(
            factual=factual,
            counterfactual=counterfactual,
            outcome_variable=outcome,
            factual_outcome=cf_result.factual_outcome,
            counterfactual_outcome=cf_result.counterfactual_outcome,
            change=cf_result.counterfactual_outcome - cf_result.factual_outcome,
            explanation=explanation,
        )

    def query_counterfactual(
        self,
        factual: Dict[str, float],
        counterfactual: Dict[str, float],
        outcome: str,
    ) -> CounterfactualResult:
        """
        Query a counterfactual scenario.

        Args:
            factual: Observed values
            counterfactual: What-if values
            outcome: Outcome variable

        Returns:
            CounterfactualResult with analysis
        """
        return self.counterfactual_engine.query(
            factual=factual,
            counterfactual=counterfactual,
            outcome=outcome,
        )

    # =========================================================================
    # Mediation Analysis
    # =========================================================================

    def _run_mediation_analysis(self) -> List[PathwayMediationAnalysis]:
        """Run mediation analysis for gene -> pathway -> phenotype chains."""
        analyses = []

        # For each gene-pathway combination
        for gene in self.config.variant_genes:
            gene_node = f"{gene}_function"
            if gene_node not in self.scm.nodes:
                continue

            for pathway in self.config.disrupted_pathways:
                if pathway not in self.scm.nodes:
                    continue

                try:
                    analysis = self._analyze_mediation(
                        gene=gene,
                        gene_node=gene_node,
                        pathway=pathway,
                        phenotype="asd_phenotype",
                    )
                    analyses.append(analysis)
                except Exception as e:
                    logger.warning(
                        f"Mediation analysis failed for {gene} -> {pathway}: {e}"
                    )

        return analyses

    def _analyze_mediation(
        self,
        gene: str,
        gene_node: str,
        pathway: str,
        phenotype: str,
    ) -> PathwayMediationAnalysis:
        """Analyze mediation for a single gene-pathway-phenotype chain."""
        # Run mediation analysis
        mediation_result = self.effect_estimator.mediation_analysis(
            treatment=gene_node,
            outcome=phenotype,
            mediator=pathway,
        )

        # Generate explanation
        explanation = (
            f"Mediation analysis: {gene} -> {pathway} -> {phenotype}\n"
            f"  Total effect: {mediation_result.total_effect:.3f}\n"
            f"  Direct effect (not through {pathway}): {mediation_result.direct_effect:.3f}\n"
            f"  Indirect effect (through {pathway}): {mediation_result.indirect_effect:.3f}\n"
            f"  Proportion mediated: {mediation_result.proportion_mediated:.1%}"
        )

        return PathwayMediationAnalysis(
            gene=gene,
            pathway=pathway,
            phenotype=phenotype,
            total_effect=mediation_result.total_effect,
            direct_effect=mediation_result.direct_effect,
            indirect_effect=mediation_result.indirect_effect,
            proportion_mediated=mediation_result.proportion_mediated,
            confidence_interval=mediation_result.confidence_interval,
            explanation=explanation,
        )

    def query_mediation(
        self,
        treatment: str,
        outcome: str,
        mediator: str,
    ) -> MediationResult:
        """
        Query mediation analysis.

        Args:
            treatment: Treatment variable (e.g., gene function)
            outcome: Outcome variable (e.g., phenotype)
            mediator: Mediator variable (e.g., pathway)

        Returns:
            MediationResult with effect decomposition
        """
        return self.effect_estimator.mediation_analysis(
            treatment=treatment,
            outcome=outcome,
            mediator=mediator,
        )

    # =========================================================================
    # Insights
    # =========================================================================

    def _identify_key_drivers(
        self,
        intervention_analyses: List[InterventionAnalysis],
        mediation_analyses: List[PathwayMediationAnalysis],
    ) -> List[str]:
        """Identify key causal drivers from analyses."""
        drivers = []

        # From intervention analyses - largest effects
        intervention_effects = sorted(
            [(a.interventions, abs(a.effect)) for a in intervention_analyses],
            key=lambda x: x[1],
            reverse=True,
        )

        for interventions, effect in intervention_effects[:3]:
            var = list(interventions.keys())[0]
            drivers.append(f"{var} (intervention effect: {effect:.3f})")

        # From mediation - highest proportion mediated
        mediation_effects = sorted(
            mediation_analyses,
            key=lambda x: abs(x.indirect_effect),
            reverse=True,
        )

        for med in mediation_effects[:2]:
            drivers.append(
                f"{med.gene} -> {med.pathway} "
                f"({med.proportion_mediated:.1%} mediated)"
            )

        return drivers

    def _identify_intervention_opportunities(
        self,
        intervention_analyses: List[InterventionAnalysis],
        mediation_analyses: List[PathwayMediationAnalysis],
    ) -> List[Dict[str, Any]]:
        """Identify potential intervention opportunities."""
        opportunities = []

        # Pathways with large intervention effects
        for analysis in intervention_analyses:
            if abs(analysis.effect) > 0.2:
                target = list(analysis.interventions.keys())[0]
                opportunities.append({
                    "target": target,
                    "type": "pathway_normalization",
                    "expected_effect": analysis.effect,
                    "description": f"Targeting {target} could reduce phenotype by {abs(analysis.effect):.2f}",
                })

        # Pathways with high mediation proportion
        for med in mediation_analyses:
            if med.proportion_mediated > 0.3:
                opportunities.append({
                    "target": med.pathway,
                    "type": "pathway_modulation",
                    "gene": med.gene,
                    "proportion_mediated": med.proportion_mediated,
                    "description": (
                        f"Modulating {med.pathway} could address "
                        f"{med.proportion_mediated:.0%} of {med.gene}'s effect"
                    ),
                })

        return opportunities


# =============================================================================
# Convenience Functions
# =============================================================================

def run_causal_analysis(
    sample_id: str,
    variant_genes: List[str],
    disrupted_pathways: List[str],
    **kwargs: Any,
) -> CausalAnalysisResult:
    """
    Convenience function to run causal analysis.

    Args:
        sample_id: Individual identifier
        variant_genes: List of genes with variants
        disrupted_pathways: List of disrupted pathways
        **kwargs: Additional configuration options

    Returns:
        CausalAnalysisResult with all outputs
    """
    config = CausalAnalysisConfig(
        sample_id=sample_id,
        variant_genes=variant_genes,
        disrupted_pathways=disrupted_pathways,
        **kwargs,
    )

    pipeline = CausalAnalysisPipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run causal analysis on individual genetic data"
    )
    parser.add_argument("--sample-id", default="individual", help="Sample identifier")
    parser.add_argument("--genes", nargs="+", default=[], help="Variant genes")
    parser.add_argument("--pathways", nargs="+", default=[], help="Disrupted pathways")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    result = run_causal_analysis(
        sample_id=args.sample_id,
        variant_genes=args.genes,
        disrupted_pathways=args.pathways,
        verbose=args.verbose,
    )

    print(result.summary)
