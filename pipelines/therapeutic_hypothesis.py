"""
Therapeutic Hypothesis Pipeline (Session 18)

End-to-end pipeline from genetic data to causally-validated therapeutic hypotheses.

This pipeline extends SubtypeDiscoveryPipeline with:
- Module 09: Symbolic rules (R1-R7) for mechanism reasoning
- Module 11: Therapeutic hypothesis generation and ranking
- Module 12: Causal validation of drug targets

IMPORTANT: All hypotheses generated are RESEARCH HYPOTHESES only.
They require clinical validation before any therapeutic consideration.

Example Usage:
    from pipelines import TherapeuticHypothesisPipeline, TherapeuticPipelineConfig

    config = TherapeuticPipelineConfig(
        data=DataConfig(
            vcf_path="cohort.vcf.gz",
            pathway_gmt_path="reactome.gmt",
        ),
        therapeutic=TherapeuticConfig(
            enable_rules=True,
            enable_causal_validation=True,
        ),
    )

    pipeline = TherapeuticHypothesisPipeline(config)
    result = pipeline.run()

    # View hypotheses
    for hyp in result.ranking_result.top_hypotheses:
        print(hyp.summary())
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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


# Import from subtype_discovery (Session 17)
from pipelines.subtype_discovery import (
    SubtypeDiscoveryPipeline,
    SubtypeDiscoveryResult,
    PipelineConfig,
    DataConfig,
    ProcessingConfig,
    PathwayScoringConfig,
    ClusteringPipelineConfig,
    PathwayDatabase,
    PathwayScoreMatrix,
    GeneBurdenMatrix,
    ClusteringResult,
    SubtypeProfile,
)

# Module 09: Symbolic Rules
_mod_09 = _import_module_from_path("symbolic_rules", "09_symbolic_rules")
RuleEngine = _mod_09.RuleEngine
BiologicalRules = _mod_09.BiologicalRules
BiologicalContext = _mod_09.BiologicalContext
IndividualData = _mod_09.IndividualData
FiredRule = _mod_09.FiredRule
ReasoningChain = _mod_09.ReasoningChain
ExplanationGenerator = _mod_09.ExplanationGenerator

# Module 11: Therapeutic Hypotheses
_mod_11 = _import_module_from_path("therapeutic_hypotheses", "11_therapeutic_hypotheses")
DrugTargetDatabase = _mod_11.DrugTargetDatabase
PathwayDrugMapper = _mod_11.PathwayDrugMapper
PathwayDrugMapperConfig = _mod_11.PathwayDrugMapperConfig
HypothesisRanker = _mod_11.HypothesisRanker
RankingConfig = _mod_11.RankingConfig
RankingResult = _mod_11.RankingResult
TherapeuticHypothesis = _mod_11.TherapeuticHypothesis
EvidenceScorer = _mod_11.EvidenceScorer
create_sample_drug_database = _mod_11.create_sample_drug_database

# Module 12: Causal Inference
_mod_12 = _import_module_from_path("causal_inference", "12_causal_inference")
StructuralCausalModel = _mod_12.StructuralCausalModel
DoCalculusEngine = _mod_12.DoCalculusEngine
CounterfactualEngine = _mod_12.CounterfactualEngine
CausalEffectEstimator = _mod_12.CausalEffectEstimator
MediationResult = _mod_12.MediationResult
create_sample_asd_scm = _mod_12.create_sample_asd_scm

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TherapeuticConfig:
    """Configuration for therapeutic hypothesis generation."""

    # Symbolic rules
    enable_rules: bool = True
    rules_to_use: Optional[List[str]] = None  # None = all rules

    # Drug database
    drug_database_path: Optional[str] = None
    use_sample_database: bool = True  # Use built-in sample database

    # Hypothesis ranking
    min_pathway_zscore: float = 1.5
    max_hypotheses: int = 50
    min_evidence_score: float = 0.2

    # Causal validation
    enable_causal_validation: bool = True
    causal_model_path: Optional[str] = None
    use_sample_causal_model: bool = True

    # Evidence weighting
    weight_evidence: float = 0.4
    weight_pathway_score: float = 0.3
    weight_drug_relevance: float = 0.2
    weight_mechanism_match: float = 0.1


@dataclass
class TherapeuticPipelineConfig:
    """Complete configuration for therapeutic hypothesis pipeline."""

    data: DataConfig
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    pathway_scoring: PathwayScoringConfig = field(default_factory=PathwayScoringConfig)
    clustering: ClusteringPipelineConfig = field(default_factory=ClusteringPipelineConfig)
    therapeutic: TherapeuticConfig = field(default_factory=TherapeuticConfig)

    verbose: bool = True
    random_state: int = 42
    output_dir: Optional[str] = None
    save_intermediate: bool = False


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class CausalValidation:
    """Causal validation results for a hypothesis."""
    hypothesis_id: str
    intervention_effect: float
    mediation_result: Optional[MediationResult] = None
    causal_confidence: float = 0.0
    is_causally_supported: bool = False
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "intervention_effect": self.intervention_effect,
            "mediation_result": self.mediation_result.to_dict() if self.mediation_result else None,
            "causal_confidence": self.causal_confidence,
            "is_causally_supported": self.is_causally_supported,
            "explanation": self.explanation,
        }


@dataclass
class IndividualAnalysis:
    """Analysis results for a single individual."""
    sample_id: str
    fired_rules: List[FiredRule]
    reasoning_chain: Optional[ReasoningChain] = None
    subtype_assignment: Optional[int] = None
    disrupted_pathways: List[str] = field(default_factory=list)
    hypotheses: List[TherapeuticHypothesis] = field(default_factory=list)
    causal_validations: List[CausalValidation] = field(default_factory=list)

    @property
    def n_rules_fired(self) -> int:
        return len(self.fired_rules)

    @property
    def n_hypotheses(self) -> int:
        return len(self.hypotheses)

    def summary(self) -> str:
        """Generate summary for this individual."""
        lines = [
            f"Individual: {self.sample_id}",
            f"  Subtype: {self.subtype_assignment}",
            f"  Rules fired: {self.n_rules_fired}",
            f"  Disrupted pathways: {len(self.disrupted_pathways)}",
            f"  Therapeutic hypotheses: {self.n_hypotheses}",
        ]
        if self.hypotheses:
            lines.append("  Top hypothesis: " + self.hypotheses[0].summary())
        return "\n".join(lines)


@dataclass
class TherapeuticHypothesisResult:
    """Complete results from therapeutic hypothesis pipeline."""

    # Subtype discovery results (from Session 17)
    subtype_result: SubtypeDiscoveryResult

    # Symbolic reasoning results
    individual_analyses: Dict[str, IndividualAnalysis]
    all_fired_rules: List[FiredRule]

    # Therapeutic hypothesis results
    ranking_result: RankingResult
    causal_validations: List[CausalValidation]

    # Aggregated data
    pathway_hypotheses_map: Dict[str, List[TherapeuticHypothesis]]
    subtype_hypotheses_map: Dict[int, List[TherapeuticHypothesis]]

    # Metadata
    config: Optional[TherapeuticPipelineConfig] = None
    runtime_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def n_individuals(self) -> int:
        return len(self.individual_analyses)

    @property
    def n_hypotheses(self) -> int:
        return len(self.ranking_result.hypotheses)

    @property
    def n_causally_validated(self) -> int:
        return sum(1 for v in self.causal_validations if v.is_causally_supported)

    @property
    def summary(self) -> str:
        """Generate comprehensive summary."""
        lines = [
            "=" * 70,
            "THERAPEUTIC HYPOTHESIS PIPELINE RESULTS",
            "=" * 70,
            f"Timestamp: {self.timestamp}",
            f"Runtime: {self.runtime_seconds:.1f} seconds",
            "",
            "SUBTYPE DISCOVERY:",
            f"  Samples analyzed: {self.n_individuals}",
            f"  Subtypes identified: {self.subtype_result.n_subtypes}",
            f"  Pathways scored: {len(self.subtype_result.pathway_scores.pathways)}",
            "",
            "SYMBOLIC REASONING:",
            f"  Total rules fired: {len(self.all_fired_rules)}",
            f"  Unique rules: {len(set(r.rule.id for r in self.all_fired_rules))}",
            "",
            "THERAPEUTIC HYPOTHESES:",
            f"  Hypotheses generated: {self.n_hypotheses}",
            f"  Pathways with hypotheses: {len(self.ranking_result.pathways_with_hypotheses)}",
            f"  High evidence count: {self.ranking_result.high_evidence_count}",
            "",
            "CAUSAL VALIDATION:",
            f"  Hypotheses validated: {len(self.causal_validations)}",
            f"  Causally supported: {self.n_causally_validated}",
            "",
            "TOP 5 HYPOTHESES:",
        ]

        for hyp in self.ranking_result.top_hypotheses[:5]:
            lines.append(f"  {hyp.summary()}")

        lines.extend([
            "",
            "=" * 70,
            "NOTE: All hypotheses require clinical validation.",
            "=" * 70,
        ])

        return "\n".join(lines)


# =============================================================================
# Main Pipeline
# =============================================================================

class TherapeuticHypothesisPipeline:
    """
    End-to-end pipeline for therapeutic hypothesis generation.

    Extends SubtypeDiscoveryPipeline with symbolic reasoning (R1-R7),
    therapeutic hypothesis ranking, and causal validation.
    """

    def __init__(self, config: TherapeuticPipelineConfig):
        """
        Initialize the pipeline.

        Args:
            config: Complete pipeline configuration
        """
        self.config = config
        self._setup_logging()

        # Convert to base PipelineConfig for subtype discovery
        self._base_config = PipelineConfig(
            data=config.data,
            processing=config.processing,
            pathway_scoring=config.pathway_scoring,
            clustering=config.clustering,
            verbose=config.verbose,
            random_state=config.random_state,
            output_dir=config.output_dir,
            save_intermediate=config.save_intermediate,
        )

        # Lazy-loaded components
        self._subtype_pipeline: Optional[SubtypeDiscoveryPipeline] = None
        self._rule_engine: Optional[RuleEngine] = None
        self._hypothesis_ranker: Optional[HypothesisRanker] = None
        self._causal_model: Optional[StructuralCausalModel] = None
        self._do_engine: Optional[DoCalculusEngine] = None
        self._effect_estimator: Optional[CausalEffectEstimator] = None

    def _setup_logging(self) -> None:
        """Configure logging."""
        level = logging.INFO if self.config.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def run(
        self,
        vcf_path: Optional[str] = None,
        biological_context: Optional[BiologicalContext] = None,
    ) -> TherapeuticHypothesisResult:
        """
        Execute the complete therapeutic hypothesis pipeline.

        Args:
            vcf_path: Optional override for VCF path
            biological_context: Optional biological context for rule evaluation

        Returns:
            TherapeuticHypothesisResult with all outputs
        """
        start_time = datetime.now()
        logger.info("Starting therapeutic hypothesis pipeline")

        # Step 1: Run subtype discovery (Session 17)
        logger.info("Step 1: Running subtype discovery")
        subtype_result = self._run_subtype_discovery(vcf_path)

        # Step 2: Apply symbolic rules
        individual_analyses = {}
        all_fired_rules = []

        if self.config.therapeutic.enable_rules:
            logger.info("Step 2: Applying symbolic rules")
            individual_analyses, all_fired_rules = self._apply_symbolic_rules(
                subtype_result, biological_context
            )
        else:
            logger.info("Step 2: Symbolic rules disabled, skipping")

        # Step 3: Generate therapeutic hypotheses
        logger.info("Step 3: Generating therapeutic hypotheses")
        ranking_result = self._generate_hypotheses(
            subtype_result, all_fired_rules
        )

        # Step 4: Causal validation
        causal_validations = []
        if self.config.therapeutic.enable_causal_validation:
            logger.info("Step 4: Running causal validation")
            causal_validations = self._validate_causally(
                ranking_result, subtype_result
            )
        else:
            logger.info("Step 4: Causal validation disabled, skipping")

        # Step 5: Aggregate results
        logger.info("Step 5: Aggregating results")
        pathway_hypotheses_map = self._group_by_pathway(ranking_result.hypotheses)
        subtype_hypotheses_map = self._group_by_subtype(
            ranking_result.hypotheses,
            subtype_result,
            individual_analyses,
        )

        runtime = (datetime.now() - start_time).total_seconds()
        logger.info(f"Pipeline completed in {runtime:.1f} seconds")

        return TherapeuticHypothesisResult(
            subtype_result=subtype_result,
            individual_analyses=individual_analyses,
            all_fired_rules=all_fired_rules,
            ranking_result=ranking_result,
            causal_validations=causal_validations,
            pathway_hypotheses_map=pathway_hypotheses_map,
            subtype_hypotheses_map=subtype_hypotheses_map,
            config=self.config,
            runtime_seconds=runtime,
        )

    # =========================================================================
    # Step 1: Subtype Discovery
    # =========================================================================

    def _run_subtype_discovery(
        self,
        vcf_path: Optional[str] = None
    ) -> SubtypeDiscoveryResult:
        """Run subtype discovery pipeline."""
        self._subtype_pipeline = SubtypeDiscoveryPipeline(self._base_config)
        return self._subtype_pipeline.run(vcf_path=vcf_path)

    # =========================================================================
    # Step 2: Symbolic Rules
    # =========================================================================

    def _apply_symbolic_rules(
        self,
        subtype_result: SubtypeDiscoveryResult,
        biological_context: Optional[BiologicalContext] = None,
    ) -> tuple[Dict[str, IndividualAnalysis], List[FiredRule]]:
        """Apply biological rules to each individual."""
        # Create biological context if not provided
        if biological_context is None:
            biological_context = self._create_biological_context(subtype_result)

        # Get rules to use
        rules = BiologicalRules.get_all_rules()
        if self.config.therapeutic.rules_to_use:
            rules = [r for r in rules if r.id in self.config.therapeutic.rules_to_use]

        # Initialize rule engine
        self._rule_engine = RuleEngine(rules, biological_context)

        # Initialize explanation generator
        explanation_generator = ExplanationGenerator()

        individual_analyses = {}
        all_fired_rules = []

        # Process each individual
        samples = subtype_result.pathway_scores.samples
        for i, sample_id in enumerate(samples):
            # Create individual data
            individual_data = self._create_individual_data(
                sample_id, i, subtype_result
            )

            # Evaluate rules
            fired_rules = self._rule_engine.evaluate(individual_data)
            all_fired_rules.extend(fired_rules)

            # Generate reasoning chain
            reasoning_chain = None
            if fired_rules:
                reasoning_chain = explanation_generator.generate_reasoning_chain(
                    sample_id, fired_rules
                )

            # Get disrupted pathways (z-score > threshold)
            pathway_scores = subtype_result.pathway_scores
            disrupted_pathways = [
                pathway_scores.pathways[j]
                for j in range(len(pathway_scores.pathways))
                if pathway_scores.scores[i, j] > self.config.therapeutic.min_pathway_zscore
            ]

            # Get subtype assignment
            subtype = int(subtype_result.clustering_result.labels[i])

            individual_analyses[sample_id] = IndividualAnalysis(
                sample_id=sample_id,
                fired_rules=fired_rules,
                reasoning_chain=reasoning_chain,
                subtype_assignment=subtype,
                disrupted_pathways=disrupted_pathways,
            )

            logger.debug(
                f"Individual {sample_id}: {len(fired_rules)} rules fired, "
                f"{len(disrupted_pathways)} disrupted pathways"
            )

        logger.info(
            f"Applied rules to {len(samples)} individuals, "
            f"{len(all_fired_rules)} total rules fired"
        )

        return individual_analyses, all_fired_rules

    def _create_biological_context(
        self,
        subtype_result: SubtypeDiscoveryResult,
    ) -> BiologicalContext:
        """Create biological context from subtype results."""
        # Create a minimal biological context
        # In production, this would load actual data
        return BiologicalContext(
            pathway_db=self._subtype_pipeline._pathway_db if self._subtype_pipeline else None,
            gene_constraints=self._subtype_pipeline._gene_constraints if self._subtype_pipeline else None,
        )

    def _create_individual_data(
        self,
        sample_id: str,
        sample_idx: int,
        subtype_result: SubtypeDiscoveryResult,
    ) -> IndividualData:
        """Create individual data for rule evaluation."""
        # Extract pathway scores for this individual
        pathway_scores = {
            subtype_result.pathway_scores.pathways[j]: subtype_result.pathway_scores.scores[sample_idx, j]
            for j in range(len(subtype_result.pathway_scores.pathways))
        }

        # Extract gene burdens for this individual
        gene_burdens = {
            subtype_result.gene_burdens.genes[j]: subtype_result.gene_burdens.scores[sample_idx, j]
            for j in range(len(subtype_result.gene_burdens.genes))
        }

        return IndividualData(
            sample_id=sample_id,
            pathway_scores=pathway_scores,
            gene_burdens=gene_burdens,
        )

    # =========================================================================
    # Step 3: Therapeutic Hypothesis Generation
    # =========================================================================

    def _generate_hypotheses(
        self,
        subtype_result: SubtypeDiscoveryResult,
        fired_rules: List[FiredRule],
    ) -> RankingResult:
        """Generate and rank therapeutic hypotheses."""
        # Initialize drug database
        drug_db = self._load_drug_database()

        # Initialize mapper and scorer
        drug_mapper = PathwayDrugMapper(
            drug_db,
            config=PathwayDrugMapperConfig(),
        )
        evidence_scorer = EvidenceScorer()

        # Initialize ranker
        ranking_config = RankingConfig(
            weight_evidence=self.config.therapeutic.weight_evidence,
            weight_pathway_score=self.config.therapeutic.weight_pathway_score,
            weight_drug_relevance=self.config.therapeutic.weight_drug_relevance,
            weight_mechanism_match=self.config.therapeutic.weight_mechanism_match,
            min_pathway_zscore=self.config.therapeutic.min_pathway_zscore,
            max_hypotheses=self.config.therapeutic.max_hypotheses,
            min_evidence_score=self.config.therapeutic.min_evidence_score,
        )

        self._hypothesis_ranker = HypothesisRanker(
            drug_mapper=drug_mapper,
            evidence_scorer=evidence_scorer,
            config=ranking_config,
        )

        # Aggregate pathway scores across cohort (mean z-score)
        cohort_pathway_scores = {}
        for j, pathway in enumerate(subtype_result.pathway_scores.pathways):
            mean_score = np.mean(subtype_result.pathway_scores.scores[:, j])
            cohort_pathway_scores[pathway] = mean_score

        # Get pathway genes mapping
        pathway_genes = {}
        if self._subtype_pipeline and self._subtype_pipeline._pathway_db:
            for pathway_id in subtype_result.pathway_scores.pathways:
                genes = self._subtype_pipeline._pathway_db.pathways.get(pathway_id, set())
                pathway_genes[pathway_id] = list(genes)

        # Rank hypotheses
        ranking_result = self._hypothesis_ranker.rank(
            pathway_scores=cohort_pathway_scores,
            pathway_genes=pathway_genes,
            fired_rules=fired_rules,
        )

        logger.info(
            f"Generated {len(ranking_result.hypotheses)} hypotheses from "
            f"{len(ranking_result.pathways_analyzed)} pathways"
        )

        return ranking_result

    def _load_drug_database(self) -> DrugTargetDatabase:
        """Load drug target database."""
        if self.config.therapeutic.drug_database_path:
            # Load from file
            return DrugTargetDatabase.load(self.config.therapeutic.drug_database_path)
        elif self.config.therapeutic.use_sample_database:
            # Use built-in sample database
            return create_sample_drug_database()
        else:
            raise ValueError(
                "No drug database configured. Set drug_database_path or "
                "enable use_sample_database"
            )

    # =========================================================================
    # Step 4: Causal Validation
    # =========================================================================

    def _validate_causally(
        self,
        ranking_result: RankingResult,
        subtype_result: SubtypeDiscoveryResult,
    ) -> List[CausalValidation]:
        """Validate hypotheses using causal inference."""
        # Initialize causal model
        self._causal_model = self._load_causal_model()
        self._do_engine = DoCalculusEngine(self._causal_model)
        self._effect_estimator = CausalEffectEstimator(
            self._causal_model, self._do_engine
        )

        validations = []

        for hypothesis in ranking_result.hypotheses[:20]:  # Validate top 20
            try:
                validation = self._validate_single_hypothesis(hypothesis)
                validations.append(validation)
            except Exception as e:
                logger.warning(f"Failed to validate hypothesis {hypothesis.drug_id}: {e}")
                validations.append(CausalValidation(
                    hypothesis_id=hypothesis.drug_id,
                    intervention_effect=0.0,
                    causal_confidence=0.0,
                    is_causally_supported=False,
                    explanation=f"Validation failed: {str(e)}",
                ))

        n_supported = sum(1 for v in validations if v.is_causally_supported)
        logger.info(
            f"Causal validation: {n_supported}/{len(validations)} hypotheses supported"
        )

        return validations

    def _load_causal_model(self) -> StructuralCausalModel:
        """Load or create structural causal model."""
        if self.config.therapeutic.causal_model_path:
            # Load from file
            return StructuralCausalModel.load(self.config.therapeutic.causal_model_path)
        elif self.config.therapeutic.use_sample_causal_model:
            # Use built-in sample model
            return create_sample_asd_scm()
        else:
            raise ValueError(
                "No causal model configured. Set causal_model_path or "
                "enable use_sample_causal_model"
            )

    def _validate_single_hypothesis(
        self,
        hypothesis: TherapeuticHypothesis,
    ) -> CausalValidation:
        """Validate a single hypothesis causally."""
        # Query intervention effect
        # "What's the effect of targeting this pathway?"
        try:
            intervention_effect = self._do_engine.average_treatment_effect(
                treatment=hypothesis.target_pathway,
                outcome="asd_phenotype",
            )
        except (ValueError, KeyError):
            # Node not in model, use default
            intervention_effect = 0.0

        # Mediation analysis if we have a specific gene target
        mediation_result = None
        if hypothesis.target_genes:
            primary_gene = hypothesis.target_genes[0]
            try:
                mediation_result = self._effect_estimator.mediation_analysis(
                    treatment=primary_gene,
                    outcome="asd_phenotype",
                    mediator=hypothesis.target_pathway,
                )
            except (ValueError, KeyError):
                # Can't perform mediation, nodes not in model
                pass

        # Calculate causal confidence
        causal_confidence = self._calculate_causal_confidence(
            intervention_effect, mediation_result
        )

        # Determine if causally supported
        is_supported = (
            abs(intervention_effect) > 0.1 and
            causal_confidence > 0.5
        )

        # Generate explanation
        explanation = self._generate_causal_explanation(
            hypothesis, intervention_effect, mediation_result
        )

        return CausalValidation(
            hypothesis_id=hypothesis.drug_id,
            intervention_effect=intervention_effect,
            mediation_result=mediation_result,
            causal_confidence=causal_confidence,
            is_causally_supported=is_supported,
            explanation=explanation,
        )

    def _calculate_causal_confidence(
        self,
        intervention_effect: float,
        mediation_result: Optional[MediationResult],
    ) -> float:
        """Calculate confidence in causal support."""
        # Base confidence from intervention effect magnitude
        base_confidence = min(abs(intervention_effect), 1.0)

        # Boost if mediation analysis supports the pathway
        if mediation_result and mediation_result.proportion_mediated > 0.3:
            base_confidence *= 1.2

        return min(base_confidence, 1.0)

    def _generate_causal_explanation(
        self,
        hypothesis: TherapeuticHypothesis,
        intervention_effect: float,
        mediation_result: Optional[MediationResult],
    ) -> str:
        """Generate explanation for causal validation."""
        lines = [
            f"Causal validation for {hypothesis.drug_name}:",
            f"  Target pathway: {hypothesis.target_pathway}",
            f"  Intervention effect: {intervention_effect:.3f}",
        ]

        if mediation_result:
            lines.extend([
                f"  Total effect: {mediation_result.total_effect:.3f}",
                f"  Direct effect: {mediation_result.direct_effect:.3f}",
                f"  Indirect effect: {mediation_result.indirect_effect:.3f}",
                f"  Proportion mediated: {mediation_result.proportion_mediated:.1%}",
            ])

        return "\n".join(lines)

    # =========================================================================
    # Step 5: Aggregation
    # =========================================================================

    def _group_by_pathway(
        self,
        hypotheses: List[TherapeuticHypothesis],
    ) -> Dict[str, List[TherapeuticHypothesis]]:
        """Group hypotheses by target pathway."""
        pathway_map: Dict[str, List[TherapeuticHypothesis]] = {}
        for hyp in hypotheses:
            if hyp.target_pathway not in pathway_map:
                pathway_map[hyp.target_pathway] = []
            pathway_map[hyp.target_pathway].append(hyp)
        return pathway_map

    def _group_by_subtype(
        self,
        hypotheses: List[TherapeuticHypothesis],
        subtype_result: SubtypeDiscoveryResult,
        individual_analyses: Dict[str, IndividualAnalysis],
    ) -> Dict[int, List[TherapeuticHypothesis]]:
        """Group hypotheses by relevant subtype."""
        subtype_map: Dict[int, List[TherapeuticHypothesis]] = {}

        # For each subtype, find hypotheses targeting its characteristic pathways
        for profile in subtype_result.subtype_profiles:
            subtype_id = profile.subtype_id
            subtype_map[subtype_id] = []

            # Get top pathways for this subtype
            top_pathway_ids = {sig.pathway_id for sig in profile.top_pathways}

            # Find hypotheses targeting these pathways
            for hyp in hypotheses:
                if hyp.target_pathway in top_pathway_ids:
                    subtype_map[subtype_id].append(hyp)

        return subtype_map


# =============================================================================
# Convenience Function
# =============================================================================

def run_therapeutic_hypothesis(
    vcf_path: str,
    pathway_path: str,
    output_dir: Optional[str] = None,
    enable_rules: bool = True,
    enable_causal: bool = True,
    **kwargs: Any,
) -> TherapeuticHypothesisResult:
    """
    Convenience function to run therapeutic hypothesis pipeline.

    Args:
        vcf_path: Path to VCF file
        pathway_path: Path to pathway GMT file
        output_dir: Optional output directory
        enable_rules: Enable symbolic rules (default True)
        enable_causal: Enable causal validation (default True)
        **kwargs: Additional configuration options

    Returns:
        TherapeuticHypothesisResult with all outputs
    """
    config = TherapeuticPipelineConfig(
        data=DataConfig(
            vcf_path=vcf_path,
            pathway_gmt_path=pathway_path,
        ),
        therapeutic=TherapeuticConfig(
            enable_rules=enable_rules,
            enable_causal_validation=enable_causal,
        ),
        output_dir=output_dir,
        save_intermediate=output_dir is not None,
    )

    pipeline = TherapeuticHypothesisPipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate therapeutic hypotheses from genetic data"
    )
    parser.add_argument("--vcf", required=True, help="Path to VCF file")
    parser.add_argument("--pathways", required=True, help="Path to pathway GMT file")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--no-rules", action="store_true", help="Disable symbolic rules")
    parser.add_argument("--no-causal", action="store_true", help="Disable causal validation")

    args = parser.parse_args()

    result = run_therapeutic_hypothesis(
        vcf_path=args.vcf,
        pathway_path=args.pathways,
        output_dir=args.output,
        enable_rules=not args.no_rules,
        enable_causal=not args.no_causal,
    )

    print(result.summary)
