"""
End-to-end pipelines for autism pathway analysis.

This module provides complete workflows that integrate multiple modules
to perform complex analyses:

Session 17:
- SubtypeDiscoveryPipeline: From VCF to pathway-based autism subtypes

Session 18:
- TherapeuticHypothesisPipeline: Subtype discovery + rules + therapeutic hypotheses
- CausalAnalysisPipeline: Standalone causal reasoning on individual cases

Example Usage:
    from pipelines import SubtypeDiscoveryPipeline, PipelineConfig, DataConfig

    # Session 17: Subtype Discovery
    config = PipelineConfig(
        data=DataConfig(
            vcf_path="cohort.vcf.gz",
            pathway_gmt_path="reactome.gmt",
        ),
    )
    pipeline = SubtypeDiscoveryPipeline(config)
    result = pipeline.run()

    # Session 18: Therapeutic Hypothesis
    from pipelines import TherapeuticHypothesisPipeline, TherapeuticPipelineConfig
    config = TherapeuticPipelineConfig(
        data=DataConfig(vcf_path="cohort.vcf.gz", pathway_gmt_path="reactome.gmt"),
    )
    pipeline = TherapeuticHypothesisPipeline(config)
    result = pipeline.run()

    # Session 18: Causal Analysis
    from pipelines import CausalAnalysisPipeline, CausalAnalysisConfig
    config = CausalAnalysisConfig(
        sample_id="PATIENT_001",
        variant_genes=["SHANK3", "CHD8"],
        disrupted_pathways=["synaptic_transmission"],
    )
    pipeline = CausalAnalysisPipeline(config)
    result = pipeline.run()
"""

# Session 17: Subtype Discovery Pipeline
from pipelines.subtype_discovery import (
    SubtypeDiscoveryPipeline,
    SubtypeDiscoveryResult,
    PipelineConfig,
    DataConfig,
    ProcessingConfig,
    PathwayScoringConfig,
    ClusteringPipelineConfig,
    run_subtype_discovery,
)

# Session 18: Therapeutic Hypothesis Pipeline
from pipelines.therapeutic_hypothesis import (
    TherapeuticHypothesisPipeline,
    TherapeuticHypothesisResult,
    TherapeuticPipelineConfig,
    TherapeuticConfig,
    IndividualAnalysis,
    CausalValidation,
    run_therapeutic_hypothesis,
)

# Session 18: Causal Analysis Pipeline
from pipelines.causal_analysis import (
    CausalAnalysisPipeline,
    CausalAnalysisResult,
    CausalAnalysisConfig,
    InterventionAnalysis,
    CounterfactualAnalysis,
    PathwayMediationAnalysis,
    run_causal_analysis,
)

__all__ = [
    # === Session 17: Subtype Discovery ===
    "SubtypeDiscoveryPipeline",
    "SubtypeDiscoveryResult",
    "PipelineConfig",
    "DataConfig",
    "ProcessingConfig",
    "PathwayScoringConfig",
    "ClusteringPipelineConfig",
    "run_subtype_discovery",
    # === Session 18: Therapeutic Hypothesis ===
    "TherapeuticHypothesisPipeline",
    "TherapeuticHypothesisResult",
    "TherapeuticPipelineConfig",
    "TherapeuticConfig",
    "IndividualAnalysis",
    "CausalValidation",
    "run_therapeutic_hypothesis",
    # === Session 18: Causal Analysis ===
    "CausalAnalysisPipeline",
    "CausalAnalysisResult",
    "CausalAnalysisConfig",
    "InterventionAnalysis",
    "CounterfactualAnalysis",
    "PathwayMediationAnalysis",
    "run_causal_analysis",
]
