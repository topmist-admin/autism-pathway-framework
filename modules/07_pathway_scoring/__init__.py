"""
Module 07: Pathway Scoring

Aggregates gene-level burden scores to pathway-level scores and provides
network-based signal refinement through diffusion algorithms.

Components:
- PathwayAggregator: Aggregates gene scores to pathway scores
- NetworkPropagator: Spreads signals through biological networks
- PathwayScoreNormalizer: Normalizes scores for downstream analysis

Example Usage:
    from modules.07_pathway_scoring import (
        PathwayAggregator,
        NetworkPropagator,
        PathwayScoreNormalizer,
        AggregationConfig,
        PropagationConfig,
        NormalizationConfig,
    )

    # Aggregate gene burdens to pathway scores
    aggregator = PathwayAggregator(AggregationConfig(
        method=AggregationMethod.WEIGHTED_SUM,
        normalize_by_pathway_size=True,
    ))
    pathway_scores = aggregator.aggregate(gene_burdens, pathway_db)

    # Optionally propagate through network
    propagator = NetworkPropagator(PropagationConfig(
        method=PropagationMethod.RANDOM_WALK,
        restart_prob=0.5,
    ))
    propagator.build_network(knowledge_graph)
    propagated_burdens = propagator.propagate_gene_burdens(gene_burdens)

    # Normalize for analysis
    normalizer = PathwayScoreNormalizer(NormalizationConfig(
        method=NormalizationMethod.ZSCORE,
    ))
    normalized_scores = normalizer.normalize(pathway_scores)
"""

import sys
from pathlib import Path

# Add module directory to path to handle numeric prefix in module name
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from aggregation import (
    AggregationConfig,
    AggregationMethod,
    PathwayAggregator,
    PathwayScoreMatrix,
)

from network_propagation import (
    NetworkPropagator,
    PropagationConfig,
    PropagationMethod,
    PropagationResult,
)

from normalization import (
    NormalizationConfig,
    NormalizationMethod,
    PathwayScoreNormalizer,
)

__all__ = [
    # Aggregation
    "AggregationConfig",
    "AggregationMethod",
    "PathwayAggregator",
    "PathwayScoreMatrix",
    # Network propagation
    "NetworkPropagator",
    "PropagationConfig",
    "PropagationMethod",
    "PropagationResult",
    # Normalization
    "NormalizationConfig",
    "NormalizationMethod",
    "PathwayScoreNormalizer",
]
