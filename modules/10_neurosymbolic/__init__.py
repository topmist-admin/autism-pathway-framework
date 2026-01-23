"""
Module 10: Neurosymbolic Integration

Neural-symbolic integration combining graph neural network predictions
with rule-based biological inference for autism genetics analysis.

This module provides:
- NeuroSymbolicModel: Unified model combining neural and symbolic components
- LearnedCombiner: Learned strategies for combining neural and symbolic predictions
- Explanation generation showing contributions from both pathways

Example usage:
    from modules.10_neurosymbolic import (
        NeuroSymbolicModel,
        NeuroSymbolicConfig,
        NeuroSymbolicOutput,
        create_neurosymbolic_model,
    )

    # Initialize with neural and symbolic components
    model = NeuroSymbolicModel(
        neural_model=gnn_model,
        rule_engine=rule_engine,
        config=NeuroSymbolicConfig(
            combination_method="attention",
            neural_weight=0.6,
        ),
    )

    # Process individual
    output = model.forward(
        individual_data=individual,
        graph_data=graph_data,
    )

    # Access combined predictions
    print(output.predictions)
    print(output.neural_contribution)
    print(output.symbolic_contribution)
    print(output.explanation)
"""

import sys
from pathlib import Path

# Add module to path for imports
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

# Combiner classes and functions
from combiner import (
    CombinationMethod,
    CombinerConfig,
    LearnedCombiner,
    NumpyAttentionCombiner,
    combine_gene_scores,
    create_symbolic_score_vector,
)

# Integration classes and functions
from integration import (
    NeuroSymbolicConfig,
    NeuroSymbolicOutput,
    NeuroSymbolicModel,
    create_neurosymbolic_model,
)

__all__ = [
    # Combination
    "CombinationMethod",
    "CombinerConfig",
    "LearnedCombiner",
    "NumpyAttentionCombiner",
    "combine_gene_scores",
    "create_symbolic_score_vector",
    # Integration
    "NeuroSymbolicConfig",
    "NeuroSymbolicOutput",
    "NeuroSymbolicModel",
    "create_neurosymbolic_model",
]

__version__ = "1.0.0"
