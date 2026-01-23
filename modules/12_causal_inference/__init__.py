"""
Module 12: Causal Inference

Causal reasoning framework for understanding genetic mechanisms
and intervention effects in ASD research.

Components:
- StructuralCausalModel: Encodes causal relationships
- DoCalculusEngine: Intervention reasoning (do-calculus)
- CounterfactualEngine: Counterfactual queries
- CausalEffectEstimator: Direct/indirect effect estimation
"""

import sys
from pathlib import Path

# Add module directory to path for imports
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

# Causal graph components
from causal_graph import (
    CausalNodeType,
    CausalEdgeType,
    CausalNode,
    CausalEdge,
    CausalQuery,
    CausalQueryBuilder,
    StructuralCausalModel,
    create_sample_asd_scm,
)

# Do-calculus components
from do_calculus import (
    Distribution,
    IntervenedModel,
    DoCalculusEngine,
)

# Counterfactual reasoning
from counterfactuals import (
    CounterfactualResult,
    CounterfactualEngine,
)

# Effect estimation
from effect_estimation import (
    MediationResult,
    EffectDecomposition,
    CausalEffectEstimator,
)

__all__ = [
    # Enums
    "CausalNodeType",
    "CausalEdgeType",
    # Data classes
    "CausalNode",
    "CausalEdge",
    "CausalQuery",
    "Distribution",
    "CounterfactualResult",
    "MediationResult",
    "EffectDecomposition",
    # Builders
    "CausalQueryBuilder",
    # Models and engines
    "StructuralCausalModel",
    "IntervenedModel",
    "DoCalculusEngine",
    "CounterfactualEngine",
    "CausalEffectEstimator",
    # Factory functions
    "create_sample_asd_scm",
]
