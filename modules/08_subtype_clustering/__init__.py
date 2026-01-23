"""
Module 08: Subtype Clustering

Identifies autism subtypes through unsupervised clustering of pathway disruption
patterns with bootstrap stability testing and subtype characterization.
"""

import sys
from pathlib import Path

# Add module directory to path to handle numeric prefix in module name
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

# Import clustering components
from clustering import (
    ClusteringMethod,
    ClusteringConfig,
    ClusteringResult,
    SubtypeClusterer,
)

# Import stability analysis
from stability import (
    StabilityConfig,
    StabilityResult,
    StabilityAnalyzer,
)

# Import characterization
from characterization import (
    CharacterizationConfig,
    PathwaySignature,
    SubtypeProfile,
    SubtypeCharacterizer,
)

__all__ = [
    # Clustering
    "ClusteringMethod",
    "ClusteringConfig",
    "ClusteringResult",
    "SubtypeClusterer",
    # Stability
    "StabilityConfig",
    "StabilityResult",
    "StabilityAnalyzer",
    # Characterization
    "CharacterizationConfig",
    "PathwaySignature",
    "SubtypeProfile",
    "SubtypeCharacterizer",
]
