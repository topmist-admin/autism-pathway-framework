"""
Autism Pathway Framework

A pathway- and network-based framework for analyzing genetic heterogeneity
in Autism Spectrum Disorder.
"""

__version__ = "0.1.0"
__author__ = "Rohit Chauhan"

from .pipeline import DemoPipeline, PipelineConfig
from .validation import ValidationGates, ValidationGatesResult, ValidationResult

__all__ = [
    "DemoPipeline",
    "PipelineConfig",
    "ValidationGates",
    "ValidationGatesResult",
    "ValidationResult",
    "__version__",
]
