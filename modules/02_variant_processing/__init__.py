"""
Module 02: Variant Processing

Provides variant processing utilities including QC filtering,
functional annotation, and gene burden calculation.
"""

try:
    # When imported as a package
    from .qc_filters import QCFilter, QCConfig, QCReport
    from .annotation import VariantAnnotator, AnnotatedVariant, VariantConsequence
    from .gene_burden import GeneBurdenCalculator, GeneBurdenMatrix, WeightConfig
except ImportError:
    # When run directly or during pytest collection
    from qc_filters import QCFilter, QCConfig, QCReport
    from annotation import VariantAnnotator, AnnotatedVariant, VariantConsequence
    from gene_burden import GeneBurdenCalculator, GeneBurdenMatrix, WeightConfig

__all__ = [
    # QC
    "QCFilter",
    "QCConfig",
    "QCReport",
    # Annotation
    "VariantAnnotator",
    "AnnotatedVariant",
    "VariantConsequence",
    # Gene Burden
    "GeneBurdenCalculator",
    "GeneBurdenMatrix",
    "WeightConfig",
]
