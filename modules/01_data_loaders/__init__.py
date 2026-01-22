"""
Module 01: Data Loaders

Provides standardized data loading utilities for the autism genetics analysis pipeline.
"""

try:
    # When imported as a package
    from .vcf_loader import VCFLoader, Variant, VariantDataset, ValidationReport
    from .pathway_loader import PathwayLoader, PathwayDatabase
    from .expression_loader import ExpressionLoader, DevelopmentalExpression
    from .single_cell_loader import SingleCellLoader, SingleCellAtlas
    from .constraint_loader import ConstraintLoader, GeneConstraints, SFARIGenes
    from .annotation_loader import AnnotationLoader
except ImportError:
    # When run directly or during pytest collection
    from vcf_loader import VCFLoader, Variant, VariantDataset, ValidationReport
    from pathway_loader import PathwayLoader, PathwayDatabase
    from expression_loader import ExpressionLoader, DevelopmentalExpression
    from single_cell_loader import SingleCellLoader, SingleCellAtlas
    from constraint_loader import ConstraintLoader, GeneConstraints, SFARIGenes
    from annotation_loader import AnnotationLoader

__all__ = [
    # VCF
    "VCFLoader",
    "Variant",
    "VariantDataset",
    "ValidationReport",
    # Pathways
    "PathwayLoader",
    "PathwayDatabase",
    # Expression
    "ExpressionLoader",
    "DevelopmentalExpression",
    # Single-cell
    "SingleCellLoader",
    "SingleCellAtlas",
    # Constraints
    "ConstraintLoader",
    "GeneConstraints",
    "SFARIGenes",
    # Annotations
    "AnnotationLoader",
]
