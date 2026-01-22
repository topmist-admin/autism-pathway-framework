"""
Module 05: Pretrained Embeddings

Extract gene/protein embeddings from foundation models:
- Geneformer: Gene embeddings from single-cell transcriptomics
- ESM-2: Protein embeddings from evolutionary sequences
- PubMedBERT: Gene embeddings from biomedical literature

Includes utilities for fusing embeddings from multiple sources.

Example Usage:
    >>> from modules.05_pretrained_embeddings import (
    ...     GeneformerExtractor,
    ...     ESM2Extractor,
    ...     LiteratureEmbedder,
    ...     fuse_embeddings,
    ... )
    >>>
    >>> # Extract Geneformer embeddings
    >>> geneformer = GeneformerExtractor()
    >>> gene_emb = geneformer.extract(["SHANK3", "CHD8", "SCN2A"])
    >>>
    >>> # Extract ESM-2 protein embeddings
    >>> esm2 = ESM2Extractor()
    >>> protein_emb = esm2.extract({"SHANK3": "MAEQQP...", "CHD8": "MEPSN..."})
    >>>
    >>> # Fuse multiple embedding sources
    >>> fused = fuse_embeddings({
    ...     "geneformer": gene_emb,
    ...     "esm2": protein_emb,
    ... }, method="concat")

Note:
    Models are downloaded from HuggingFace Hub on first use.
    If models are unavailable, extractors fall back to deterministic
    pseudo-embeddings for development and testing.
"""

try:
    from .base import (
        # Data structures
        NodeEmbeddings,
        VariantEffect,
        FineTuneConfig,
        # Enums
        ExtractionMode,
        EmbeddingSource,
        # Base class
        BaseEmbeddingExtractor,
        # Utilities
        normalize_gene_id,
        batch_generator,
    )
    from .geneformer import (
        GeneformerExtractor,
        create_geneformer_extractor,
        get_geneformer_embedding_dim,
        GENEFORMER_EMBEDDING_DIM,
    )
    from .esm2 import (
        ESM2Extractor,
        create_esm2_extractor,
        get_esm2_embedding_dim,
        list_esm2_models,
        ESM2_MODELS,
    )
    from .literature import (
        LiteratureEmbedder,
        create_literature_embedder,
        get_literature_embedding_dim,
        list_literature_models,
        LITERATURE_MODELS,
    )
    from .fusion import (
        EmbeddingFusion,
        FusionMethod,
        FusionConfig,
        MultiSourceEmbedder,
        fuse_embeddings,
        align_embeddings,
    )
except ImportError:
    from base import (
        NodeEmbeddings,
        VariantEffect,
        FineTuneConfig,
        ExtractionMode,
        EmbeddingSource,
        BaseEmbeddingExtractor,
        normalize_gene_id,
        batch_generator,
    )
    from geneformer import (
        GeneformerExtractor,
        create_geneformer_extractor,
        get_geneformer_embedding_dim,
        GENEFORMER_EMBEDDING_DIM,
    )
    from esm2 import (
        ESM2Extractor,
        create_esm2_extractor,
        get_esm2_embedding_dim,
        list_esm2_models,
        ESM2_MODELS,
    )
    from literature import (
        LiteratureEmbedder,
        create_literature_embedder,
        get_literature_embedding_dim,
        list_literature_models,
        LITERATURE_MODELS,
    )
    from fusion import (
        EmbeddingFusion,
        FusionMethod,
        FusionConfig,
        MultiSourceEmbedder,
        fuse_embeddings,
        align_embeddings,
    )

__all__ = [
    # Data structures
    "NodeEmbeddings",
    "VariantEffect",
    "FineTuneConfig",
    # Enums
    "ExtractionMode",
    "EmbeddingSource",
    "FusionMethod",
    # Base class
    "BaseEmbeddingExtractor",
    # Geneformer
    "GeneformerExtractor",
    "create_geneformer_extractor",
    "get_geneformer_embedding_dim",
    "GENEFORMER_EMBEDDING_DIM",
    # ESM-2
    "ESM2Extractor",
    "create_esm2_extractor",
    "get_esm2_embedding_dim",
    "list_esm2_models",
    "ESM2_MODELS",
    # Literature
    "LiteratureEmbedder",
    "create_literature_embedder",
    "get_literature_embedding_dim",
    "list_literature_models",
    "LITERATURE_MODELS",
    # Fusion
    "EmbeddingFusion",
    "FusionConfig",
    "MultiSourceEmbedder",
    "fuse_embeddings",
    "align_embeddings",
    # Utilities
    "normalize_gene_id",
    "batch_generator",
]
