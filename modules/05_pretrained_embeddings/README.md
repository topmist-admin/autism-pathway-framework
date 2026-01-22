# Module 05: Pretrained Embeddings

Extract gene and protein embeddings from foundation models pretrained on massive biological datasets.

---

## Overview

This module provides interfaces to three types of foundation models:

| Extractor | Model | Trained On | Captures |
|-----------|-------|------------|----------|
| **GeneformerExtractor** | Geneformer | 30M single-cell transcriptomes | Gene function, cell-type specificity |
| **ESM2Extractor** | ESM-2 | 250M protein sequences | Protein structure, evolutionary info |
| **LiteratureEmbedder** | PubMedBERT | Biomedical literature | Functional annotations, disease links |

### Why Pretrained Embeddings?

1. **Transfer Learning**: Leverage knowledge from massive datasets
2. **Rich Representations**: Capture complex biological relationships
3. **Complementary Views**: Different models capture different aspects
4. **Zero-shot Capability**: Embed genes without task-specific training

---

## Quick Start

### Basic Extraction

```python
from modules.05_pretrained_embeddings import (
    GeneformerExtractor,
    ESM2Extractor,
    LiteratureEmbedder,
    fuse_embeddings,
)

# 1. Geneformer: Gene embeddings from single-cell data
geneformer = GeneformerExtractor()
gene_embeddings = geneformer.extract(["SHANK3", "CHD8", "SCN2A", "NRXN1"])

# 2. ESM-2: Protein embeddings from sequences
esm2 = ESM2Extractor()
protein_embeddings = esm2.extract({
    "SHANK3": "MAEQQPVPSLPRLGR...",
    "CHD8": "MEPSNQQSVDLQ...",
})

# 3. Literature: Embeddings from functional descriptions
literature = LiteratureEmbedder()
lit_embeddings = literature.extract_from_descriptions({
    "SHANK3": "Scaffold protein in postsynaptic density...",
    "CHD8": "Chromatin remodeling factor that regulates...",
})

# 4. Fuse multiple embedding sources
fused = fuse_embeddings({
    "geneformer": gene_embeddings,
    "esm2": protein_embeddings,
    "literature": lit_embeddings,
}, method="concat")

# Use embeddings
similar_genes = fused.most_similar("SHANK3", k=5)
```

### Variant Effect Prediction

```python
from modules.05_pretrained_embeddings import ESM2Extractor

esm2 = ESM2Extractor()

# Predict pathogenicity of a missense variant
effect = esm2.predict_variant_effect(
    gene_id="SHANK3",
    wt_sequence="MAEQQPVPSLPRLGRKIKS...",
    position=123,
    ref_aa="R",
    alt_aa="W",
)

print(f"Pathogenicity: {effect.pathogenicity_score:.3f}")
print(f"Embedding shift: {effect.embedding_shift:.3f}")
```

---

## API Reference

### GeneformerExtractor

Extract gene embeddings from Geneformer (256 dimensions).

```python
class GeneformerExtractor:
    def __init__(
        self,
        model_name: str = "ctheodoris/Geneformer",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        use_fallback: bool = True,  # Use pseudo-embeddings if model unavailable
    )

    def extract(
        self,
        gene_ids: List[str],
        mode: ExtractionMode = ExtractionMode.FROZEN,
        normalize: bool = True,
    ) -> NodeEmbeddings

    def extract_with_context(
        self,
        gene_ids: List[str],
        cell_type: str,
    ) -> NodeEmbeddings
```

### ESM2Extractor

Extract protein embeddings from ESM-2 (320-2560 dimensions depending on model).

```python
class ESM2Extractor:
    def __init__(
        self,
        model_variant: str = "esm2_t33_650M",  # Options: t6_8M, t12_35M, t30_150M, t33_650M, t36_3B
        device: str = "cpu",
        use_fallback: bool = True,
    )

    def extract(
        self,
        protein_sequences: Dict[str, str],  # gene_id -> amino acid sequence
        pooling: str = "mean",  # "mean", "cls", "last"
        normalize: bool = True,
    ) -> NodeEmbeddings

    def predict_variant_effect(
        self,
        gene_id: str,
        wt_sequence: str,
        position: int,  # 1-based
        ref_aa: str,
        alt_aa: str,
    ) -> VariantEffect
```

**Available ESM-2 models:**

| Variant | Parameters | Embedding Dim | Speed |
|---------|------------|---------------|-------|
| `esm2_t6_8M` | 8M | 320 | Fastest |
| `esm2_t12_35M` | 35M | 480 | Fast |
| `esm2_t30_150M` | 150M | 640 | Medium |
| `esm2_t33_650M` | 650M | 1280 | Slow |
| `esm2_t36_3B` | 3B | 2560 | Slowest |

### LiteratureEmbedder

Extract embeddings from biomedical text using PubMedBERT (768 dimensions).

```python
class LiteratureEmbedder:
    def __init__(
        self,
        model_variant: str = "pubmedbert",  # Options: pubmedbert, biobert, biogpt, scibert
        device: str = "cpu",
        use_fallback: bool = True,
    )

    def extract_from_descriptions(
        self,
        gene_descriptions: Dict[str, str],  # gene_id -> description text
        pooling: str = "mean",
        normalize: bool = True,
    ) -> NodeEmbeddings

    def extract_from_abstracts(
        self,
        gene_abstracts: Dict[str, List[str]],  # gene_id -> list of abstracts
        aggregation: str = "mean",  # "mean", "max"
    ) -> NodeEmbeddings
```

### EmbeddingFusion

Combine embeddings from multiple sources.

```python
class EmbeddingFusion:
    def __init__(self, config: FusionConfig)

    def fuse(
        self,
        embeddings: Dict[str, NodeEmbeddings],
        node_ids: Optional[List[str]] = None,  # Uses intersection if None
    ) -> NodeEmbeddings

# Convenience function
def fuse_embeddings(
    embeddings: Dict[str, NodeEmbeddings],
    method: str = "concat",  # "concat", "average", "weighted_sum", "pca", "attention"
    output_dim: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
) -> NodeEmbeddings
```

**Fusion methods:**

| Method | Description | Requires Same Dim |
|--------|-------------|-------------------|
| `concat` | Concatenate all sources | No |
| `average` | Element-wise average | Yes |
| `weighted_sum` | Weighted combination | Yes |
| `pca` | Concat + PCA reduction | No |
| `attention` | Self-attention fusion | No (pads) |

---

## Fallback Mode

If HuggingFace models are unavailable (network issues, missing dependencies), all extractors automatically fall back to generating **deterministic pseudo-embeddings** based on input content.

This is useful for:
- Development and testing without GPU
- CI/CD pipelines
- Offline environments

Fallback embeddings:
- Are deterministic (same input → same output)
- Incorporate basic features (gene name, sequence composition, keywords)
- Are NOT biologically meaningful

To disable fallback and require real models:
```python
extractor = GeneformerExtractor(use_fallback=False)
```

---

## Model Dependencies

Models are downloaded from HuggingFace Hub on first use:

```bash
# Install transformers for model loading
pip install transformers torch

# Models are cached in ~/.cache/huggingface/
# Or specify custom cache:
extractor = GeneformerExtractor(cache_dir="/path/to/cache")
```

**Approximate model sizes:**

| Model | Download Size | RAM Usage |
|-------|---------------|-----------|
| Geneformer | ~500 MB | ~1 GB |
| ESM-2 (650M) | ~2.5 GB | ~5 GB |
| ESM-2 (8M) | ~30 MB | ~100 MB |
| PubMedBERT | ~400 MB | ~800 MB |

---

## Integration Examples

### With Module 03 Knowledge Graph

```python
from modules.03_knowledge_graph import KnowledgeGraphBuilder
from modules.05_pretrained_embeddings import GeneformerExtractor

# Build knowledge graph
kg = KnowledgeGraphBuilder().add_genes(genes).add_pathways(pathways).build()

# Extract embeddings for all genes in the graph
gene_ids = kg.get_nodes_by_type(NodeType.GENE)
geneformer = GeneformerExtractor()
embeddings = geneformer.extract(gene_ids)

# Use as node features for GNN
```

### With Module 04 Graph Embeddings

```python
from modules.04_graph_embeddings import train_embeddings
from modules.05_pretrained_embeddings import fuse_embeddings

# Get graph structure embeddings
graph_model, graph_emb = train_embeddings(kg, model_type="transe")

# Get pretrained embeddings
geneformer_emb = GeneformerExtractor().extract(gene_ids)

# Fuse both types
combined = fuse_embeddings({
    "graph": graph_emb,
    "geneformer": geneformer_emb,
}, method="concat")
```

---

## Testing

```bash
# Run all tests
python -m pytest modules/05_pretrained_embeddings/tests/ -v

# Run specific test class
python -m pytest modules/05_pretrained_embeddings/tests/test_pretrained.py::TestESM2Extractor -v
```

---

## Files

| File | Description |
|------|-------------|
| `base.py` | NodeEmbeddings, VariantEffect, base classes |
| `geneformer.py` | Geneformer extractor |
| `esm2.py` | ESM-2 extractor with variant prediction |
| `literature.py` | PubMedBERT/BioBERT embedder |
| `fusion.py` | Multi-source embedding fusion |
| `__init__.py` | Module exports |
| `tests/test_pretrained.py` | Unit tests |

---

## References

1. **Geneformer**: Theodoris et al. (2023). "Transfer learning enables predictions in network biology." Nature.
2. **ESM-2**: Lin et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science.
3. **PubMedBERT**: Gu et al. (2021). "Domain-Specific Pretraining for Vertical Search." ACL.
