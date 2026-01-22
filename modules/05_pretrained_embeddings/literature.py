"""
Literature-based Gene Embedding Extractor

Extract gene embeddings from biomedical literature using PubMedBERT
or BioGPT. Useful for capturing functional annotations, disease
associations, and biological context from text.

References:
    - PubMedBERT: Gu et al. (2021) "Domain-Specific Pretraining for Vertical Search"
    - BioGPT: Luo et al. (2022) "BioGPT: Generative Pre-trained Transformer for Biomedical Text"
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
import logging
import re

import numpy as np

try:
    from .base import (
        BaseEmbeddingExtractor,
        NodeEmbeddings,
        EmbeddingSource,
        ExtractionMode,
        normalize_gene_id,
        batch_generator,
    )
except ImportError:
    from base import (
        BaseEmbeddingExtractor,
        NodeEmbeddings,
        EmbeddingSource,
        ExtractionMode,
        normalize_gene_id,
        batch_generator,
    )

logger = logging.getLogger(__name__)

# Available biomedical language models
LITERATURE_MODELS = {
    "pubmedbert": {
        "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "embedding_dim": 768,
        "description": "PubMedBERT trained on PubMed abstracts",
    },
    "pubmedbert_fulltext": {
        "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "embedding_dim": 768,
        "description": "PubMedBERT trained on full-text articles",
    },
    "biobert": {
        "name": "dmis-lab/biobert-v1.1",
        "embedding_dim": 768,
        "description": "BioBERT v1.1",
    },
    "biogpt": {
        "name": "microsoft/biogpt",
        "embedding_dim": 1024,
        "description": "BioGPT generative model",
    },
    "scibert": {
        "name": "allenai/scibert_scivocab_uncased",
        "embedding_dim": 768,
        "description": "SciBERT for scientific text",
    },
}

DEFAULT_LITERATURE_MODEL = "pubmedbert"


class LiteratureEmbedder(BaseEmbeddingExtractor):
    """
    Extract gene embeddings from biomedical literature.

    Uses PubMedBERT or similar models to embed gene descriptions,
    functional annotations, or aggregated literature about each gene.

    This is useful for capturing:
    - Functional annotations (GO terms, pathways)
    - Disease associations
    - Drug interactions
    - Biological context not captured by sequence models

    Example:
        >>> embedder = LiteratureEmbedder()
        >>> descriptions = {
        ...     "SHANK3": "SHANK3 is a scaffold protein in the postsynaptic density...",
        ...     "CHD8": "CHD8 is a chromatin remodeling factor that regulates...",
        ... }
        >>> embeddings = embedder.extract_from_descriptions(descriptions)

    Note:
        If the model is not installed, falls back to deterministic
        pseudo-embeddings based on text content.
    """

    def __init__(
        self,
        model_variant: str = DEFAULT_LITERATURE_MODEL,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        use_fallback: bool = True,
    ):
        """
        Initialize literature embedder.

        Args:
            model_variant: Model variant (e.g., "pubmedbert", "biogpt")
            device: Device for inference ("cpu" or "cuda")
            cache_dir: Directory for caching downloaded models
            use_fallback: If True, use fallback mode when model unavailable
        """
        if model_variant not in LITERATURE_MODELS:
            available = ", ".join(LITERATURE_MODELS.keys())
            raise ValueError(
                f"Unknown model variant: {model_variant}. "
                f"Available: {available}"
            )

        self._model_config = LITERATURE_MODELS[model_variant]
        model_name = self._model_config["name"]

        super().__init__(model_name, device, cache_dir)
        self.model_variant = model_variant
        self.use_fallback = use_fallback
        self._fallback_mode = False

    @property
    def embedding_dim(self) -> int:
        """Return model embedding dimension."""
        return self._model_config["embedding_dim"]

    @property
    def source(self) -> EmbeddingSource:
        """Return embedding source type."""
        return EmbeddingSource.PUBMEDBERT

    def _load_model(self) -> None:
        """Load the biomedical language model."""
        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading literature model: {self.model_name}")

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                )
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                )
                self._model.to(self.device)
                self._model.eval()

                logger.info(f"Loaded {self.model_variant} with dim {self.embedding_dim}")
                self._fallback_mode = False

            except Exception as e:
                if self.use_fallback:
                    logger.warning(
                        f"Could not load model: {e}. "
                        "Using fallback mode with deterministic pseudo-embeddings."
                    )
                    self._fallback_mode = True
                else:
                    raise

        except ImportError:
            if self.use_fallback:
                logger.warning(
                    "transformers library not installed. "
                    "Using fallback mode with deterministic pseudo-embeddings."
                )
                self._fallback_mode = True
            else:
                raise ImportError(
                    "Please install transformers: pip install transformers"
                )

    def _generate_fallback_embedding(self, text: str) -> np.ndarray:
        """
        Generate deterministic pseudo-embedding based on text content.

        Incorporates basic text features like word counts and key terms.
        """
        # Create deterministic seed from text hash
        text_hash = hashlib.sha256(text.lower().encode()).digest()
        seed = int.from_bytes(text_hash[:4], byteorder="big")

        # Generate base embedding
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self.embedding_dim).astype(np.float32)

        # Incorporate text features
        words = text.lower().split()
        word_count = len(words)
        unique_words = len(set(words))

        # Biomedical keywords that might indicate function
        bio_keywords = {
            "synapse": 0,
            "neuron": 1,
            "chromatin": 2,
            "transcription": 3,
            "receptor": 4,
            "kinase": 5,
            "channel": 6,
            "mutation": 7,
            "autism": 8,
            "developmental": 9,
            "expression": 10,
            "protein": 11,
            "pathway": 12,
            "signaling": 13,
            "brain": 14,
        }

        for keyword, idx in bio_keywords.items():
            if keyword in text.lower() and idx < self.embedding_dim:
                embedding[idx] += 1.0

        # Add text statistics
        embedding[0] += np.log1p(word_count) / 10
        embedding[1] += unique_words / max(word_count, 1)

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        return embedding

    def extract(
        self,
        inputs: Union[List[str], Dict[str, str]],
        **kwargs: Any,
    ) -> NodeEmbeddings:
        """
        Extract embeddings from text inputs.

        Args:
            inputs: Either a list of texts or dict mapping IDs to texts
            **kwargs: Additional parameters (batch_size, max_length, etc.)

        Returns:
            NodeEmbeddings instance
        """
        if isinstance(inputs, dict):
            return self.extract_from_descriptions(inputs, **kwargs)
        else:
            # Treat as descriptions with auto-generated IDs
            descriptions = {f"text_{i}": text for i, text in enumerate(inputs)}
            return self.extract_from_descriptions(descriptions, **kwargs)

    def extract_from_descriptions(
        self,
        gene_descriptions: Dict[str, str],
        batch_size: int = 16,
        max_length: int = 512,
        pooling: str = "mean",
        normalize: bool = True,
    ) -> NodeEmbeddings:
        """
        Extract embeddings from gene functional descriptions.

        Args:
            gene_descriptions: Dict mapping gene ID to description text
            batch_size: Batch size for inference
            max_length: Maximum token length
            pooling: Pooling method ("mean", "cls")
            normalize: Whether to L2-normalize embeddings

        Returns:
            NodeEmbeddings with text-based representations
        """
        self.ensure_loaded()

        gene_ids = list(gene_descriptions.keys())
        descriptions = [gene_descriptions[g] for g in gene_ids]

        logger.info(f"Extracting embeddings for {len(gene_ids)} gene descriptions")

        if self._fallback_mode:
            embeddings = self._extract_fallback(descriptions)
        else:
            embeddings = self._extract_with_model(
                descriptions, batch_size, max_length, pooling
            )

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)

        return NodeEmbeddings(
            node_ids=gene_ids,
            embeddings=embeddings,
            source=self.source,
            model_name=self.model_name,
            extraction_mode=ExtractionMode.FROZEN,
            metadata={
                "fallback_mode": self._fallback_mode,
                "pooling": pooling,
                "max_length": max_length,
                "normalized": normalize,
                "n_genes": len(gene_ids),
            },
        )

    def _extract_fallback(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings using fallback mode."""
        embeddings = []
        for text in texts:
            emb = self._generate_fallback_embedding(text)
            embeddings.append(emb)
        return np.stack(embeddings)

    def _extract_with_model(
        self,
        texts: List[str],
        batch_size: int,
        max_length: int,
        pooling: str,
    ) -> np.ndarray:
        """Extract embeddings using the actual model."""
        import torch

        embeddings = []

        with torch.no_grad():
            for batch_texts in batch_generator(texts, batch_size):
                # Tokenize
                inputs = self._tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get model outputs
                outputs = self._model(**inputs)
                hidden_states = outputs.last_hidden_state

                # Pool embeddings
                if pooling == "mean":
                    attention_mask = inputs["attention_mask"]
                    mask_expanded = attention_mask.unsqueeze(-1).expand(
                        hidden_states.size()
                    )
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

                elif pooling == "cls":
                    batch_embeddings = hidden_states[:, 0, :].cpu().numpy()

                else:
                    raise ValueError(f"Unknown pooling method: {pooling}")

                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def extract_from_abstracts(
        self,
        gene_abstracts: Dict[str, List[str]],
        aggregation: str = "mean",
        **kwargs: Any,
    ) -> NodeEmbeddings:
        """
        Extract embeddings from aggregated literature abstracts.

        Args:
            gene_abstracts: Dict mapping gene ID to list of abstract texts
            aggregation: How to aggregate multiple abstracts ("mean", "max")
            **kwargs: Additional parameters passed to extract_from_descriptions

        Returns:
            NodeEmbeddings with aggregated literature representations
        """
        self.ensure_loaded()

        # First, embed all individual abstracts
        all_texts = []
        text_to_gene = []
        for gene_id, abstracts in gene_abstracts.items():
            for abstract in abstracts:
                all_texts.append(abstract)
                text_to_gene.append(gene_id)

        if not all_texts:
            return NodeEmbeddings(
                node_ids=[],
                embeddings=np.array([]).reshape(0, self.embedding_dim),
                source=self.source,
                model_name=self.model_name,
            )

        # Extract embeddings for all abstracts
        all_descriptions = {f"text_{i}": text for i, text in enumerate(all_texts)}
        all_embeddings = self.extract_from_descriptions(all_descriptions, **kwargs)

        # Aggregate by gene
        gene_ids = list(gene_abstracts.keys())
        aggregated = np.zeros((len(gene_ids), self.embedding_dim))
        gene_to_idx = {g: i for i, g in enumerate(gene_ids)}

        gene_counts = {g: 0 for g in gene_ids}

        for i, gene_id in enumerate(text_to_gene):
            emb = all_embeddings.embeddings[i]
            idx = gene_to_idx[gene_id]

            if aggregation == "mean":
                aggregated[idx] += emb
                gene_counts[gene_id] += 1
            elif aggregation == "max":
                aggregated[idx] = np.maximum(aggregated[idx], emb)

        # Normalize mean aggregation
        if aggregation == "mean":
            for gene_id, count in gene_counts.items():
                if count > 0:
                    idx = gene_to_idx[gene_id]
                    aggregated[idx] /= count

        # L2 normalize
        norms = np.linalg.norm(aggregated, axis=1, keepdims=True)
        aggregated = aggregated / (norms + 1e-10)

        return NodeEmbeddings(
            node_ids=gene_ids,
            embeddings=aggregated,
            source=self.source,
            model_name=self.model_name,
            extraction_mode=ExtractionMode.FROZEN,
            metadata={
                "aggregation": aggregation,
                "total_abstracts": len(all_texts),
            },
        )


def create_literature_embedder(
    model_variant: str = DEFAULT_LITERATURE_MODEL,
    device: str = "cpu",
    cache_dir: Optional[str] = None,
    use_fallback: bool = True,
) -> LiteratureEmbedder:
    """
    Factory function to create a literature embedder.

    Args:
        model_variant: Model variant
        device: Device for inference
        cache_dir: Model cache directory
        use_fallback: Whether to use fallback mode if model unavailable

    Returns:
        LiteratureEmbedder instance
    """
    return LiteratureEmbedder(
        model_variant=model_variant,
        device=device,
        cache_dir=cache_dir,
        use_fallback=use_fallback,
    )


def get_literature_embedding_dim(model_variant: str = DEFAULT_LITERATURE_MODEL) -> int:
    """Return embedding dimension for a literature model variant."""
    if model_variant not in LITERATURE_MODELS:
        raise ValueError(f"Unknown model variant: {model_variant}")
    return LITERATURE_MODELS[model_variant]["embedding_dim"]


def list_literature_models() -> Dict[str, Dict[str, Any]]:
    """Return available literature model variants."""
    return LITERATURE_MODELS.copy()
