"""
Geneformer Embedding Extractor

Extract gene embeddings from Geneformer, a foundation model pretrained on
~30 million single-cell transcriptomes.

Reference:
    Theodoris et al. (2023). "Transfer learning enables predictions in network biology"
    https://www.nature.com/articles/s41586-023-06139-9

Model: https://huggingface.co/ctheodoris/Geneformer
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
import logging

import numpy as np

try:
    from .base import (
        BaseEmbeddingExtractor,
        NodeEmbeddings,
        EmbeddingSource,
        ExtractionMode,
        FineTuneConfig,
        normalize_gene_id,
        batch_generator,
    )
except ImportError:
    from base import (
        BaseEmbeddingExtractor,
        NodeEmbeddings,
        EmbeddingSource,
        ExtractionMode,
        FineTuneConfig,
        normalize_gene_id,
        batch_generator,
    )

logger = logging.getLogger(__name__)

# Geneformer model configuration
GENEFORMER_MODEL_NAME = "ctheodoris/Geneformer"
GENEFORMER_EMBEDDING_DIM = 256  # Geneformer uses 256-dim embeddings


class GeneformerExtractor(BaseEmbeddingExtractor):
    """
    Extract gene embeddings from Geneformer.

    Geneformer is a context-aware, attention-based deep learning model
    pretrained on a large-scale corpus of ~30 million single-cell
    transcriptomes. It learns gene representations that capture
    functional relationships.

    The model can be used in two modes:
    1. FROZEN: Use pretrained weights as-is (fast, no training)
    2. FINE_TUNED: Fine-tune on domain-specific data (requires GPU)

    Example:
        >>> extractor = GeneformerExtractor()
        >>> genes = ["SHANK3", "CHD8", "SCN2A", "NRXN1"]
        >>> embeddings = extractor.extract(genes)
        >>> similar = embeddings.most_similar("SHANK3", k=5)

    Note:
        If Geneformer is not installed, the extractor falls back to
        generating deterministic pseudo-embeddings based on gene names.
        This is useful for development and testing.
    """

    def __init__(
        self,
        model_name: str = GENEFORMER_MODEL_NAME,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        use_fallback: bool = True,
    ):
        """
        Initialize Geneformer extractor.

        Args:
            model_name: HuggingFace model name or local path
            device: Device for inference ("cpu" or "cuda")
            cache_dir: Directory for caching downloaded models
            use_fallback: If True, use fallback mode when model unavailable
        """
        super().__init__(model_name, device, cache_dir)
        self.use_fallback = use_fallback
        self._fallback_mode = False
        self._gene_token_dict: Dict[str, int] = {}

    @property
    def embedding_dim(self) -> int:
        """Return Geneformer embedding dimension."""
        return GENEFORMER_EMBEDDING_DIM

    @property
    def source(self) -> EmbeddingSource:
        """Return embedding source type."""
        return EmbeddingSource.GENEFORMER

    def _load_model(self) -> None:
        """Load Geneformer model from HuggingFace."""
        try:
            from transformers import BertModel, BertConfig

            logger.info(f"Loading Geneformer model: {self.model_name}")

            # Try to load the actual Geneformer model
            try:
                self._model = BertModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                )
                self._model.to(self.device)
                self._model.eval()

                # Load gene token dictionary
                self._load_gene_token_dict()

                logger.info(
                    f"Loaded Geneformer with {len(self._gene_token_dict)} gene tokens"
                )
                self._fallback_mode = False

            except Exception as e:
                if self.use_fallback:
                    logger.warning(
                        f"Could not load Geneformer model: {e}. "
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

    def _load_gene_token_dict(self) -> None:
        """Load the gene-to-token mapping from Geneformer."""
        try:
            # Geneformer uses a custom gene token dictionary
            # This maps gene symbols/Ensembl IDs to token indices
            import json
            from huggingface_hub import hf_hub_download

            token_dict_path = hf_hub_download(
                repo_id=self.model_name,
                filename="token_dictionary.pkl",
                cache_dir=self.cache_dir,
            )

            import pickle

            with open(token_dict_path, "rb") as f:
                self._gene_token_dict = pickle.load(f)

        except Exception as e:
            logger.warning(f"Could not load gene token dictionary: {e}")
            self._gene_token_dict = {}

    def _generate_fallback_embedding(self, gene_id: str) -> np.ndarray:
        """
        Generate deterministic pseudo-embedding based on gene name.

        Uses hash of gene name to create reproducible embeddings.
        Not biologically meaningful but useful for development/testing.
        """
        # Create deterministic seed from gene name
        hash_bytes = hashlib.sha256(gene_id.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], byteorder="big")

        # Generate deterministic random embedding
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self.embedding_dim).astype(np.float32)

        # Normalize to unit length
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        return embedding

    def extract(
        self,
        gene_ids: List[str],
        mode: ExtractionMode = ExtractionMode.FROZEN,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> NodeEmbeddings:
        """
        Extract embeddings for a list of genes.

        Args:
            gene_ids: List of gene symbols or Ensembl IDs
            mode: Extraction mode (FROZEN or FINE_TUNED)
            batch_size: Batch size for inference
            normalize: Whether to L2-normalize embeddings

        Returns:
            NodeEmbeddings with gene representations
        """
        self.ensure_loaded()

        # Normalize gene IDs
        normalized_ids = [normalize_gene_id(g) for g in gene_ids]
        unique_ids = list(dict.fromkeys(normalized_ids))  # Preserve order, remove dups

        logger.info(f"Extracting embeddings for {len(unique_ids)} genes")

        if self._fallback_mode:
            embeddings = self._extract_fallback(unique_ids)
        else:
            embeddings = self._extract_with_model(unique_ids, batch_size)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)

        return NodeEmbeddings(
            node_ids=unique_ids,
            embeddings=embeddings,
            source=self.source,
            model_name=self.model_name,
            extraction_mode=mode,
            metadata={
                "fallback_mode": self._fallback_mode,
                "normalized": normalize,
                "n_genes": len(unique_ids),
            },
        )

    def _extract_fallback(self, gene_ids: List[str]) -> np.ndarray:
        """Extract embeddings using fallback mode."""
        embeddings = []
        for gene_id in gene_ids:
            emb = self._generate_fallback_embedding(gene_id)
            embeddings.append(emb)
        return np.stack(embeddings)

    def _extract_with_model(
        self,
        gene_ids: List[str],
        batch_size: int,
    ) -> np.ndarray:
        """Extract embeddings using the actual Geneformer model."""
        import torch

        embeddings = []

        # Get token indices for genes
        token_indices = []
        valid_genes = []

        for gene_id in gene_ids:
            # Try different formats
            token_idx = self._gene_token_dict.get(gene_id)
            if token_idx is None:
                # Try lowercase
                token_idx = self._gene_token_dict.get(gene_id.lower())
            if token_idx is None:
                # Try with ENSG prefix variations
                for key in self._gene_token_dict:
                    if gene_id in key or key in gene_id:
                        token_idx = self._gene_token_dict[key]
                        break

            if token_idx is not None:
                token_indices.append(token_idx)
                valid_genes.append(gene_id)
            else:
                # Use fallback for unknown genes
                emb = self._generate_fallback_embedding(gene_id)
                embeddings.append(emb)
                valid_genes.append(gene_id)
                token_indices.append(-1)  # Marker for fallback

        # Process in batches
        with torch.no_grad():
            for i in range(0, len(token_indices), batch_size):
                batch_indices = token_indices[i : i + batch_size]

                # Filter out fallback markers
                real_indices = [idx for idx in batch_indices if idx >= 0]

                if real_indices:
                    # Create input tensor
                    input_ids = torch.tensor([real_indices], device=self.device)

                    # Get embeddings from model
                    outputs = self._model(input_ids)
                    batch_embeddings = outputs.last_hidden_state[0].cpu().numpy()

                    # Add to results
                    real_idx = 0
                    for idx in batch_indices:
                        if idx >= 0:
                            embeddings.append(batch_embeddings[real_idx])
                            real_idx += 1
                        # Fallback embeddings already added above

        return np.stack(embeddings)

    def extract_with_context(
        self,
        gene_ids: List[str],
        cell_type: str,
        expression_profile: Optional[Dict[str, float]] = None,
    ) -> NodeEmbeddings:
        """
        Extract cell-type-contextualized embeddings.

        Geneformer can produce different embeddings for the same gene
        depending on cellular context (expression of other genes).

        Args:
            gene_ids: List of genes to embed
            cell_type: Cell type context (e.g., "neuron", "astrocyte")
            expression_profile: Optional expression values for context genes

        Returns:
            NodeEmbeddings with context-specific representations
        """
        self.ensure_loaded()

        if self._fallback_mode:
            # In fallback mode, incorporate cell type into hash
            embeddings = []
            for gene_id in gene_ids:
                context_id = f"{gene_id}_{cell_type}"
                emb = self._generate_fallback_embedding(context_id)
                embeddings.append(emb)

            return NodeEmbeddings(
                node_ids=gene_ids,
                embeddings=np.stack(embeddings),
                source=self.source,
                model_name=self.model_name,
                extraction_mode=ExtractionMode.FROZEN,
                metadata={
                    "cell_type": cell_type,
                    "contextualized": True,
                    "fallback_mode": True,
                },
            )

        # For actual model, would use expression profile to order genes
        # and extract embeddings with attention to context
        logger.warning(
            "Full contextualized extraction requires Geneformer fine-tuning. "
            "Using standard extraction with cell type metadata."
        )
        embeddings = self.extract(gene_ids)
        embeddings.metadata["cell_type"] = cell_type
        embeddings.metadata["contextualized"] = True
        return embeddings


def create_geneformer_extractor(
    device: str = "cpu",
    cache_dir: Optional[str] = None,
    use_fallback: bool = True,
) -> GeneformerExtractor:
    """
    Factory function to create a Geneformer extractor.

    Args:
        device: Device for inference
        cache_dir: Model cache directory
        use_fallback: Whether to use fallback mode if model unavailable

    Returns:
        GeneformerExtractor instance
    """
    return GeneformerExtractor(
        model_name=GENEFORMER_MODEL_NAME,
        device=device,
        cache_dir=cache_dir,
        use_fallback=use_fallback,
    )


def get_geneformer_embedding_dim() -> int:
    """Return Geneformer embedding dimension."""
    return GENEFORMER_EMBEDDING_DIM
