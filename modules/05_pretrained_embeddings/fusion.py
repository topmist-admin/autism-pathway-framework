"""
Embedding Fusion

Fuse embeddings from multiple sources (Geneformer, ESM-2, literature,
knowledge graph) into unified gene representations.

Supports multiple fusion strategies:
- Concatenation
- Weighted sum
- Attention-based fusion
- Learned (MLP) fusion
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np

try:
    from .base import NodeEmbeddings, EmbeddingSource
except ImportError:
    from base import NodeEmbeddings, EmbeddingSource

logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    """Available embedding fusion methods."""

    CONCAT = "concat"  # Simple concatenation
    WEIGHTED_SUM = "weighted_sum"  # Fixed or learned weights
    ATTENTION = "attention"  # Self-attention fusion
    LEARNED = "learned"  # MLP-based fusion
    PCA = "pca"  # PCA dimensionality reduction after concat
    AVERAGE = "average"  # Simple averaging (requires same dim)


@dataclass
class FusionConfig:
    """Configuration for embedding fusion."""

    method: FusionMethod = FusionMethod.CONCAT
    output_dim: Optional[int] = None  # None = auto (sum of dims for concat)
    normalize_inputs: bool = True
    normalize_output: bool = True

    # For weighted_sum
    weights: Optional[Dict[str, float]] = None

    # For learned fusion
    hidden_dims: List[int] = field(default_factory=lambda: [256])
    dropout: float = 0.1

    # For PCA
    n_components: Optional[int] = None  # None = keep 95% variance

    def __post_init__(self):
        if isinstance(self.method, str):
            self.method = FusionMethod(self.method)


class EmbeddingFusion:
    """
    Fuse embeddings from multiple sources into unified representations.

    This class handles the challenge of combining embeddings from
    different models (Geneformer, ESM-2, literature, knowledge graph)
    which may have different dimensions and capture different aspects
    of gene function.

    Example:
        >>> fusion = EmbeddingFusion(FusionConfig(method=FusionMethod.CONCAT))
        >>> combined = fusion.fuse({
        ...     "geneformer": geneformer_embeddings,
        ...     "esm2": esm2_embeddings,
        ...     "literature": literature_embeddings,
        ... })

    Fusion methods:
    - CONCAT: Concatenate all embeddings (best for preserving information)
    - WEIGHTED_SUM: Weighted combination (requires same dimensions)
    - ATTENTION: Self-attention over embedding sources
    - LEARNED: Train an MLP to combine embeddings (requires labels)
    - PCA: Concatenate then reduce dimensionality with PCA
    - AVERAGE: Simple average (requires same dimensions)
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """
        Initialize fusion module.

        Args:
            config: Fusion configuration
        """
        self.config = config or FusionConfig()
        self._pca_model = None
        self._learned_model = None
        self._source_dims: Dict[str, int] = {}

    def fuse(
        self,
        embeddings: Dict[str, NodeEmbeddings],
        node_ids: Optional[List[str]] = None,
    ) -> NodeEmbeddings:
        """
        Fuse embeddings from multiple sources.

        Args:
            embeddings: Dict mapping source name to NodeEmbeddings
            node_ids: Optional list of node IDs to include
                     (uses intersection if not provided)

        Returns:
            Fused NodeEmbeddings
        """
        if not embeddings:
            raise ValueError("No embeddings provided for fusion")

        # Find common nodes across all sources
        if node_ids is None:
            node_sets = [set(emb.node_ids) for emb in embeddings.values()]
            common_nodes = set.intersection(*node_sets)
            node_ids = sorted(common_nodes)

        if not node_ids:
            logger.warning("No common nodes found across embedding sources")
            # Return empty embeddings
            first_source = next(iter(embeddings.values()))
            return NodeEmbeddings(
                node_ids=[],
                embeddings=np.array([]).reshape(0, self.config.output_dim or 0),
                source=EmbeddingSource.FUSED,
                model_name="fused",
            )

        logger.info(
            f"Fusing {len(embeddings)} embedding sources for {len(node_ids)} nodes "
            f"using {self.config.method.value} method"
        )

        # Extract aligned embeddings for common nodes
        aligned_embeddings = {}
        for source_name, source_emb in embeddings.items():
            embs, found_ids = source_emb.get_batch(node_ids)
            if len(found_ids) != len(node_ids):
                missing = set(node_ids) - set(found_ids)
                logger.warning(
                    f"Source '{source_name}' missing {len(missing)} nodes"
                )
            aligned_embeddings[source_name] = embs
            self._source_dims[source_name] = source_emb.embedding_dim

        # Normalize inputs if configured
        if self.config.normalize_inputs:
            for name in aligned_embeddings:
                embs = aligned_embeddings[name]
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                aligned_embeddings[name] = embs / (norms + 1e-10)

        # Apply fusion method
        if self.config.method == FusionMethod.CONCAT:
            fused = self._fuse_concat(aligned_embeddings)
        elif self.config.method == FusionMethod.WEIGHTED_SUM:
            fused = self._fuse_weighted_sum(aligned_embeddings)
        elif self.config.method == FusionMethod.AVERAGE:
            fused = self._fuse_average(aligned_embeddings)
        elif self.config.method == FusionMethod.PCA:
            fused = self._fuse_pca(aligned_embeddings)
        elif self.config.method == FusionMethod.ATTENTION:
            fused = self._fuse_attention(aligned_embeddings)
        elif self.config.method == FusionMethod.LEARNED:
            fused = self._fuse_learned(aligned_embeddings)
        else:
            raise ValueError(f"Unknown fusion method: {self.config.method}")

        # Normalize output if configured
        if self.config.normalize_output:
            norms = np.linalg.norm(fused, axis=1, keepdims=True)
            fused = fused / (norms + 1e-10)

        return NodeEmbeddings(
            node_ids=node_ids,
            embeddings=fused,
            source=EmbeddingSource.FUSED,
            model_name=f"fused_{self.config.method.value}",
            metadata={
                "fusion_method": self.config.method.value,
                "source_dims": self._source_dims,
                "sources": list(embeddings.keys()),
                "n_nodes": len(node_ids),
            },
        )

    def _fuse_concat(self, aligned_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate all embedding sources."""
        # Sort by source name for reproducibility
        source_names = sorted(aligned_embeddings.keys())
        arrays = [aligned_embeddings[name] for name in source_names]
        return np.concatenate(arrays, axis=1)

    def _fuse_weighted_sum(
        self, aligned_embeddings: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute weighted sum of embeddings."""
        # Get weights
        weights = self.config.weights or {}

        # Check dimensions match
        dims = [emb.shape[1] for emb in aligned_embeddings.values()]
        if len(set(dims)) > 1:
            raise ValueError(
                f"weighted_sum requires same dimensions. Got: {dims}. "
                "Use concat or learned fusion instead."
            )

        # Default to equal weights
        n_sources = len(aligned_embeddings)
        default_weight = 1.0 / n_sources

        result = None
        total_weight = 0.0

        for source_name, embs in aligned_embeddings.items():
            weight = weights.get(source_name, default_weight)
            if result is None:
                result = weight * embs
            else:
                result += weight * embs
            total_weight += weight

        # Normalize by total weight
        return result / total_weight

    def _fuse_average(self, aligned_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple average of embeddings."""
        # Check dimensions match
        dims = [emb.shape[1] for emb in aligned_embeddings.values()]
        if len(set(dims)) > 1:
            raise ValueError(
                f"average requires same dimensions. Got: {dims}. "
                "Use concat fusion instead."
            )

        stacked = np.stack(list(aligned_embeddings.values()), axis=0)
        return np.mean(stacked, axis=0)

    def _fuse_pca(self, aligned_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate and reduce with PCA."""
        from sklearn.decomposition import PCA

        # First concatenate
        concat = self._fuse_concat(aligned_embeddings)

        # Determine number of components
        n_components = self.config.n_components
        if n_components is None:
            # Keep 95% variance or output_dim, whichever is specified
            if self.config.output_dim:
                n_components = self.config.output_dim
            else:
                n_components = 0.95  # Keep 95% variance

        # Fit and transform
        if self._pca_model is None:
            self._pca_model = PCA(n_components=n_components)
            result = self._pca_model.fit_transform(concat)
            logger.info(
                f"PCA reduced from {concat.shape[1]} to {result.shape[1]} dims "
                f"(explained variance: {sum(self._pca_model.explained_variance_ratio_):.2%})"
            )
        else:
            result = self._pca_model.transform(concat)

        return result

    def _fuse_attention(
        self, aligned_embeddings: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Self-attention fusion over embedding sources."""
        # Stack embeddings: (n_nodes, n_sources, dim)
        source_names = sorted(aligned_embeddings.keys())
        arrays = [aligned_embeddings[name] for name in source_names]

        # Pad to same dimension if needed
        max_dim = max(arr.shape[1] for arr in arrays)
        padded = []
        for arr in arrays:
            if arr.shape[1] < max_dim:
                pad_width = ((0, 0), (0, max_dim - arr.shape[1]))
                arr = np.pad(arr, pad_width, mode="constant", constant_values=0)
            padded.append(arr)

        stacked = np.stack(padded, axis=1)  # (n_nodes, n_sources, max_dim)
        n_nodes, n_sources, dim = stacked.shape

        # Compute attention scores
        # Simple dot-product self-attention
        queries = stacked  # (n_nodes, n_sources, dim)
        keys = stacked
        values = stacked

        # Attention weights: softmax(Q @ K^T / sqrt(d))
        scores = np.einsum("nsd,ntd->nst", queries, keys) / np.sqrt(dim)
        attention = self._softmax(scores, axis=-1)  # (n_nodes, n_sources, n_sources)

        # Weighted combination
        attended = np.einsum("nst,ntd->nsd", attention, values)

        # Average over sources
        result = np.mean(attended, axis=1)  # (n_nodes, max_dim)

        # Optionally project to output_dim
        if self.config.output_dim and self.config.output_dim != dim:
            # Simple linear projection
            W = np.random.randn(dim, self.config.output_dim) / np.sqrt(dim)
            result = result @ W

        return result

    def _fuse_learned(self, aligned_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        MLP-based learned fusion.

        Note: For actual training, this would need labels.
        Here we use random weights as a placeholder.
        """
        # Concatenate first
        concat = self._fuse_concat(aligned_embeddings)
        input_dim = concat.shape[1]

        # Determine output dimension
        output_dim = self.config.output_dim or input_dim // 2

        # Build simple MLP (in practice, this would be trained)
        hidden_dims = self.config.hidden_dims

        # Initialize weights (Xavier initialization)
        W1 = np.random.randn(input_dim, hidden_dims[0]) / np.sqrt(input_dim)
        b1 = np.zeros(hidden_dims[0])
        W2 = np.random.randn(hidden_dims[0], output_dim) / np.sqrt(hidden_dims[0])
        b2 = np.zeros(output_dim)

        # Forward pass
        hidden = np.maximum(0, concat @ W1 + b1)  # ReLU
        output = hidden @ W2 + b2

        logger.warning(
            "Learned fusion using random weights. "
            "For actual learned fusion, train with fit_learned() method."
        )

        return output

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax along axis."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def fit_learned(
        self,
        embeddings: Dict[str, NodeEmbeddings],
        labels: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
    ) -> None:
        """
        Train learned fusion weights using labels.

        Args:
            embeddings: Source embeddings
            labels: Target labels for supervised training
            epochs: Training epochs
            learning_rate: Learning rate
        """
        # This would implement actual training
        # For now, placeholder
        logger.info(
            "fit_learned() not fully implemented. "
            "Using default random weights."
        )


def fuse_embeddings(
    embeddings: Dict[str, NodeEmbeddings],
    method: str = "concat",
    output_dim: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
) -> NodeEmbeddings:
    """
    Convenience function to fuse embeddings.

    Args:
        embeddings: Dict mapping source name to NodeEmbeddings
        method: Fusion method ("concat", "weighted_sum", "average", "pca")
        output_dim: Optional output dimension
        weights: Optional weights for weighted_sum

    Returns:
        Fused NodeEmbeddings
    """
    config = FusionConfig(
        method=FusionMethod(method),
        output_dim=output_dim,
        weights=weights,
    )
    fusion = EmbeddingFusion(config)
    return fusion.fuse(embeddings)


def align_embeddings(
    embeddings: Dict[str, NodeEmbeddings],
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Align embeddings to common node IDs.

    Args:
        embeddings: Dict mapping source name to NodeEmbeddings

    Returns:
        Tuple of (common_node_ids, aligned_embeddings_dict)
    """
    # Find common nodes
    node_sets = [set(emb.node_ids) for emb in embeddings.values()]
    common_nodes = sorted(set.intersection(*node_sets))

    # Extract aligned embeddings
    aligned = {}
    for source_name, source_emb in embeddings.items():
        embs, _ = source_emb.get_batch(common_nodes)
        aligned[source_name] = embs

    return common_nodes, aligned


class MultiSourceEmbedder:
    """
    Convenience class to extract and fuse embeddings from multiple sources.

    Example:
        >>> embedder = MultiSourceEmbedder()
        >>> embeddings = embedder.embed(
        ...     gene_ids=["SHANK3", "CHD8"],
        ...     protein_sequences={"SHANK3": "MAE...", "CHD8": "MEP..."},
        ...     descriptions={"SHANK3": "Scaffold protein...", "CHD8": "Chromatin..."},
        ...     fusion_method="concat",
        ... )
    """

    def __init__(
        self,
        use_geneformer: bool = True,
        use_esm2: bool = True,
        use_literature: bool = True,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize multi-source embedder.

        Args:
            use_geneformer: Whether to use Geneformer
            use_esm2: Whether to use ESM-2
            use_literature: Whether to use literature embeddings
            device: Device for inference
            cache_dir: Model cache directory
        """
        self.extractors = {}

        if use_geneformer:
            try:
                from .geneformer import GeneformerExtractor
                self.extractors["geneformer"] = GeneformerExtractor(
                    device=device, cache_dir=cache_dir
                )
            except ImportError:
                from geneformer import GeneformerExtractor
                self.extractors["geneformer"] = GeneformerExtractor(
                    device=device, cache_dir=cache_dir
                )

        if use_esm2:
            try:
                from .esm2 import ESM2Extractor
                self.extractors["esm2"] = ESM2Extractor(
                    device=device, cache_dir=cache_dir
                )
            except ImportError:
                from esm2 import ESM2Extractor
                self.extractors["esm2"] = ESM2Extractor(
                    device=device, cache_dir=cache_dir
                )

        if use_literature:
            try:
                from .literature import LiteratureEmbedder
                self.extractors["literature"] = LiteratureEmbedder(
                    device=device, cache_dir=cache_dir
                )
            except ImportError:
                from literature import LiteratureEmbedder
                self.extractors["literature"] = LiteratureEmbedder(
                    device=device, cache_dir=cache_dir
                )

    def embed(
        self,
        gene_ids: Optional[List[str]] = None,
        protein_sequences: Optional[Dict[str, str]] = None,
        descriptions: Optional[Dict[str, str]] = None,
        fusion_method: str = "concat",
        output_dim: Optional[int] = None,
    ) -> NodeEmbeddings:
        """
        Extract and fuse embeddings from available sources.

        Args:
            gene_ids: Gene IDs for Geneformer
            protein_sequences: Protein sequences for ESM-2
            descriptions: Gene descriptions for literature embedder
            fusion_method: How to fuse embeddings
            output_dim: Optional output dimension

        Returns:
            Fused NodeEmbeddings
        """
        embeddings = {}

        # Geneformer
        if "geneformer" in self.extractors and gene_ids:
            embeddings["geneformer"] = self.extractors["geneformer"].extract(gene_ids)

        # ESM-2
        if "esm2" in self.extractors and protein_sequences:
            embeddings["esm2"] = self.extractors["esm2"].extract(protein_sequences)

        # Literature
        if "literature" in self.extractors and descriptions:
            embeddings["literature"] = self.extractors["literature"].extract_from_descriptions(
                descriptions
            )

        if not embeddings:
            raise ValueError("No embeddings extracted. Provide at least one input type.")

        # Fuse
        return fuse_embeddings(
            embeddings,
            method=fusion_method,
            output_dim=output_dim,
        )
