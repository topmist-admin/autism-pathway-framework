"""
Base classes and data structures for pretrained embedding extractors.

Provides common interfaces for extracting embeddings from foundation models
like Geneformer, ESM-2, and PubMedBERT.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import pickle

import numpy as np

logger = logging.getLogger(__name__)


class ExtractionMode(Enum):
    """Mode for embedding extraction."""

    FROZEN = "frozen"  # Use pretrained weights as-is
    FINE_TUNED = "fine_tuned"  # Fine-tune on domain-specific data


class EmbeddingSource(Enum):
    """Source of embeddings."""

    GENEFORMER = "geneformer"
    ESM2 = "esm2"
    PUBMEDBERT = "pubmedbert"
    BIOGPT = "biogpt"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    FUSED = "fused"


@dataclass
class NodeEmbeddings:
    """
    Container for node embeddings from pretrained models.

    Compatible with Module 04's NodeEmbeddings but with additional
    metadata for tracking embedding source and extraction parameters.
    """

    node_ids: List[str]
    embeddings: np.ndarray  # shape: (n_nodes, embedding_dim)
    embedding_dim: int = field(init=False)
    source: EmbeddingSource = EmbeddingSource.GENEFORMER
    model_name: str = ""
    extraction_mode: ExtractionMode = ExtractionMode.FROZEN
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and set derived fields."""
        if len(self.node_ids) != self.embeddings.shape[0]:
            raise ValueError(
                f"Number of node_ids ({len(self.node_ids)}) must match "
                f"number of embeddings ({self.embeddings.shape[0]})"
            )
        self.embedding_dim = self.embeddings.shape[1]
        self._id_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_ids)}

    def __len__(self) -> int:
        return len(self.node_ids)

    def get(self, node_id: str) -> Optional[np.ndarray]:
        """Get embedding for a node."""
        idx = self._id_to_idx.get(node_id)
        if idx is None:
            return None
        return self.embeddings[idx].copy()

    def get_batch(self, node_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Get embeddings for multiple nodes."""
        found_ids = []
        found_embeddings = []

        for node_id in node_ids:
            emb = self.get(node_id)
            if emb is not None:
                found_ids.append(node_id)
                found_embeddings.append(emb)

        if not found_embeddings:
            return np.array([]).reshape(0, self.embedding_dim), []

        return np.stack(found_embeddings), found_ids

    def most_similar(
        self,
        node_id: str,
        k: int = 10,
        exclude_self: bool = True,
    ) -> List[Tuple[str, float]]:
        """Find most similar nodes by cosine similarity."""
        query_emb = self.get(node_id)
        if query_emb is None:
            return []

        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        emb_norms = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        )

        similarities = emb_norms @ query_norm

        if exclude_self:
            query_idx = self._id_to_idx[node_id]
            similarities[query_idx] = -np.inf

        top_k_idx = np.argsort(similarities)[-k:][::-1]

        return [
            (self.node_ids[idx], float(similarities[idx]))
            for idx in top_k_idx
            if similarities[idx] > -np.inf
        ]

    def compute_similarity(self, node_id1: str, node_id2: str) -> Optional[float]:
        """Compute cosine similarity between two nodes."""
        emb1 = self.get(node_id1)
        emb2 = self.get(node_id2)

        if emb1 is None or emb2 is None:
            return None

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def subset(self, node_ids: List[str]) -> "NodeEmbeddings":
        """Create a subset with only specified nodes."""
        found_ids = []
        found_embeddings = []

        for node_id in node_ids:
            emb = self.get(node_id)
            if emb is not None:
                found_ids.append(node_id)
                found_embeddings.append(emb)

        if not found_embeddings:
            return NodeEmbeddings(
                node_ids=[],
                embeddings=np.array([]).reshape(0, self.embedding_dim),
                source=self.source,
                model_name=self.model_name,
                extraction_mode=self.extraction_mode,
            )

        return NodeEmbeddings(
            node_ids=found_ids,
            embeddings=np.stack(found_embeddings),
            source=self.source,
            model_name=self.model_name,
            extraction_mode=self.extraction_mode,
            metadata={**self.metadata, "subset_of": len(self.node_ids)},
        )

    def save(self, path: str) -> None:
        """Save embeddings to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".npz":
            np.savez(
                path,
                node_ids=np.array(self.node_ids, dtype=object),
                embeddings=self.embeddings,
                source=self.source.value,
                model_name=self.model_name,
                extraction_mode=self.extraction_mode.value,
                metadata=json.dumps(self.metadata),
            )
        elif path.suffix == ".pkl":
            with open(path, "wb") as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        logger.info(f"Saved {len(self)} embeddings to {path}")

    @classmethod
    def load(cls, path: str) -> "NodeEmbeddings":
        """Load embeddings from file."""
        path = Path(path)

        if path.suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            return cls(
                node_ids=list(data["node_ids"]),
                embeddings=data["embeddings"],
                source=EmbeddingSource(str(data["source"])),
                model_name=str(data["model_name"]),
                extraction_mode=ExtractionMode(str(data["extraction_mode"])),
                metadata=json.loads(str(data["metadata"])),
            )
        elif path.suffix == ".pkl":
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary mapping node_id to embedding."""
        return {
            node_id: self.embeddings[idx]
            for node_id, idx in self._id_to_idx.items()
        }


@dataclass
class VariantEffect:
    """Predicted effect of a genetic variant on protein embeddings."""

    gene_id: str
    variant_id: str  # e.g., "p.R123W"
    position: int
    ref_aa: str
    alt_aa: str
    pathogenicity_score: float  # 0-1, higher = more pathogenic
    embedding_shift: float  # Magnitude of embedding change
    log_likelihood_ratio: float  # ESM-2 LLR score
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gene_id": self.gene_id,
            "variant_id": self.variant_id,
            "position": self.position,
            "ref_aa": self.ref_aa,
            "alt_aa": self.alt_aa,
            "pathogenicity_score": self.pathogenicity_score,
            "embedding_shift": self.embedding_shift,
            "log_likelihood_ratio": self.log_likelihood_ratio,
            "confidence": self.confidence,
        }


@dataclass
class FineTuneConfig:
    """Configuration for foundation model fine-tuning."""

    learning_rate: float = 1e-5
    epochs: int = 10
    batch_size: int = 32
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_length: int = 512

    # Multi-task fine-tuning
    tasks: List[str] = field(default_factory=list)
    task_weights: Dict[str, float] = field(default_factory=dict)

    # Adapter-based fine-tuning (parameter efficient)
    use_adapters: bool = True
    adapter_dim: int = 64

    # Early stopping
    patience: int = 3
    min_delta: float = 0.001

    # Device
    device: str = "cuda"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "max_length": self.max_length,
            "tasks": self.tasks,
            "task_weights": self.task_weights,
            "use_adapters": self.use_adapters,
            "adapter_dim": self.adapter_dim,
            "device": self.device,
        }


class BaseEmbeddingExtractor(ABC):
    """
    Abstract base class for pretrained embedding extractors.

    All extractors must implement the extract() method.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize extractor.

        Args:
            model_name: Name or path of the pretrained model
            device: Device for inference ("cpu" or "cuda")
            cache_dir: Directory for caching downloaded models
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @abstractmethod
    def _load_model(self) -> None:
        """Load the pretrained model and tokenizer."""
        pass

    def ensure_loaded(self) -> None:
        """Ensure model is loaded before extraction."""
        if not self._is_loaded:
            self._load_model()
            self._is_loaded = True

    @abstractmethod
    def extract(
        self,
        inputs: Union[List[str], Dict[str, str]],
        **kwargs: Any,
    ) -> NodeEmbeddings:
        """
        Extract embeddings from inputs.

        Args:
            inputs: Gene IDs, protein sequences, or text descriptions
            **kwargs: Additional extraction parameters

        Returns:
            NodeEmbeddings instance
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension of the model."""
        pass

    @property
    @abstractmethod
    def source(self) -> EmbeddingSource:
        """Return the embedding source type."""
        pass


def normalize_gene_id(gene_id: str) -> str:
    """
    Normalize gene identifier to standard format.

    Handles common variations:
    - SHANK3, shank3, Shank3 -> SHANK3
    - ENSG00000148498 -> ENSG00000148498
    """
    gene_id = gene_id.strip()

    # If it looks like an Ensembl ID, keep as-is
    if gene_id.startswith("ENSG") or gene_id.startswith("ENSP"):
        return gene_id

    # Otherwise, uppercase (gene symbols)
    return gene_id.upper()


def batch_generator(
    items: List[Any],
    batch_size: int,
) -> List[List[Any]]:
    """Generate batches from a list of items."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
