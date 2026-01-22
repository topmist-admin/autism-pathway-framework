"""
Base classes for graph embedding models.

Provides common interfaces and data structures for knowledge graph embeddings.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import pickle

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NodeEmbeddings:
    """
    Container for node embeddings.

    Stores learned vector representations for nodes in a knowledge graph.
    """

    node_ids: List[str]
    embeddings: np.ndarray  # shape: (n_nodes, embedding_dim)
    embedding_dim: int = field(init=False)
    model_type: str = "unknown"
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
        """Return number of nodes."""
        return len(self.node_ids)

    def get(self, node_id: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a node.

        Args:
            node_id: Node identifier

        Returns:
            Embedding vector or None if not found
        """
        idx = self._id_to_idx.get(node_id)
        if idx is None:
            return None
        return self.embeddings[idx].copy()

    def get_batch(self, node_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Get embeddings for multiple nodes.

        Args:
            node_ids: List of node identifiers

        Returns:
            Tuple of (embeddings array, list of found node_ids)
        """
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
        """
        Find most similar nodes by cosine similarity.

        Args:
            node_id: Query node
            k: Number of results to return
            exclude_self: Whether to exclude the query node from results

        Returns:
            List of (node_id, similarity_score) tuples, sorted by similarity
        """
        query_emb = self.get(node_id)
        if query_emb is None:
            return []

        # Normalize for cosine similarity
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        emb_norms = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        )

        # Compute similarities
        similarities = emb_norms @ query_norm

        # Get top-k indices
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
        """
        Compute cosine similarity between two nodes.

        Args:
            node_id1: First node
            node_id2: Second node

        Returns:
            Cosine similarity or None if either node not found
        """
        emb1 = self.get(node_id1)
        emb2 = self.get(node_id2)

        if emb1 is None or emb2 is None:
            return None

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def save(self, path: str) -> None:
        """
        Save embeddings to file.

        Args:
            path: Output path (.npz or .pkl)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".npz":
            np.savez(
                path,
                node_ids=np.array(self.node_ids, dtype=object),
                embeddings=self.embeddings,
                model_type=self.model_type,
                metadata=json.dumps(self.metadata),
            )
        elif path.suffix == ".pkl":
            with open(path, "wb") as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        logger.info(f"Saved embeddings to {path}")

    @classmethod
    def load(cls, path: str) -> "NodeEmbeddings":
        """
        Load embeddings from file.

        Args:
            path: Input path

        Returns:
            NodeEmbeddings instance
        """
        path = Path(path)

        if path.suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            return cls(
                node_ids=list(data["node_ids"]),
                embeddings=data["embeddings"],
                model_type=str(data["model_type"]),
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
class RelationEmbeddings:
    """
    Container for relation (edge type) embeddings.

    Used by TransE, RotatE, and similar models.
    """

    relation_ids: List[str]
    embeddings: np.ndarray  # shape: (n_relations, embedding_dim)
    embedding_dim: int = field(init=False)

    def __post_init__(self):
        """Validate and set derived fields."""
        if len(self.relation_ids) != self.embeddings.shape[0]:
            raise ValueError(
                f"Number of relation_ids ({len(self.relation_ids)}) must match "
                f"number of embeddings ({self.embeddings.shape[0]})"
            )
        self.embedding_dim = self.embeddings.shape[1]
        self._id_to_idx = {rel_id: idx for idx, rel_id in enumerate(self.relation_ids)}

    def get(self, relation_id: str) -> Optional[np.ndarray]:
        """Get embedding for a relation."""
        idx = self._id_to_idx.get(relation_id)
        if idx is None:
            return None
        return self.embeddings[idx].copy()


@dataclass
class TrainingHistory:
    """Training history for embedding models."""

    epochs: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def add_epoch(
        self,
        epoch: int,
        loss: float,
        **kwargs: float,
    ) -> None:
        """Record metrics for an epoch."""
        self.epochs.append(epoch)
        self.losses.append(loss)
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    @property
    def best_epoch(self) -> int:
        """Return epoch with lowest loss."""
        if not self.losses:
            return 0
        return self.epochs[np.argmin(self.losses)]

    @property
    def final_loss(self) -> float:
        """Return final training loss."""
        return self.losses[-1] if self.losses else float("inf")


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for embedding models."""

    # Link prediction metrics
    mean_rank: float = 0.0
    mean_reciprocal_rank: float = 0.0
    hits_at_1: float = 0.0
    hits_at_3: float = 0.0
    hits_at_10: float = 0.0

    # Additional metrics
    num_test_triples: int = 0
    filtered: bool = True  # Whether filtered setting was used

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "mean_rank": self.mean_rank,
            "mrr": self.mean_reciprocal_rank,
            "hits@1": self.hits_at_1,
            "hits@3": self.hits_at_3,
            "hits@10": self.hits_at_10,
            "num_test_triples": self.num_test_triples,
        }


class BaseEmbeddingModel(ABC):
    """
    Abstract base class for knowledge graph embedding models.

    Defines the interface that all embedding models must implement.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        learning_rate: float = 0.01,
        random_state: Optional[int] = None,
    ):
        """
        Initialize base model.

        Args:
            embedding_dim: Dimension of embeddings
            learning_rate: Learning rate for optimization
            random_state: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        self._node_embeddings: Optional[np.ndarray] = None
        self._relation_embeddings: Optional[np.ndarray] = None
        self._node_to_idx: Dict[str, int] = {}
        self._relation_to_idx: Dict[str, int] = {}
        self._idx_to_node: Dict[int, str] = {}
        self._idx_to_relation: Dict[int, str] = {}
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the model."""
        return len(self._node_to_idx)

    @property
    def n_relations(self) -> int:
        """Number of relation types in the model."""
        return len(self._relation_to_idx)

    @abstractmethod
    def _score_triple(
        self,
        head_idx: int,
        relation_idx: int,
        tail_idx: int,
    ) -> float:
        """
        Compute score for a single triple.

        Lower scores indicate more plausible triples.
        """
        pass

    @abstractmethod
    def _compute_loss(
        self,
        positive_triples: np.ndarray,
        negative_triples: np.ndarray,
    ) -> float:
        """Compute training loss for a batch."""
        pass

    @abstractmethod
    def _update_embeddings(
        self,
        positive_triples: np.ndarray,
        negative_triples: np.ndarray,
    ) -> None:
        """Update embeddings based on gradients."""
        pass

    def _initialize_embeddings(
        self,
        n_nodes: int,
        n_relations: int,
    ) -> None:
        """Initialize embedding matrices."""
        # Xavier/Glorot initialization
        scale = np.sqrt(6.0 / (n_nodes + self.embedding_dim))
        self._node_embeddings = np.random.uniform(
            -scale, scale, (n_nodes, self.embedding_dim)
        )

        scale = np.sqrt(6.0 / (n_relations + self.embedding_dim))
        self._relation_embeddings = np.random.uniform(
            -scale, scale, (n_relations, self.embedding_dim)
        )

    def _build_vocabulary(self, graph: Any) -> Tuple[List[str], List[str]]:
        """
        Build node and relation vocabularies from a knowledge graph.

        Args:
            graph: KnowledgeGraph instance

        Returns:
            Tuple of (node_ids, relation_ids)
        """
        # Get all nodes
        node_ids = list(graph.graph.nodes())

        # Get all relation types
        relation_ids = set()
        for _, _, data in graph.graph.edges(data=True):
            rel_type = data.get("edge_type", "unknown")
            relation_ids.add(rel_type)
        relation_ids = sorted(relation_ids)

        # Build mappings
        self._node_to_idx = {node: idx for idx, node in enumerate(node_ids)}
        self._idx_to_node = {idx: node for node, idx in self._node_to_idx.items()}
        self._relation_to_idx = {rel: idx for idx, rel in enumerate(relation_ids)}
        self._idx_to_relation = {idx: rel for rel, idx in self._relation_to_idx.items()}

        return node_ids, relation_ids

    def _get_triples(self, graph: Any) -> np.ndarray:
        """
        Extract triples from knowledge graph.

        Args:
            graph: KnowledgeGraph instance

        Returns:
            Array of shape (n_triples, 3) with [head_idx, relation_idx, tail_idx]
        """
        triples = []
        for source, target, data in graph.graph.edges(data=True):
            rel_type = data.get("edge_type", "unknown")
            head_idx = self._node_to_idx.get(source)
            tail_idx = self._node_to_idx.get(target)
            rel_idx = self._relation_to_idx.get(rel_type)

            if head_idx is not None and tail_idx is not None and rel_idx is not None:
                triples.append([head_idx, rel_idx, tail_idx])

        return np.array(triples, dtype=np.int64)

    def _generate_negative_samples(
        self,
        positive_triples: np.ndarray,
        n_negative: int = 1,
    ) -> np.ndarray:
        """
        Generate negative samples by corrupting positive triples.

        Args:
            positive_triples: Array of positive triples
            n_negative: Number of negative samples per positive

        Returns:
            Array of negative triples
        """
        n_nodes = self.n_nodes
        negative_triples = []

        for triple in positive_triples:
            head, rel, tail = triple

            for _ in range(n_negative):
                # Randomly corrupt head or tail
                if np.random.random() < 0.5:
                    # Corrupt head
                    new_head = np.random.randint(n_nodes)
                    while new_head == head:
                        new_head = np.random.randint(n_nodes)
                    negative_triples.append([new_head, rel, tail])
                else:
                    # Corrupt tail
                    new_tail = np.random.randint(n_nodes)
                    while new_tail == tail:
                        new_tail = np.random.randint(n_nodes)
                    negative_triples.append([head, rel, new_tail])

        return np.array(negative_triples, dtype=np.int64)

    def train(
        self,
        graph: Any,
        epochs: int = 100,
        batch_size: int = 256,
        n_negative: int = 1,
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Train the embedding model on a knowledge graph.

        Args:
            graph: KnowledgeGraph instance from Module 03
            epochs: Number of training epochs
            batch_size: Batch size for training
            n_negative: Number of negative samples per positive
            verbose: Whether to print progress

        Returns:
            TrainingHistory with loss per epoch
        """
        # Build vocabulary and initialize
        node_ids, relation_ids = self._build_vocabulary(graph)
        self._initialize_embeddings(len(node_ids), len(relation_ids))

        # Get training triples
        triples = self._get_triples(graph)
        n_triples = len(triples)

        if n_triples == 0:
            raise ValueError("No triples found in knowledge graph")

        logger.info(
            f"Training {self.__class__.__name__} with {self.n_nodes} nodes, "
            f"{self.n_relations} relations, {n_triples} triples"
        )

        history = TrainingHistory()

        for epoch in range(epochs):
            # Shuffle triples
            np.random.shuffle(triples)

            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, n_triples, batch_size):
                batch_end = min(batch_start + batch_size, n_triples)
                positive_batch = triples[batch_start:batch_end]

                # Generate negative samples
                negative_batch = self._generate_negative_samples(
                    positive_batch, n_negative
                )

                # Compute loss and update
                batch_loss = self._compute_loss(positive_batch, negative_batch)
                self._update_embeddings(positive_batch, negative_batch)

                epoch_loss += batch_loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            history.add_epoch(epoch, avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self._is_trained = True
        logger.info(f"Training complete. Final loss: {history.final_loss:.4f}")

        return history

    def get_node_embeddings(self) -> NodeEmbeddings:
        """
        Get trained node embeddings.

        Returns:
            NodeEmbeddings instance
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before getting embeddings")

        node_ids = [self._idx_to_node[i] for i in range(self.n_nodes)]

        return NodeEmbeddings(
            node_ids=node_ids,
            embeddings=self._node_embeddings.copy(),
            model_type=self.__class__.__name__,
            metadata={
                "embedding_dim": self.embedding_dim,
                "n_relations": self.n_relations,
            },
        )

    def get_relation_embeddings(self) -> RelationEmbeddings:
        """
        Get trained relation embeddings.

        Returns:
            RelationEmbeddings instance
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before getting embeddings")

        relation_ids = [self._idx_to_relation[i] for i in range(self.n_relations)]

        return RelationEmbeddings(
            relation_ids=relation_ids,
            embeddings=self._relation_embeddings.copy(),
        )

    def predict_link(
        self,
        head: str,
        relation: str,
        tail: str,
    ) -> Optional[float]:
        """
        Predict plausibility score for a triple.

        Args:
            head: Head node ID
            relation: Relation type
            tail: Tail node ID

        Returns:
            Score (lower is more plausible) or None if nodes/relation not found
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")

        head_idx = self._node_to_idx.get(head)
        rel_idx = self._relation_to_idx.get(relation)
        tail_idx = self._node_to_idx.get(tail)

        if head_idx is None or rel_idx is None or tail_idx is None:
            return None

        return self._score_triple(head_idx, rel_idx, tail_idx)

    def predict_tail(
        self,
        head: str,
        relation: str,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Predict most likely tail nodes for a (head, relation, ?) query.

        Args:
            head: Head node ID
            relation: Relation type
            k: Number of predictions to return

        Returns:
            List of (tail_id, score) tuples, sorted by score (ascending)
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")

        head_idx = self._node_to_idx.get(head)
        rel_idx = self._relation_to_idx.get(relation)

        if head_idx is None or rel_idx is None:
            return []

        # Score all possible tails
        scores = []
        for tail_idx in range(self.n_nodes):
            score = self._score_triple(head_idx, rel_idx, tail_idx)
            scores.append((tail_idx, score))

        # Sort by score (ascending - lower is better)
        scores.sort(key=lambda x: x[1])

        return [
            (self._idx_to_node[tail_idx], score)
            for tail_idx, score in scores[:k]
        ]

    def save(self, path: str) -> None:
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str) -> "BaseEmbeddingModel":
        """Load model from file."""
        with open(path, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Loaded model from {path}")
        return model
