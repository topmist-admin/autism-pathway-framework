"""
TransE: Translating Embeddings for Modeling Multi-relational Data

TransE models relationships as translations in the embedding space.
For a valid triple (h, r, t), the model learns embeddings such that: h + r ≈ t

Reference:
    Bordes et al. (2013). "Translating Embeddings for Modeling Multi-relational Data"
    https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
"""

from typing import Any, Optional
import logging

import numpy as np

try:
    from .base import BaseEmbeddingModel, TrainingHistory
except ImportError:
    from base import BaseEmbeddingModel, TrainingHistory

logger = logging.getLogger(__name__)


class TransEModel(BaseEmbeddingModel):
    """
    TransE knowledge graph embedding model.

    TransE represents entities as points in a low-dimensional embedding space
    and relations as translation vectors. For a valid triple (head, relation, tail),
    the embedding of the tail should be close to the embedding of the head plus
    the relation vector: head + relation ≈ tail.

    The model is trained using a margin-based ranking loss that pushes the score
    of positive triples below that of negative (corrupted) triples.

    Example:
        >>> from modules.03_knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder
        >>> # Build a simple knowledge graph
        >>> kg = KnowledgeGraphBuilder().add_genes(["A", "B", "C"]).build()
        >>> kg.add_edge("A", "B", EdgeType.GENE_INTERACTS)
        >>>
        >>> # Train TransE
        >>> model = TransEModel(embedding_dim=64, margin=1.0)
        >>> history = model.train(kg, epochs=100)
        >>>
        >>> # Get embeddings
        >>> embeddings = model.get_node_embeddings()
        >>> emb_a = embeddings.get("A")
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        margin: float = 1.0,
        norm: int = 1,
        learning_rate: float = 0.01,
        random_state: Optional[int] = None,
    ):
        """
        Initialize TransE model.

        Args:
            embedding_dim: Dimension of entity and relation embeddings
            margin: Margin for the ranking loss (gamma in the paper)
            norm: L1 (1) or L2 (2) norm for distance calculation
            learning_rate: Learning rate for SGD
            random_state: Random seed for reproducibility
        """
        super().__init__(
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.margin = margin
        self.norm = norm

        if norm not in (1, 2):
            raise ValueError(f"norm must be 1 or 2, got {norm}")

    def _normalize_embeddings(self) -> None:
        """Normalize entity embeddings to unit length."""
        norms = np.linalg.norm(self._node_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        self._node_embeddings = self._node_embeddings / norms

    def _initialize_embeddings(self, n_nodes: int, n_relations: int) -> None:
        """Initialize embeddings with normalization."""
        super()._initialize_embeddings(n_nodes, n_relations)
        # Normalize entity embeddings
        self._normalize_embeddings()

    def _distance(
        self,
        head_emb: np.ndarray,
        relation_emb: np.ndarray,
        tail_emb: np.ndarray,
    ) -> np.ndarray:
        """
        Compute distance score for triples.

        TransE score: ||h + r - t||

        Args:
            head_emb: Head entity embeddings
            relation_emb: Relation embeddings
            tail_emb: Tail entity embeddings

        Returns:
            Distance scores (lower is better)
        """
        diff = head_emb + relation_emb - tail_emb

        if self.norm == 1:
            return np.sum(np.abs(diff), axis=-1)
        else:  # L2 norm
            return np.sqrt(np.sum(diff ** 2, axis=-1))

    def _score_triple(
        self,
        head_idx: int,
        relation_idx: int,
        tail_idx: int,
    ) -> float:
        """
        Compute score for a single triple.

        Args:
            head_idx: Index of head entity
            relation_idx: Index of relation
            tail_idx: Index of tail entity

        Returns:
            Distance score (lower indicates more plausible triple)
        """
        head_emb = self._node_embeddings[head_idx]
        relation_emb = self._relation_embeddings[relation_idx]
        tail_emb = self._node_embeddings[tail_idx]

        return float(self._distance(head_emb, relation_emb, tail_emb))

    def _compute_loss(
        self,
        positive_triples: np.ndarray,
        negative_triples: np.ndarray,
    ) -> float:
        """
        Compute margin-based ranking loss.

        Loss = max(0, margin + d(h, r, t) - d(h', r, t'))

        where (h, r, t) is a positive triple and (h', r, t') is negative.

        Args:
            positive_triples: Array of positive triples [head, rel, tail]
            negative_triples: Array of negative triples

        Returns:
            Total loss for the batch
        """
        # Get embeddings for positive triples
        pos_heads = self._node_embeddings[positive_triples[:, 0]]
        pos_relations = self._relation_embeddings[positive_triples[:, 1]]
        pos_tails = self._node_embeddings[positive_triples[:, 2]]

        # Get embeddings for negative triples
        neg_heads = self._node_embeddings[negative_triples[:, 0]]
        neg_relations = self._relation_embeddings[negative_triples[:, 1]]
        neg_tails = self._node_embeddings[negative_triples[:, 2]]

        # Compute distances
        pos_distances = self._distance(pos_heads, pos_relations, pos_tails)
        neg_distances = self._distance(neg_heads, neg_relations, neg_tails)

        # Margin-based ranking loss
        # Note: negative triples may be more than positive if n_negative > 1
        # We handle this by repeating positive distances
        if len(neg_distances) > len(pos_distances):
            n_neg_per_pos = len(neg_distances) // len(pos_distances)
            pos_distances = np.repeat(pos_distances, n_neg_per_pos)

        loss = np.maximum(0, self.margin + pos_distances - neg_distances)
        return float(np.mean(loss))

    def _update_embeddings(
        self,
        positive_triples: np.ndarray,
        negative_triples: np.ndarray,
    ) -> None:
        """
        Update embeddings using gradient descent.

        For TransE, the gradient of the loss with respect to embeddings is:
        - For positive triples: push h + r closer to t
        - For negative triples: push h + r away from t

        Args:
            positive_triples: Array of positive triples
            negative_triples: Array of negative triples
        """
        lr = self.learning_rate

        # Process positive triples
        for triple in positive_triples:
            head_idx, rel_idx, tail_idx = triple

            head_emb = self._node_embeddings[head_idx]
            rel_emb = self._relation_embeddings[rel_idx]
            tail_emb = self._node_embeddings[tail_idx]

            # Gradient direction: h + r - t
            diff = head_emb + rel_emb - tail_emb

            if self.norm == 1:
                grad = np.sign(diff)
            else:
                norm = np.linalg.norm(diff) + 1e-10
                grad = diff / norm

            # Update to minimize distance for positive triples
            self._node_embeddings[head_idx] -= lr * grad
            self._relation_embeddings[rel_idx] -= lr * grad
            self._node_embeddings[tail_idx] += lr * grad

        # Process negative triples
        for triple in negative_triples:
            head_idx, rel_idx, tail_idx = triple

            head_emb = self._node_embeddings[head_idx]
            rel_emb = self._relation_embeddings[rel_idx]
            tail_emb = self._node_embeddings[tail_idx]

            diff = head_emb + rel_emb - tail_emb

            if self.norm == 1:
                grad = np.sign(diff)
            else:
                norm = np.linalg.norm(diff) + 1e-10
                grad = diff / norm

            # Update to maximize distance for negative triples
            self._node_embeddings[head_idx] += lr * grad
            self._relation_embeddings[rel_idx] += lr * grad
            self._node_embeddings[tail_idx] -= lr * grad

        # Re-normalize entity embeddings
        self._normalize_embeddings()

    def train(
        self,
        graph: Any,
        epochs: int = 100,
        batch_size: int = 256,
        n_negative: int = 1,
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Train TransE model on a knowledge graph.

        Args:
            graph: KnowledgeGraph instance from Module 03
            epochs: Number of training epochs
            batch_size: Batch size for training
            n_negative: Number of negative samples per positive triple
            verbose: Whether to print progress

        Returns:
            TrainingHistory with loss per epoch
        """
        logger.info(
            f"Training TransE: dim={self.embedding_dim}, margin={self.margin}, "
            f"norm=L{self.norm}, lr={self.learning_rate}"
        )
        return super().train(
            graph=graph,
            epochs=epochs,
            batch_size=batch_size,
            n_negative=n_negative,
            verbose=verbose,
        )


def create_transe_model(
    embedding_dim: int = 128,
    margin: float = 1.0,
    norm: int = 1,
    learning_rate: float = 0.01,
    random_state: Optional[int] = None,
) -> TransEModel:
    """
    Factory function to create a TransE model with recommended defaults.

    Args:
        embedding_dim: Dimension of embeddings (default: 128)
        margin: Margin for ranking loss (default: 1.0)
        norm: L1 or L2 norm (default: L1)
        learning_rate: Learning rate (default: 0.01)
        random_state: Random seed

    Returns:
        TransEModel instance
    """
    return TransEModel(
        embedding_dim=embedding_dim,
        margin=margin,
        norm=norm,
        learning_rate=learning_rate,
        random_state=random_state,
    )
