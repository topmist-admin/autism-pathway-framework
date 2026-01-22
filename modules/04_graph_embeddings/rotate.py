"""
RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space

RotatE models relations as rotations in complex space, allowing it to capture
various relation patterns including symmetry, antisymmetry, inversion, and composition.

For a valid triple (h, r, t): t = h ∘ r (element-wise Hadamard product in complex space)

Reference:
    Sun et al. (2019). "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space"
    https://arxiv.org/abs/1902.10197
"""

from typing import Any, Optional
import logging

import numpy as np

try:
    from .base import BaseEmbeddingModel, TrainingHistory
except ImportError:
    from base import BaseEmbeddingModel, TrainingHistory

logger = logging.getLogger(__name__)


class RotatEModel(BaseEmbeddingModel):
    """
    RotatE knowledge graph embedding model.

    RotatE represents entities as complex vectors and relations as rotations
    in complex space. For a valid triple (head, relation, tail), the tail
    embedding should be close to the head embedding rotated by the relation:
    tail ≈ head ∘ relation

    The relation is constrained to have unit modulus, so it represents a pure
    rotation in the complex plane.

    Advantages over TransE:
    - Can model symmetric relations: r ∘ r = 1
    - Can model antisymmetric relations: r ≠ r^(-1)
    - Can model inverse relations: r2 = r1^(-1)
    - Can model composition: r3 = r1 ∘ r2

    Example:
        >>> model = RotatEModel(embedding_dim=64, margin=6.0)
        >>> history = model.train(kg, epochs=100)
        >>> embeddings = model.get_node_embeddings()
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        margin: float = 6.0,
        learning_rate: float = 0.001,
        adversarial_temperature: float = 1.0,
        random_state: Optional[int] = None,
    ):
        """
        Initialize RotatE model.

        Args:
            embedding_dim: Dimension of embeddings (must be even for complex representation)
            margin: Margin for the ranking loss (gamma)
            learning_rate: Learning rate for SGD
            adversarial_temperature: Temperature for self-adversarial sampling
            random_state: Random seed for reproducibility
        """
        # Ensure embedding_dim is even (for real + imaginary parts)
        if embedding_dim % 2 != 0:
            embedding_dim = embedding_dim + 1
            logger.warning(
                f"RotatE requires even embedding_dim. Adjusted to {embedding_dim}"
            )

        super().__init__(
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.margin = margin
        self.adversarial_temperature = adversarial_temperature

        # Complex dimension is half of embedding_dim
        self._complex_dim = embedding_dim // 2

    def _initialize_embeddings(self, n_nodes: int, n_relations: int) -> None:
        """
        Initialize entity and relation embeddings.

        Entity embeddings: uniform in [-embedding_range, embedding_range]
        Relation embeddings: phase angles in [-pi, pi]
        """
        embedding_range = (self.margin + 2.0) / self.embedding_dim

        # Entity embeddings (real and imaginary parts concatenated)
        self._node_embeddings = np.random.uniform(
            -embedding_range, embedding_range, (n_nodes, self.embedding_dim)
        )

        # Relation embeddings as phases (angles)
        # Store as [cos(theta), sin(theta)] for each dimension
        phases = np.random.uniform(-np.pi, np.pi, (n_relations, self._complex_dim))
        self._relation_phases = phases

        # Convert to complex representation
        self._relation_embeddings = np.zeros((n_relations, self.embedding_dim))
        self._relation_embeddings[:, :self._complex_dim] = np.cos(phases)
        self._relation_embeddings[:, self._complex_dim:] = np.sin(phases)

    def _to_complex(self, embeddings: np.ndarray) -> tuple:
        """
        Split embeddings into real and imaginary parts.

        Args:
            embeddings: Array of shape (..., embedding_dim)

        Returns:
            Tuple of (real_part, imag_part) each of shape (..., complex_dim)
        """
        real = embeddings[..., :self._complex_dim]
        imag = embeddings[..., self._complex_dim:]
        return real, imag

    def _complex_multiply(
        self,
        re1: np.ndarray,
        im1: np.ndarray,
        re2: np.ndarray,
        im2: np.ndarray,
    ) -> tuple:
        """
        Perform complex multiplication: (re1 + i*im1) * (re2 + i*im2).

        Args:
            re1, im1: Real and imaginary parts of first operand
            re2, im2: Real and imaginary parts of second operand

        Returns:
            Tuple of (real_result, imag_result)
        """
        real = re1 * re2 - im1 * im2
        imag = re1 * im2 + im1 * re2
        return real, imag

    def _distance(
        self,
        head_emb: np.ndarray,
        relation_emb: np.ndarray,
        tail_emb: np.ndarray,
    ) -> np.ndarray:
        """
        Compute RotatE distance score.

        Score: ||h ∘ r - t||

        where ∘ is element-wise complex multiplication (rotation).

        Args:
            head_emb: Head entity embeddings
            relation_emb: Relation embeddings (rotation vectors)
            tail_emb: Tail entity embeddings

        Returns:
            Distance scores (lower is better)
        """
        # Split into real and imaginary parts
        head_re, head_im = self._to_complex(head_emb)
        rel_re, rel_im = self._to_complex(relation_emb)
        tail_re, tail_im = self._to_complex(tail_emb)

        # Rotate head by relation
        rotated_re, rotated_im = self._complex_multiply(
            head_re, head_im, rel_re, rel_im
        )

        # Compute distance to tail
        diff_re = rotated_re - tail_re
        diff_im = rotated_im - tail_im

        # L2 distance in complex space
        distance = np.sqrt(np.sum(diff_re ** 2 + diff_im ** 2, axis=-1))
        return distance

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
        Compute self-adversarial negative sampling loss.

        Loss = -log(sigmoid(gamma - d(h, r, t)))
               - sum_i p_i * log(sigmoid(d(h'_i, r, t'_i) - gamma))

        where p_i is self-adversarial weight.

        Args:
            positive_triples: Array of positive triples
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

        # Positive loss: -log(sigmoid(gamma - d_pos))
        pos_scores = self.margin - pos_distances
        pos_loss = -np.mean(self._log_sigmoid(pos_scores))

        # Negative loss with self-adversarial weights
        neg_scores = neg_distances - self.margin
        if self.adversarial_temperature > 0:
            # Self-adversarial sampling weights
            weights = self._softmax(neg_distances * self.adversarial_temperature)
            neg_loss = -np.sum(weights * self._log_sigmoid(neg_scores))
        else:
            neg_loss = -np.mean(self._log_sigmoid(neg_scores))

        return float(pos_loss + neg_loss)

    def _log_sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable log-sigmoid."""
        return np.where(
            x >= 0,
            -np.log1p(np.exp(-x)),
            x - np.log1p(np.exp(x)),
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-10)

    def _update_embeddings(
        self,
        positive_triples: np.ndarray,
        negative_triples: np.ndarray,
    ) -> None:
        """
        Update embeddings using gradient descent.

        For RotatE, we need to update:
        - Entity embeddings (real and imaginary parts)
        - Relation phases (keeping unit modulus constraint)

        Args:
            positive_triples: Array of positive triples
            negative_triples: Array of negative triples
        """
        lr = self.learning_rate

        # Update based on positive triples
        for triple in positive_triples:
            head_idx, rel_idx, tail_idx = triple

            head_emb = self._node_embeddings[head_idx]
            rel_emb = self._relation_embeddings[rel_idx]
            tail_emb = self._node_embeddings[tail_idx]

            # Compute gradient direction
            head_re, head_im = self._to_complex(head_emb)
            rel_re, rel_im = self._to_complex(rel_emb)
            tail_re, tail_im = self._to_complex(tail_emb)

            rotated_re, rotated_im = self._complex_multiply(
                head_re, head_im, rel_re, rel_im
            )

            diff_re = rotated_re - tail_re
            diff_im = rotated_im - tail_im

            # Update to minimize distance
            # Gradient w.r.t. head (real and imaginary)
            grad_head_re = diff_re * rel_re + diff_im * rel_im
            grad_head_im = diff_im * rel_re - diff_re * rel_im

            # Gradient w.r.t. tail
            grad_tail_re = -diff_re
            grad_tail_im = -diff_im

            # Apply updates
            self._node_embeddings[head_idx, :self._complex_dim] -= lr * grad_head_re
            self._node_embeddings[head_idx, self._complex_dim:] -= lr * grad_head_im
            self._node_embeddings[tail_idx, :self._complex_dim] -= lr * grad_tail_re
            self._node_embeddings[tail_idx, self._complex_dim:] -= lr * grad_tail_im

            # Update relation phase
            grad_phase = np.sum(
                diff_re * (-head_re * rel_im - head_im * rel_re) +
                diff_im * (head_re * rel_re - head_im * rel_im)
            )
            self._relation_phases[rel_idx] -= lr * grad_phase * 0.1

        # Update based on negative triples (push apart)
        for triple in negative_triples:
            head_idx, rel_idx, tail_idx = triple

            head_emb = self._node_embeddings[head_idx]
            rel_emb = self._relation_embeddings[rel_idx]
            tail_emb = self._node_embeddings[tail_idx]

            head_re, head_im = self._to_complex(head_emb)
            rel_re, rel_im = self._to_complex(rel_emb)
            tail_re, tail_im = self._to_complex(tail_emb)

            rotated_re, rotated_im = self._complex_multiply(
                head_re, head_im, rel_re, rel_im
            )

            diff_re = rotated_re - tail_re
            diff_im = rotated_im - tail_im

            # Gradient direction (opposite for negative)
            grad_head_re = diff_re * rel_re + diff_im * rel_im
            grad_head_im = diff_im * rel_re - diff_re * rel_im
            grad_tail_re = -diff_re
            grad_tail_im = -diff_im

            # Push apart (add gradient instead of subtract)
            self._node_embeddings[head_idx, :self._complex_dim] += lr * grad_head_re * 0.5
            self._node_embeddings[head_idx, self._complex_dim:] += lr * grad_head_im * 0.5
            self._node_embeddings[tail_idx, :self._complex_dim] += lr * grad_tail_re * 0.5
            self._node_embeddings[tail_idx, self._complex_dim:] += lr * grad_tail_im * 0.5

        # Reconstruct relation embeddings from phases (maintain unit modulus)
        self._relation_embeddings[:, :self._complex_dim] = np.cos(self._relation_phases)
        self._relation_embeddings[:, self._complex_dim:] = np.sin(self._relation_phases)

    def train(
        self,
        graph: Any,
        epochs: int = 100,
        batch_size: int = 256,
        n_negative: int = 1,
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Train RotatE model on a knowledge graph.

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
            f"Training RotatE: dim={self.embedding_dim}, margin={self.margin}, "
            f"lr={self.learning_rate}, temp={self.adversarial_temperature}"
        )
        return super().train(
            graph=graph,
            epochs=epochs,
            batch_size=batch_size,
            n_negative=n_negative,
            verbose=verbose,
        )


def create_rotate_model(
    embedding_dim: int = 128,
    margin: float = 6.0,
    learning_rate: float = 0.001,
    adversarial_temperature: float = 1.0,
    random_state: Optional[int] = None,
) -> RotatEModel:
    """
    Factory function to create a RotatE model with recommended defaults.

    Args:
        embedding_dim: Dimension of embeddings (default: 128)
        margin: Margin for ranking loss (default: 6.0)
        learning_rate: Learning rate (default: 0.001)
        adversarial_temperature: Temperature for self-adversarial sampling
        random_state: Random seed

    Returns:
        RotatEModel instance
    """
    return RotatEModel(
        embedding_dim=embedding_dim,
        margin=margin,
        learning_rate=learning_rate,
        adversarial_temperature=adversarial_temperature,
        random_state=random_state,
    )
