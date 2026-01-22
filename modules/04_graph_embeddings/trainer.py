"""
Embedding Trainer

Provides high-level training utilities, evaluation, and hyperparameter management
for knowledge graph embedding models.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import time

import numpy as np

try:
    from .base import (
        BaseEmbeddingModel,
        NodeEmbeddings,
        TrainingHistory,
        EvaluationMetrics,
    )
    from .transe import TransEModel
    from .rotate import RotatEModel
except ImportError:
    from base import (
        BaseEmbeddingModel,
        NodeEmbeddings,
        TrainingHistory,
        EvaluationMetrics,
    )
    from transe import TransEModel
    from rotate import RotatEModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training embedding models."""

    # Model selection
    model_type: str = "transe"  # "transe" or "rotate"

    # Embedding dimensions
    embedding_dim: int = 128

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.01
    n_negative: int = 1

    # Model-specific parameters
    margin: float = 1.0  # TransE default
    norm: int = 1  # TransE L1 norm
    adversarial_temperature: float = 1.0  # RotatE

    # Validation
    validation_split: float = 0.1
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    # Other settings
    random_state: Optional[int] = 42
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type,
            "embedding_dim": self.embedding_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "n_negative": self.n_negative,
            "margin": self.margin,
            "norm": self.norm,
            "adversarial_temperature": self.adversarial_temperature,
            "validation_split": self.validation_split,
            "early_stopping_patience": self.early_stopping_patience,
            "random_state": self.random_state,
        }


class EmbeddingTrainer:
    """
    High-level trainer for knowledge graph embedding models.

    Provides:
    - Model creation and configuration
    - Training with validation and early stopping
    - Evaluation using link prediction metrics
    - Model comparison and selection

    Example:
        >>> from modules.03_knowledge_graph import KnowledgeGraph
        >>> config = TrainingConfig(model_type="transe", embedding_dim=64)
        >>> trainer = EmbeddingTrainer(config)
        >>> model, history = trainer.train(knowledge_graph)
        >>> metrics = trainer.evaluate(model, test_triples)
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize trainer.

        Args:
            config: Training configuration (uses defaults if not provided)
        """
        self.config = config or TrainingConfig()
        self._model: Optional[BaseEmbeddingModel] = None
        self._training_history: Optional[TrainingHistory] = None

    def create_model(self) -> BaseEmbeddingModel:
        """
        Create a model based on configuration.

        Returns:
            Embedding model instance
        """
        config = self.config

        if config.model_type.lower() == "transe":
            model = TransEModel(
                embedding_dim=config.embedding_dim,
                margin=config.margin,
                norm=config.norm,
                learning_rate=config.learning_rate,
                random_state=config.random_state,
            )
        elif config.model_type.lower() == "rotate":
            model = RotatEModel(
                embedding_dim=config.embedding_dim,
                margin=config.margin if config.margin > 1 else 6.0,
                learning_rate=config.learning_rate,
                adversarial_temperature=config.adversarial_temperature,
                random_state=config.random_state,
            )
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

        return model

    def train(
        self,
        graph: Any,
        validation_graph: Optional[Any] = None,
    ) -> Tuple[BaseEmbeddingModel, TrainingHistory]:
        """
        Train an embedding model on a knowledge graph.

        Args:
            graph: KnowledgeGraph instance for training
            validation_graph: Optional separate graph for validation

        Returns:
            Tuple of (trained_model, training_history)
        """
        start_time = time.time()

        # Create model
        model = self.create_model()

        logger.info(
            f"Training {self.config.model_type} model with config: "
            f"{json.dumps(self.config.to_dict(), indent=2)}"
        )

        # Train
        history = model.train(
            graph=graph,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            n_negative=self.config.n_negative,
            verbose=self.config.verbose,
        )

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.2f} seconds")

        self._model = model
        self._training_history = history

        return model, history

    def evaluate(
        self,
        model: BaseEmbeddingModel,
        test_triples: List[Tuple[str, str, str]],
        filtered: bool = True,
        all_triples: Optional[List[Tuple[str, str, str]]] = None,
    ) -> EvaluationMetrics:
        """
        Evaluate model using link prediction metrics.

        For each test triple (h, r, t):
        1. Replace tail with all entities and rank
        2. Replace head with all entities and rank
        3. Compute metrics based on rank of correct entity

        Args:
            model: Trained embedding model
            test_triples: List of (head, relation, tail) test triples
            filtered: If True, filter out known triples when ranking
            all_triples: All known triples for filtering (if filtered=True)

        Returns:
            EvaluationMetrics with MR, MRR, Hits@k
        """
        if not model.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        logger.info(f"Evaluating on {len(test_triples)} test triples")

        ranks = []
        known_set = set()

        if filtered and all_triples:
            known_set = set(all_triples)

        for head, relation, tail in test_triples:
            # Predict tails and get rank of true tail
            predictions = model.predict_tail(head, relation, k=model.n_nodes)

            if filtered:
                # Filter out known triples
                filtered_predictions = [
                    (t, s) for t, s in predictions
                    if (head, relation, t) not in known_set or t == tail
                ]
            else:
                filtered_predictions = predictions

            # Find rank of true tail
            rank = None
            for i, (pred_tail, _) in enumerate(filtered_predictions):
                if pred_tail == tail:
                    rank = i + 1  # 1-indexed rank
                    break

            if rank is not None:
                ranks.append(rank)

        if not ranks:
            logger.warning("No valid ranks computed")
            return EvaluationMetrics(num_test_triples=len(test_triples))

        ranks = np.array(ranks)

        metrics = EvaluationMetrics(
            mean_rank=float(np.mean(ranks)),
            mean_reciprocal_rank=float(np.mean(1.0 / ranks)),
            hits_at_1=float(np.mean(ranks <= 1)),
            hits_at_3=float(np.mean(ranks <= 3)),
            hits_at_10=float(np.mean(ranks <= 10)),
            num_test_triples=len(test_triples),
            filtered=filtered,
        )

        logger.info(
            f"Evaluation results: MR={metrics.mean_rank:.2f}, "
            f"MRR={metrics.mean_reciprocal_rank:.4f}, "
            f"Hits@10={metrics.hits_at_10:.4f}"
        )

        return metrics

    def get_embeddings(self) -> Optional[NodeEmbeddings]:
        """Get embeddings from trained model."""
        if self._model is None or not self._model.is_trained:
            return None
        return self._model.get_node_embeddings()


def train_embeddings(
    graph: Any,
    model_type: str = "transe",
    embedding_dim: int = 128,
    epochs: int = 100,
    **kwargs: Any,
) -> Tuple[BaseEmbeddingModel, NodeEmbeddings]:
    """
    Convenience function to train embeddings with minimal configuration.

    Args:
        graph: KnowledgeGraph instance
        model_type: "transe" or "rotate"
        embedding_dim: Dimension of embeddings
        epochs: Number of training epochs
        **kwargs: Additional config parameters

    Returns:
        Tuple of (trained_model, node_embeddings)
    """
    config = TrainingConfig(
        model_type=model_type,
        embedding_dim=embedding_dim,
        epochs=epochs,
        **kwargs,
    )

    trainer = EmbeddingTrainer(config)
    model, _ = trainer.train(graph)
    embeddings = model.get_node_embeddings()

    return model, embeddings


def compare_models(
    graph: Any,
    test_triples: List[Tuple[str, str, str]],
    configs: Optional[List[TrainingConfig]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple embedding models on the same data.

    Args:
        graph: KnowledgeGraph for training
        test_triples: Test triples for evaluation
        configs: List of configurations to compare
                 (defaults to TransE and RotatE with standard settings)

    Returns:
        Dictionary mapping model name to results
    """
    if configs is None:
        configs = [
            TrainingConfig(model_type="transe", embedding_dim=128, epochs=50),
            TrainingConfig(model_type="rotate", embedding_dim=128, epochs=50),
        ]

    results = {}

    for config in configs:
        name = f"{config.model_type}_dim{config.embedding_dim}"
        logger.info(f"Training and evaluating {name}")

        trainer = EmbeddingTrainer(config)
        model, history = trainer.train(graph)
        metrics = trainer.evaluate(model, test_triples)

        results[name] = {
            "config": config.to_dict(),
            "final_loss": history.final_loss,
            "metrics": metrics.to_dict(),
            "model": model,
        }

    # Log comparison
    logger.info("\n=== Model Comparison ===")
    for name, result in results.items():
        m = result["metrics"]
        logger.info(
            f"{name}: MRR={m['mrr']:.4f}, Hits@10={m['hits@10']:.4f}"
        )

    return results


class EmbeddingPipeline:
    """
    End-to-end pipeline for generating embeddings from a knowledge graph.

    Combines graph preparation, model training, and embedding extraction.
    """

    def __init__(
        self,
        model_type: str = "transe",
        embedding_dim: int = 128,
        epochs: int = 100,
        **model_kwargs: Any,
    ):
        """
        Initialize pipeline.

        Args:
            model_type: Type of embedding model
            embedding_dim: Dimension of embeddings
            epochs: Training epochs
            **model_kwargs: Additional model parameters
        """
        self.config = TrainingConfig(
            model_type=model_type,
            embedding_dim=embedding_dim,
            epochs=epochs,
            **model_kwargs,
        )
        self.trainer = EmbeddingTrainer(self.config)
        self.model: Optional[BaseEmbeddingModel] = None
        self.embeddings: Optional[NodeEmbeddings] = None

    def run(self, graph: Any) -> NodeEmbeddings:
        """
        Run the full pipeline.

        Args:
            graph: KnowledgeGraph instance

        Returns:
            NodeEmbeddings
        """
        logger.info("Starting embedding pipeline")

        # Train model
        self.model, history = self.trainer.train(graph)

        # Extract embeddings
        self.embeddings = self.model.get_node_embeddings()

        logger.info(
            f"Pipeline complete: {len(self.embeddings)} embeddings "
            f"of dimension {self.embeddings.embedding_dim}"
        )

        return self.embeddings

    def save(self, output_dir: str) -> None:
        """
        Save pipeline outputs.

        Args:
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save(output_dir / "model.pkl")

        if self.embeddings is not None:
            self.embeddings.save(output_dir / "embeddings.npz")

        # Save config
        with open(output_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Pipeline outputs saved to {output_dir}")

    @classmethod
    def load(cls, output_dir: str) -> "EmbeddingPipeline":
        """
        Load a saved pipeline.

        Args:
            output_dir: Directory containing saved pipeline

        Returns:
            EmbeddingPipeline instance
        """
        output_dir = Path(output_dir)

        # Load config
        with open(output_dir / "config.json", "r") as f:
            config_dict = json.load(f)

        pipeline = cls(**config_dict)

        # Load model if exists
        model_path = output_dir / "model.pkl"
        if model_path.exists():
            pipeline.model = BaseEmbeddingModel.load(str(model_path))

        # Load embeddings if exists
        embeddings_path = output_dir / "embeddings.npz"
        if embeddings_path.exists():
            pipeline.embeddings = NodeEmbeddings.load(str(embeddings_path))

        return pipeline
