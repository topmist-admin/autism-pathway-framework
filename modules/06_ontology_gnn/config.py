"""
Configuration management for Ontology-Aware GNN module.

Provides configuration classes and utilities for:
- Model architecture
- Training hyperparameters
- Data preprocessing
- Integration with other modules
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AggregationType(Enum):
    """Aggregation methods for message passing."""
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    ATTENTION = "attention"


class ActivationType(Enum):
    """Activation functions."""
    RELU = "relu"
    GELU = "gelu"
    LEAKY_RELU = "leaky_relu"
    NONE = "none"


class PriorCombination(Enum):
    """Methods for combining biological priors."""
    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"
    LEARNED = "learned"


@dataclass
class ModelConfig:
    """
    Model architecture configuration.

    Attributes:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        num_layers: Number of message passing layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_residual: Whether to use residual connections
        use_layer_norm: Whether to use layer normalization
        aggregation: Aggregation type for message passing
        activation: Activation function
    """
    input_dim: int = 256
    hidden_dim: int = 256
    output_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    use_residual: bool = True
    use_layer_norm: bool = True
    aggregation: str = "attention"
    activation: str = "relu"


@dataclass
class EdgeTypeConfig:
    """
    Configuration for edge types in the heterogeneous graph.

    Attributes:
        name: Edge type name
        source_type: Source node type
        target_type: Target node type
        is_directed: Whether edge is directed
        weight_key: Key for edge weights (if any)
    """
    name: str
    source_type: str
    target_type: str
    is_directed: bool = True
    weight_key: Optional[str] = None


@dataclass
class GraphConfig:
    """
    Graph structure configuration.

    Attributes:
        node_types: List of node types in the graph
        edge_types: List of edge type configurations
        max_nodes: Maximum number of nodes (for batching)
        max_edges: Maximum number of edges (for batching)
    """
    node_types: List[str] = field(default_factory=lambda: ["gene", "pathway", "go_term"])

    edge_types: List[EdgeTypeConfig] = field(default_factory=lambda: [
        EdgeTypeConfig("gene_interacts", "gene", "gene", is_directed=False),
        EdgeTypeConfig("gene_in_pathway", "gene", "pathway", is_directed=True),
        EdgeTypeConfig("gene_has_go", "gene", "go_term", is_directed=True),
        EdgeTypeConfig("pathway_contains", "pathway", "gene", is_directed=True),
        EdgeTypeConfig("go_is_a", "go_term", "go_term", is_directed=True),
    ])

    max_nodes: Optional[int] = None
    max_edges: Optional[int] = None

    def get_edge_type_names(self) -> List[str]:
        """Get list of edge type names."""
        return [et.name for et in self.edge_types]


@dataclass
class BioPriorConfig:
    """
    Configuration for biological prior knowledge.

    Attributes:
        prior_types: List of prior types to use
        combination_method: How to combine multiple priors
        normalize_priors: Whether to normalize prior values
        missing_value: Value for missing priors
    """
    prior_types: List[str] = field(default_factory=lambda: [
        "pli",           # pLI score (probability of loss-of-function intolerance)
        "loeuf",         # LOEUF score (LoF observed/expected upper bound fraction)
        "expression",    # Brain expression level
        "sfari_score",   # SFARI gene confidence score
    ])

    combination_method: str = "learned"
    normalize_priors: bool = True
    missing_value: float = 0.5


@dataclass
class HierarchyConfig:
    """
    Configuration for ontology hierarchy handling.

    Attributes:
        num_levels: Number of hierarchy levels to propagate
        aggregation: Aggregation method for hierarchy
        include_go_hierarchy: Whether to use GO term hierarchy
        include_pathway_hierarchy: Whether to use pathway hierarchy
    """
    num_levels: int = 3
    aggregation: str = "attention"
    include_go_hierarchy: bool = True
    include_pathway_hierarchy: bool = True


@dataclass
class TrainingConfig:
    """
    Training configuration.

    Attributes:
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        batch_size: Batch size for training
        epochs: Number of training epochs
        patience: Early stopping patience
        min_lr: Minimum learning rate for scheduler
        warmup_epochs: Number of warmup epochs
        gradient_clip: Gradient clipping value
    """
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 100
    patience: int = 20
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    gradient_clip: float = 1.0


@dataclass
class TaskConfig:
    """
    Task-specific configuration.

    Attributes:
        tasks: List of tasks to train on
        task_weights: Weights for multi-task learning
        classification_threshold: Threshold for gene classification
        link_prediction_negative_ratio: Ratio of negative samples for link prediction
    """
    tasks: List[str] = field(default_factory=lambda: [
        "gene_classification",
        "link_prediction",
    ])

    task_weights: Dict[str, float] = field(default_factory=lambda: {
        "gene_classification": 1.0,
        "link_prediction": 0.5,
    })

    classification_threshold: float = 0.5
    link_prediction_negative_ratio: int = 5


@dataclass
class OntologyGNNConfig:
    """
    Complete configuration for Ontology-Aware GNN.

    Combines all sub-configurations into a single config object.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    bio_priors: BioPriorConfig = field(default_factory=BioPriorConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    task: TaskConfig = field(default_factory=TaskConfig)

    # Runtime settings
    device: str = "cpu"
    seed: int = 42
    num_workers: int = 4
    log_interval: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OntologyGNNConfig":
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**d.get("model", {})),
            graph=GraphConfig(**d.get("graph", {})),
            bio_priors=BioPriorConfig(**d.get("bio_priors", {})),
            hierarchy=HierarchyConfig(**d.get("hierarchy", {})),
            training=TrainingConfig(**d.get("training", {})),
            task=TaskConfig(**d.get("task", {})),
            device=d.get("device", "cpu"),
            seed=d.get("seed", 42),
            num_workers=d.get("num_workers", 4),
            log_interval=d.get("log_interval", 10),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save config to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Config saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "OntologyGNNConfig":
        """Load config from JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            d = json.load(f)
        logger.info(f"Config loaded from {path}")
        return cls.from_dict(d)


def create_default_config() -> OntologyGNNConfig:
    """Create default configuration."""
    return OntologyGNNConfig()


def create_autism_config() -> OntologyGNNConfig:
    """
    Create configuration optimized for autism gene discovery.

    Uses settings that work well for:
    - SFARI gene database
    - Synaptic pathway analysis
    - GO term enrichment
    """
    return OntologyGNNConfig(
        model=ModelConfig(
            input_dim=256,      # From pretrained embeddings
            hidden_dim=256,
            output_dim=128,
            num_layers=4,       # Deeper for complex relationships
            num_heads=8,
            dropout=0.2,        # Higher dropout for smaller datasets
        ),
        bio_priors=BioPriorConfig(
            prior_types=["pli", "loeuf", "expression", "sfari_score"],
            combination_method="learned",
        ),
        hierarchy=HierarchyConfig(
            num_levels=4,       # More levels for GO hierarchy
            include_go_hierarchy=True,
            include_pathway_hierarchy=True,
        ),
        training=TrainingConfig(
            learning_rate=5e-4,
            weight_decay=1e-4,
            epochs=200,
            patience=30,
        ),
        task=TaskConfig(
            tasks=["gene_classification", "link_prediction"],
            task_weights={
                "gene_classification": 1.0,
                "link_prediction": 0.3,
            },
        ),
    )


def create_lightweight_config() -> OntologyGNNConfig:
    """
    Create lightweight configuration for testing and development.

    Uses smaller dimensions and fewer layers for faster iteration.
    """
    return OntologyGNNConfig(
        model=ModelConfig(
            input_dim=64,
            hidden_dim=64,
            output_dim=32,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
        ),
        hierarchy=HierarchyConfig(
            num_levels=2,
        ),
        training=TrainingConfig(
            learning_rate=1e-3,
            epochs=50,
            patience=10,
        ),
    )
