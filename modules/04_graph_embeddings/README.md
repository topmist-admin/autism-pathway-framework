# Module 04: Graph Embeddings

Knowledge graph embedding models for learning vector representations of biological entities and their relationships.

---

## Overview

This module provides implementations of knowledge graph embedding (KGE) algorithms that learn low-dimensional vector representations for nodes (genes, pathways, GO terms) and relations (interactions, pathway membership) in the biological knowledge graph.

### Why Graph Embeddings?

| Benefit | Description |
|---------|-------------|
| **Dimensionality Reduction** | Compress complex graph structure into dense vectors |
| **Similarity Search** | Find similar genes/pathways via vector similarity |
| **Link Prediction** | Predict missing interactions or annotations |
| **Downstream ML** | Use embeddings as features for clustering, classification |

---

## Implemented Models

### TransE (Translation-based Embeddings)

**Paper**: [Bordes et al. (2013)](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)

**Principle**: Models relations as translations in embedding space.
```
For valid triple (h, r, t): h + r ≈ t
```

**Strengths**:
- Simple and fast
- Works well for 1-to-1 relations
- Good baseline model

**Limitations**:
- Struggles with symmetric/reflexive relations
- Cannot model 1-to-N or N-to-1 relations well

### RotatE (Rotation-based Embeddings)

**Paper**: [Sun et al. (2019)](https://arxiv.org/abs/1902.10197)

**Principle**: Models relations as rotations in complex space.
```
For valid triple (h, r, t): t = h ∘ r (complex multiplication)
```

**Strengths**:
- Handles symmetric relations (r ∘ r = 1)
- Handles antisymmetric relations
- Handles inverse relations (r₂ = r₁⁻¹)
- Handles composition (r₃ = r₁ ∘ r₂)

**When to use**:
- Complex relation patterns
- Biological hierarchies (GO, pathways)
- Bidirectional interactions

---

## Quick Start

### Basic Usage

```python
from modules.03_knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder
from modules.04_graph_embeddings import TransEModel, train_embeddings

# Build or load a knowledge graph
kg = (
    KnowledgeGraphBuilder()
    .add_genes(["SHANK3", "CHD8", "SCN2A", "NRXN1"])
    .add_pathways(pathway_db)
    .add_ppi(ppi_network)
    .build()
)

# Option 1: Direct model usage
model = TransEModel(embedding_dim=128, margin=1.0)
history = model.train(kg, epochs=100)
embeddings = model.get_node_embeddings()

# Option 2: Convenience function
model, embeddings = train_embeddings(
    kg,
    model_type="transe",
    embedding_dim=128,
    epochs=100,
)

# Use embeddings
shank3_emb = embeddings.get("SHANK3")
similar_genes = embeddings.most_similar("SHANK3", k=10)
```

### Using the Trainer

```python
from modules.04_graph_embeddings import TrainingConfig, EmbeddingTrainer

config = TrainingConfig(
    model_type="rotate",      # or "transe"
    embedding_dim=128,
    epochs=100,
    batch_size=256,
    learning_rate=0.001,
    margin=6.0,               # Higher for RotatE
    random_state=42,
)

trainer = EmbeddingTrainer(config)
model, history = trainer.train(kg)

# Evaluate on held-out triples
test_triples = [("SHANK3", "gene_interacts_gene", "NRXN1"), ...]
metrics = trainer.evaluate(model, test_triples)
print(f"MRR: {metrics.mean_reciprocal_rank:.4f}")
print(f"Hits@10: {metrics.hits_at_10:.4f}")
```

### Full Pipeline

```python
from modules.04_graph_embeddings import EmbeddingPipeline

pipeline = EmbeddingPipeline(
    model_type="transe",
    embedding_dim=128,
    epochs=100,
)

# Run pipeline
embeddings = pipeline.run(kg)

# Save outputs
pipeline.save("output/embeddings/")

# Load later
pipeline = EmbeddingPipeline.load("output/embeddings/")
```

---

## API Reference

### NodeEmbeddings

Container for learned node embeddings.

```python
@dataclass
class NodeEmbeddings:
    node_ids: List[str]
    embeddings: np.ndarray  # shape: (n_nodes, embedding_dim)
    embedding_dim: int
    model_type: str

    def get(self, node_id: str) -> Optional[np.ndarray]
    def get_batch(self, node_ids: List[str]) -> Tuple[np.ndarray, List[str]]
    def most_similar(self, node_id: str, k: int = 10) -> List[Tuple[str, float]]
    def compute_similarity(self, node_id1: str, node_id2: str) -> Optional[float]
    def save(self, path: str) -> None
    def load(cls, path: str) -> "NodeEmbeddings"
    def to_dict(self) -> Dict[str, np.ndarray]
```

### TransEModel

```python
class TransEModel(BaseEmbeddingModel):
    def __init__(
        self,
        embedding_dim: int = 128,
        margin: float = 1.0,
        norm: int = 1,           # L1 or L2
        learning_rate: float = 0.01,
        random_state: Optional[int] = None,
    )

    def train(
        self,
        graph: KnowledgeGraph,
        epochs: int = 100,
        batch_size: int = 256,
        n_negative: int = 1,
        verbose: bool = True,
    ) -> TrainingHistory

    def get_node_embeddings(self) -> NodeEmbeddings
    def get_relation_embeddings(self) -> RelationEmbeddings
    def predict_link(self, head: str, relation: str, tail: str) -> float
    def predict_tail(self, head: str, relation: str, k: int) -> List[Tuple[str, float]]
```

### RotatEModel

```python
class RotatEModel(BaseEmbeddingModel):
    def __init__(
        self,
        embedding_dim: int = 128,        # Must be even
        margin: float = 6.0,
        learning_rate: float = 0.001,
        adversarial_temperature: float = 1.0,
        random_state: Optional[int] = None,
    )
    # Same methods as TransEModel
```

### EvaluationMetrics

```python
@dataclass
class EvaluationMetrics:
    mean_rank: float              # Lower is better
    mean_reciprocal_rank: float   # Higher is better (0-1)
    hits_at_1: float              # Proportion with rank 1
    hits_at_3: float              # Proportion with rank ≤ 3
    hits_at_10: float             # Proportion with rank ≤ 10
    num_test_triples: int
```

---

## Hyperparameter Guide

### TransE Recommendations

| Parameter | Default | Typical Range | Notes |
|-----------|---------|---------------|-------|
| `embedding_dim` | 128 | 50-256 | Higher = more capacity, slower |
| `margin` | 1.0 | 0.5-2.0 | Gap between positive/negative scores |
| `norm` | 1 (L1) | 1 or 2 | L1 often works better |
| `learning_rate` | 0.01 | 0.001-0.1 | Decrease if unstable |
| `epochs` | 100 | 50-500 | Monitor loss convergence |
| `n_negative` | 1 | 1-10 | More negatives = slower but better |

### RotatE Recommendations

| Parameter | Default | Typical Range | Notes |
|-----------|---------|---------------|-------|
| `embedding_dim` | 128 | 64-256 | Must be even |
| `margin` | 6.0 | 3.0-12.0 | Higher than TransE |
| `learning_rate` | 0.001 | 0.0001-0.01 | Lower than TransE |
| `adversarial_temperature` | 1.0 | 0.5-2.0 | Self-adversarial sampling |

---

## Best Practices

### 1. Start with TransE

TransE is faster and often sufficient. Use RotatE if:
- You have many symmetric relations
- You need to model hierarchies
- TransE performance plateaus

### 2. Monitor Training Loss

```python
import matplotlib.pyplot as plt

history = model.train(kg, epochs=100)

plt.plot(history.epochs, history.losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
```

### 3. Use Validation for Early Stopping

```python
config = TrainingConfig(
    early_stopping_patience=10,
    validation_split=0.1,
)
```

### 4. Save Checkpoints

```python
# Save after training
model.save("models/transe_v1.pkl")
embeddings.save("embeddings/gene_embeddings.npz")

# Load later
model = TransEModel.load("models/transe_v1.pkl")
embeddings = NodeEmbeddings.load("embeddings/gene_embeddings.npz")
```

---

## Integration with Module 03

This module consumes `KnowledgeGraph` from Module 03:

```python
from modules.03_knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder
from modules.04_graph_embeddings import TransEModel

# Module 03: Build graph
kg = (
    KnowledgeGraphBuilder()
    .add_genes(gene_list)
    .add_pathways(pathway_db)
    .add_ppi(ppi_network)
    .build()
)

# Module 04: Learn embeddings
model = TransEModel(embedding_dim=128)
model.train(kg, epochs=100)
embeddings = model.get_node_embeddings()

# Use embeddings for downstream tasks
# - Gene similarity search
# - Pathway enrichment features
# - Clustering input
```

---

## Testing

```bash
# Run all tests
python -m pytest modules/04_graph_embeddings/tests/ -v

# Run specific test class
python -m pytest modules/04_graph_embeddings/tests/test_embeddings.py::TestTransEModel -v
```

---

## Files

| File | Description |
|------|-------------|
| `base.py` | Base classes, NodeEmbeddings, TrainingHistory |
| `transe.py` | TransE model implementation |
| `rotate.py` | RotatE model implementation |
| `trainer.py` | High-level training utilities |
| `__init__.py` | Module exports |
| `tests/test_embeddings.py` | Unit tests |

---

## References

1. **TransE**: Bordes, A., et al. (2013). "Translating Embeddings for Modeling Multi-relational Data." NIPS.
2. **RotatE**: Sun, Z., et al. (2019). "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space." ICLR.
3. **Survey**: Wang, Q., et al. (2017). "Knowledge Graph Embedding: A Survey of Approaches and Applications."
