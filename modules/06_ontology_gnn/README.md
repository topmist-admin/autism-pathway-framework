# Module 06: Ontology-Aware Graph Neural Network

Graph neural network architecture designed for biological knowledge graphs with support for heterogeneous node/edge types, biological attention mechanisms, and ontology hierarchy aggregation.

---

## Overview

This module implements a GNN that understands biological structure:

| Component | Purpose |
|-----------|---------|
| **EdgeTypeTransform** | Different transformations for PPI vs pathway vs GO edges |
| **MessagePassingLayer** | Heterogeneous message passing with residual connections |
| **BiologicalAttention** | Attention modulated by constraint scores (pLI, LOEUF) |
| **HierarchicalAggregator** | Propagate information through GO/pathway hierarchies |
| **OntologyAwareGNN** | Full model combining all components |

### Key Features

1. **Heterogeneous Graph Support**: Different node types (genes, pathways, GO terms) and edge types
2. **Biological Priors**: Incorporate pLI, LOEUF, expression, SFARI scores into attention
3. **Hierarchy Awareness**: Aggregate features following GO term hierarchy
4. **Multi-Task Learning**: Joint gene classification and link prediction
5. **Interpretability**: Attention weights reveal important relationships

---

## Quick Start

### Basic Usage

```python
from modules.06_ontology_gnn import (
    OntologyAwareGNN,
    GNNConfig,
    prepare_graph_data,
)

# 1. Create model configuration
config = GNNConfig(
    input_dim=256,      # From pretrained embeddings
    hidden_dim=256,
    output_dim=128,
    num_layers=3,
    num_heads=8,
    edge_types=["gene_interacts", "gene_in_pathway", "gene_has_go"],
    node_types=["gene", "pathway", "go_term"],
    prior_types=["pli", "expression", "sfari_score"],
)

# 2. Create model
model = OntologyAwareGNN(config)

# 3. Prepare data from knowledge graph
graph_data = prepare_graph_data(
    knowledge_graph,
    node_embeddings=pretrained_embeddings,
    bio_priors={"pli": pli_scores, "expression": expr_scores},
)

# 4. Forward pass
output = model(
    node_features=graph_data.node_features,
    edge_index=graph_data.edge_index,
    edge_type=graph_data.edge_type,
    edge_type_names=graph_data.edge_type_names,
    node_type_indices=graph_data.node_type_indices,
    bio_priors=graph_data.bio_priors,
)

# 5. Get gene embeddings
gene_embeddings = output.node_embeddings["gene"]
```

### With Classification Labels

```python
# Add labels for autism gene classification
labels = torch.tensor([1, 0, 1, 0, ...])  # 1=ASD gene, 0=not

output = model(
    node_features=graph_data.node_features,
    edge_index=graph_data.edge_index,
    edge_type=graph_data.edge_type,
    node_type_indices={"gene": torch.arange(num_genes)},
    labels=labels,
)

print(f"Classification loss: {output.loss_components['classification']}")
print(f"Gene logits shape: {output.gene_logits.shape}")
```

---

## API Reference

### OntologyAwareGNN

Main model class.

```python
class OntologyAwareGNN(nn.Module):
    def __init__(
        self,
        config: GNNConfig,
        input_dims: Optional[Dict[str, int]] = None,  # Per-type input dimensions
    )

    def forward(
        self,
        node_features: Dict[str, torch.Tensor],   # {node_type: [num_nodes, dim]}
        edge_index: torch.Tensor,                  # [2, num_edges]
        edge_type: torch.Tensor,                   # [num_edges]
        edge_type_names: List[str],                # Type index -> name
        node_type_indices: Dict[str, torch.Tensor], # {type: indices}
        bio_priors: Optional[Dict[str, torch.Tensor]] = None,
        hierarchy_edges: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> GNNOutput

    def encode(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor  # Combined embeddings
```

### GNNConfig

Model configuration.

```python
@dataclass
class GNNConfig:
    input_dim: int = 256
    hidden_dim: int = 256
    output_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    edge_types: List[str] = [...]
    node_types: List[str] = ["gene", "pathway", "go_term"]
    prior_types: List[str] = ["pli", "loeuf", "expression", "sfari_score"]
    dropout: float = 0.1
    use_residual: bool = True
    use_layer_norm: bool = True
    aggregation: str = "attention"
    num_hierarchy_levels: int = 3
    task_heads: List[str] = ["gene_classification", "link_prediction"]
```

### GNNOutput

Model output container.

```python
@dataclass
class GNNOutput:
    node_embeddings: Dict[str, torch.Tensor]  # {node_type: embeddings}
    gene_logits: Optional[torch.Tensor]       # Classification logits
    link_scores: Optional[torch.Tensor]       # Link prediction scores
    attention_weights: Optional[Dict[str, torch.Tensor]]
    loss: Optional[float]
    loss_components: Optional[Dict[str, float]]
```

### Attention Mechanisms

```python
# Biological attention with prior knowledge
class BiologicalAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        prior_types: List[str] = ["pli"],
        dropout: float = 0.1,
    )
    def forward(query, key, value, bio_priors=None) -> (output, attn_weights)

# Edge-type-specific attention
class EdgeTypeAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_types: List[str],
        heads_per_type: int = 2,
    )
    def forward(x, edge_index, edge_type, edge_type_names) -> output

# GO semantic similarity attention
class GOSemanticAttention(nn.Module):
    def __init__(hidden_dim: int, num_heads: int = 4)
    def forward(x, go_similarity=None) -> (output, attn_weights)

# Gene-pathway co-attention
class PathwayCoAttention(nn.Module):
    def __init__(gene_dim: int, pathway_dim: int, num_heads: int = 4)
    def forward(gene_feat, pathway_feat, membership) -> (gene_out, pathway_out)
```

### Utility Functions

```python
# Prepare graph data from knowledge graph
def prepare_graph_data(
    knowledge_graph,
    node_embeddings: Optional[Dict[str, np.ndarray]] = None,
    bio_priors: Optional[Dict[str, Dict[str, float]]] = None,
    use_torch: bool = True,
) -> GraphData

# Normalize biological priors
def normalize_priors(
    priors: Dict[str, np.ndarray],
    method: str = "minmax",  # "minmax", "zscore", "rank"
) -> Dict[str, np.ndarray]

# Compute classification metrics
def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]  # accuracy, precision, recall, f1, specificity

# Compute link prediction metrics
def compute_link_prediction_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]  # hits@10, hits@50, mrr, auc

# Create negative samples for link prediction
def create_negative_samples(
    edge_index: np.ndarray,
    num_nodes: int,
    num_negative: int,
) -> Tuple[np.ndarray, np.ndarray]

# Split edges for train/val/test
def split_edges(
    edge_index: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

---

## Configuration Presets

```python
from modules.06_ontology_gnn import (
    create_default_config,
    create_autism_config,
    create_lightweight_config,
)

# Default configuration
config = create_default_config()

# Optimized for autism gene discovery
config = create_autism_config()
# - Deeper network (4 layers)
# - Higher dropout (0.2) for smaller datasets
# - All biological priors enabled
# - Longer training (200 epochs)

# Lightweight for testing
config = create_lightweight_config()
# - Smaller dimensions (64)
# - Fewer layers (2)
# - Faster iteration
```

---

## Integration with Other Modules

### With Module 03 (Knowledge Graph)

```python
from modules.03_knowledge_graph import KnowledgeGraphBuilder
from modules.06_ontology_gnn import OntologyAwareGNN, prepare_graph_data

# Build knowledge graph
kg = KnowledgeGraphBuilder()
kg.add_genes(gene_list)
kg.add_ppi_edges(ppi_data)
kg.add_pathway_memberships(pathway_data)
graph = kg.build()

# Prepare for GNN
graph_data = prepare_graph_data(graph)
```

### With Module 05 (Pretrained Embeddings)

```python
from modules.05_pretrained_embeddings import GeneformerExtractor, fuse_embeddings
from modules.06_ontology_gnn import OntologyAwareGNN, prepare_graph_data

# Get pretrained embeddings
geneformer = GeneformerExtractor()
esm2 = ESM2Extractor()

gene_emb = geneformer.extract(gene_ids)
protein_emb = esm2.extract(protein_sequences)

# Fuse embeddings
fused = fuse_embeddings({
    "geneformer": gene_emb,
    "esm2": protein_emb,
}, method="concat")

# Use as GNN input
graph_data = prepare_graph_data(
    knowledge_graph,
    node_embeddings=fused.to_dict(),
)
```

---

## Biological Priors

The model can incorporate various biological annotations:

| Prior | Description | Range | Source |
|-------|-------------|-------|--------|
| `pli` | Probability of LoF intolerance | 0-1 | gnomAD |
| `loeuf` | LoF observed/expected upper bound | 0-2 | gnomAD |
| `expression` | Brain expression level | varies | GTEx/BrainSpan |
| `sfari_score` | SFARI autism gene confidence | 1-3 | SFARI Gene |

```python
bio_priors = {
    "pli": {gene_id: pli_score for gene_id, pli_score in pli_data},
    "expression": {gene_id: expr_level for gene_id, expr_level in expr_data},
    "sfari_score": {gene_id: sfari for gene_id, sfari in sfari_data},
}

graph_data = prepare_graph_data(kg, bio_priors=bio_priors)
```

---

## Training

```python
from modules.06_ontology_gnn import OntologyAwareGNN, GNNConfig, GNNTrainer

# Create model
config = GNNConfig(hidden_dim=256, num_layers=3)
model = OntologyAwareGNN(config)

# Create trainer
trainer = GNNTrainer(
    model,
    learning_rate=1e-3,
    weight_decay=1e-5,
    device="cuda",
)

# Training loop
for epoch in range(100):
    loss_dict = trainer.train_step(
        node_features=graph_data.node_features,
        edge_index=graph_data.edge_index,
        edge_type=graph_data.edge_type,
        labels=labels,
        node_type_indices=graph_data.node_type_indices,
    )

    metrics = trainer.evaluate(
        node_features=graph_data.node_features,
        edge_index=graph_data.edge_index,
        edge_type=graph_data.edge_type,
        labels=labels,
    )

    print(f"Epoch {epoch}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")
```

---

## Testing

```bash
# Run all tests
python -m pytest modules/06_ontology_gnn/tests/ -v

# Run specific test class
python -m pytest modules/06_ontology_gnn/tests/test_ontology_gnn.py::TestModel -v

# Run with coverage
python -m pytest modules/06_ontology_gnn/tests/ -v --cov=modules/06_ontology_gnn
```

---

## Files

| File | Description |
|------|-------------|
| `layers.py` | GNN layers (EdgeTypeTransform, MessagePassingLayer, HierarchicalAggregator) |
| `attention.py` | Attention mechanisms (BiologicalAttention, EdgeTypeAttention, GOSemanticAttention) |
| `model.py` | Main OntologyAwareGNN model and trainer |
| `config.py` | Configuration dataclasses and presets |
| `utils.py` | Graph utilities, metrics, data preparation |
| `__init__.py` | Module exports |
| `tests/test_ontology_gnn.py` | Unit tests |

---

## Architecture Diagram

```
Input Features (per node type)
         │
         ▼
┌─────────────────────┐
│  Input Projection   │  ← Project heterogeneous dims to common hidden_dim
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  + Node Type Emb    │  ← Add learnable type embeddings
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Message Passing    │  ← EdgeTypeTransform + Aggregation
│  Layer 1...N        │     Different transforms per edge type
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Hierarchy Agg      │  ← Propagate through GO hierarchy
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Biological Attention│  ← Modulated by pLI, expression, etc.
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Output Projection  │  ← Project to output_dim
└─────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ Class  │ │ Link   │  ← Task-specific heads
│ Head   │ │ Head   │
└────────┘ └────────┘
```

---

## Dependencies

- **Required**: numpy
- **Recommended**: torch (PyTorch) for full functionality
- Falls back to numpy implementations if PyTorch unavailable

---

## References

1. **Heterogeneous GNN**: Schlichtkrull et al. (2018). "Modeling Relational Data with Graph Convolutional Networks."
2. **Biological Attention**: Applying attention mechanisms with domain knowledge for interpretability.
3. **GO Hierarchy**: The Gene Ontology Consortium. "Gene Ontology: tool for the unification of biology."
