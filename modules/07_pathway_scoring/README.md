# Module 07: Pathway Scoring

Aggregates gene-level burden scores to pathway-level scores using various methods, with optional network-based signal refinement through diffusion algorithms.

## Overview

This module transforms individual genetic variant burden scores (from Module 02) into pathway-level disruption scores, enabling pathway-level analysis of genetic heterogeneity. It provides:

1. **Pathway Aggregation**: Multiple methods for combining gene scores into pathway scores
2. **Network Propagation**: Diffusion-based signal refinement through biological networks
3. **Score Normalization**: Various normalization strategies for downstream analysis

## Components

### PathwayAggregator

Aggregates gene burden scores to pathway scores using configurable methods.

```python
from modules.07_pathway_scoring import PathwayAggregator, AggregationConfig, AggregationMethod

# Configure aggregation
config = AggregationConfig(
    method=AggregationMethod.WEIGHTED_SUM,
    min_pathway_size=5,
    max_pathway_size=500,
    normalize_by_pathway_size=True,
    weight_by_gene_coverage=True,
)

# Create aggregator and compute scores
aggregator = PathwayAggregator(config)
pathway_scores = aggregator.aggregate(gene_burdens, pathway_db)

# Access results
print(f"Scored {pathway_scores.n_pathways} pathways for {pathway_scores.n_samples} samples")
top_pathways = pathway_scores.get_top_pathways("sample_001", n=10)
```

#### Aggregation Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `WEIGHTED_SUM` | Sum of gene scores | Default, balanced approach |
| `MEAN` | Mean of gene scores | Pathway size-independent |
| `MAX` | Maximum gene score | Detect single strong hits |
| `SUM` | Simple sum | Raw signal accumulation |
| `MEDIAN` | Median of gene scores | Robust to outliers |
| `TOP_K_MEAN` | Mean of top K genes | Focus on strongest signals |
| `SQRT_SUM` | Square root of sum | Reduce many-small-hit bias |

### NetworkPropagator

Spreads gene-level signals through biological networks using random walk or heat diffusion algorithms.

```python
from modules.07_pathway_scoring import NetworkPropagator, PropagationConfig, PropagationMethod

# Configure propagation
config = PropagationConfig(
    method=PropagationMethod.RANDOM_WALK,
    restart_prob=0.5,  # Alpha parameter
    n_iterations=100,
)

# Build network from knowledge graph
propagator = NetworkPropagator(config)
propagator.build_network(knowledge_graph, edge_types=["gene_interacts_gene"])

# Propagate single sample's gene scores
propagated = propagator.propagate({"SHANK3": 1.0, "CHD8": 0.8})

# Or propagate entire burden matrix
propagated_burdens = propagator.propagate_gene_burdens(gene_burdens)
```

#### Propagation Methods

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `RANDOM_WALK` | Random walk with restart | `restart_prob` (0-1) |
| `HEAT_DIFFUSION` | Heat diffusion kernel | `diffusion_time` |
| `INSULATED` | Insulated heat diffusion | `diffusion_time` |
| `PAGERANK` | Personalized PageRank | `restart_prob` |

### PathwayScoreNormalizer

Normalizes pathway scores for cross-sample and cross-pathway comparisons.

```python
from modules.07_pathway_scoring import PathwayScoreNormalizer, NormalizationConfig, NormalizationMethod

# Configure normalization
config = NormalizationConfig(
    method=NormalizationMethod.ZSCORE,
    clip_outliers=True,
    outlier_std=3.0,
)

# Normalize scores
normalizer = PathwayScoreNormalizer(config)
normalized_scores = normalizer.normalize(pathway_scores)

# Compute significance
significance = normalizer.compute_significance(normalized_scores)
```

#### Normalization Methods

| Method | Description | Output Range |
|--------|-------------|--------------|
| `ZSCORE` | Standard z-score | (-∞, +∞), mean=0, std=1 |
| `ZSCORE_ROBUST` | Median/MAD-based | (-∞, +∞) |
| `MINMAX` | Min-max scaling | [0, 1] |
| `RANK` | Rank-based | [0, 1] |
| `PERCENTILE` | Percentile ranks | [0, 100] |
| `QUANTILE` | Quantile normalization | Varies |
| `LOG` | Log transformation | (-∞, +∞) |
| `SAMPLE_ZSCORE` | Per-sample z-score | (-∞, +∞) |

## Data Structures

### PathwayScoreMatrix

```python
@dataclass
class PathwayScoreMatrix:
    samples: List[str]           # Sample IDs
    pathways: List[str]          # Pathway IDs
    scores: np.ndarray           # Shape: (n_samples, n_pathways)
    pathway_names: Dict[str, str]  # Pathway ID -> name
    contributing_genes: Dict[Tuple[str, str], List[str]]  # Track which genes contributed
    metadata: Dict[str, Any]
```

Key methods:
- `get_sample(sample_id)` - Get pathway scores for a sample
- `get_pathway(pathway_id)` - Get sample scores for a pathway
- `get_top_pathways(sample_id, n)` - Get top N scoring pathways
- `get_contributing_genes(sample_id, pathway_id)` - Get genes that contributed
- `filter_pathways(min_score, min_samples_hit)` - Filter by criteria
- `to_dataframe()` - Convert to pandas DataFrame

## Full Pipeline Example

```python
from modules.01_data_loaders import PathwayLoader
from modules.02_variant_processing import GeneBurdenCalculator
from modules.03_knowledge_graph import KnowledgeGraph
from modules.07_pathway_scoring import (
    PathwayAggregator,
    NetworkPropagator,
    PathwayScoreNormalizer,
    AggregationConfig,
    PropagationConfig,
    NormalizationConfig,
)

# Load data
pathway_db = PathwayLoader().load_reactome("data/reactome.gmt")
knowledge_graph = KnowledgeGraph.load("data/kg.gpickle")
gene_burdens = GeneBurdenCalculator().compute(variants)

# Step 1: Optional network propagation
propagator = NetworkPropagator(PropagationConfig(restart_prob=0.5))
propagator.build_network(knowledge_graph)
propagated_burdens = propagator.propagate_gene_burdens(gene_burdens)

# Step 2: Aggregate to pathways
aggregator = PathwayAggregator(AggregationConfig(
    method=AggregationMethod.WEIGHTED_SUM,
    normalize_by_pathway_size=True,
))
pathway_scores = aggregator.aggregate(propagated_burdens, pathway_db)

# Step 3: Normalize
normalizer = PathwayScoreNormalizer(NormalizationConfig(
    method=NormalizationMethod.ZSCORE,
))
normalized = normalizer.normalize(pathway_scores)

# Analyze results
for sample in normalized.samples:
    top = normalized.get_top_pathways(sample, n=5)
    print(f"\n{sample} top disrupted pathways:")
    for pathway_id, score in top:
        name = normalized.pathway_names.get(pathway_id, pathway_id)
        genes = normalized.get_contributing_genes(sample, pathway_id)
        print(f"  {name}: {score:.2f} ({len(genes)} genes)")
```

## Configuration Reference

### AggregationConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | AggregationMethod | WEIGHTED_SUM | Aggregation method |
| `min_pathway_size` | int | 5 | Min genes in pathway |
| `max_pathway_size` | int | 500 | Max genes in pathway |
| `normalize_by_pathway_size` | bool | True | Normalize by √size |
| `weight_by_gene_coverage` | bool | True | Weight by coverage |
| `top_k` | int | 10 | K for TOP_K_MEAN |
| `min_genes_hit` | int | 1 | Min genes with burden |

### PropagationConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | PropagationMethod | RANDOM_WALK | Propagation method |
| `restart_prob` | float | 0.5 | RWR restart probability |
| `diffusion_time` | float | 0.1 | Heat diffusion time |
| `n_iterations` | int | 100 | Max iterations |
| `convergence_threshold` | float | 1e-6 | Convergence threshold |
| `normalize_edges` | bool | True | Normalize edge weights |

### NormalizationConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | NormalizationMethod | ZSCORE | Normalization method |
| `center` | bool | True | Subtract mean |
| `scale` | bool | True | Divide by std |
| `clip_outliers` | bool | False | Clip extreme values |
| `outlier_std` | float | 3.0 | Outlier threshold (stds) |
| `log_base` | float | 2.0 | Base for log transform |
| `pseudocount` | float | 1.0 | Pseudocount for log |

## Dependencies

- Module 01 (Data Loaders): PathwayDatabase
- Module 02 (Variant Processing): GeneBurdenMatrix
- Module 03 (Knowledge Graph): KnowledgeGraph (optional, for network propagation)

## Testing

```bash
# Run module tests
python -m pytest modules/07_pathway_scoring/tests/ -v

# Run specific test class
python -m pytest modules/07_pathway_scoring/tests/test_scoring.py::TestPathwayAggregator -v
```

## Implementation Notes

### Pathway Size Normalization

When `normalize_by_pathway_size=True`, sum-based methods divide by √(pathway_size) to prevent larger pathways from dominating scores.

### Network Propagation

Network propagation helps:
- Recover signal from genes not directly hit but connected to hit genes
- Smooth noisy signal across the network
- Leverage known biological relationships

Recommended `restart_prob` values:
- 0.3-0.5: More propagation, longer-range effects
- 0.6-0.8: Less propagation, more local to seed genes

### Sparse Data Handling

The module efficiently handles sparse gene burden matrices (most genes have zero burden in most samples) by:
- Only iterating over non-zero entries
- Using sparse matrix operations for network propagation
- Tracking only non-zero pathway scores
