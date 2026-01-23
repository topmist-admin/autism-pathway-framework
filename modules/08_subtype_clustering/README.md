# Module 08: Subtype Clustering

Identifies autism subtypes through unsupervised clustering of pathway disruption patterns with bootstrap stability testing and biological characterization.

## Overview

This module takes normalized pathway scores from Module 07 and identifies distinct autism subtypes based on shared patterns of pathway disruption. It provides:

1. **Multiple Clustering Methods**: GMM, spectral, hierarchical, and k-means clustering
2. **Stability Analysis**: Bootstrap-based assessment of cluster robustness
3. **Subtype Characterization**: Biological profiles including pathway signatures and gene involvement

## Components

### SubtypeClusterer

Performs unsupervised clustering to identify autism subtypes.

```python
from modules.08_subtype_clustering import SubtypeClusterer, ClusteringConfig, ClusteringMethod

# Configure clustering
config = ClusteringConfig(
    method=ClusteringMethod.GMM,
    n_clusters=3,
    random_state=42,
)

# Create clusterer and identify subtypes
clusterer = SubtypeClusterer(config)
result = clusterer.cluster(pathway_scores, sample_ids)

# Access results
print(f"Identified {result.n_clusters} subtypes")
print(f"Silhouette score: {result.silhouette_score:.3f}")

for cluster_id in range(result.n_clusters):
    samples = result.get_samples_in_cluster(cluster_id)
    print(f"Subtype {cluster_id}: {len(samples)} samples")
```

#### Clustering Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `GMM` | Gaussian Mixture Model | Default, provides soft assignments |
| `SPECTRAL` | Spectral clustering | Non-convex clusters |
| `HIERARCHICAL` | Agglomerative clustering | Interpretable dendrograms |
| `KMEANS` | K-means clustering | Fast, large datasets |

### StabilityAnalyzer

Assesses clustering robustness through bootstrap resampling.

```python
from modules.08_subtype_clustering import StabilityAnalyzer, StabilityConfig

# Configure stability analysis
config = StabilityConfig(
    n_bootstrap=100,
    sample_fraction=0.8,
    random_state=42,
)

# Analyze stability
analyzer = StabilityAnalyzer(config)
stability = analyzer.analyze_stability(pathway_scores, clusterer, sample_ids)

# Check results
print(f"Mean ARI: {stability.mean_ari:.3f} ({stability.stability_rating})")
print(f"95% CI: [{stability.ari_ci_low:.3f}, {stability.ari_ci_high:.3f}]")

# Identify unstable samples
unstable = stability.get_unstable_samples(threshold=0.7)
print(f"Unstable samples: {len(unstable)}")
```

#### Stability Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| ARI | Adjusted Rand Index | > 0.8 |
| NMI | Normalized Mutual Information | > 0.8 |
| Sample stability | Per-sample consistency | > 0.9 |
| Cluster stability | Per-cluster consistency | > 0.8 |

### SubtypeCharacterizer

Generates biological profiles for identified subtypes.

```python
from modules.08_subtype_clustering import SubtypeCharacterizer, CharacterizationConfig

# Configure characterization
config = CharacterizationConfig(
    n_top_pathways=10,
    p_value_threshold=0.05,
    use_fdr_correction=True,
)

# Characterize subtypes
characterizer = SubtypeCharacterizer(config)
profiles = characterizer.characterize(
    clustering_result,
    pathway_scores,
    pathway_ids,
    pathway_names,
)

# Examine profiles
for profile in profiles:
    print(profile.summary)
    print(f"  Top pathways:")
    for sig in profile.top_pathways[:3]:
        print(f"    {sig.pathway_name}: effect={sig.effect_size:.2f}, {sig.direction}")

# Generate report
report = characterizer.generate_report(profiles)
print(report)
```

## Data Structures

### ClusteringResult

```python
@dataclass
class ClusteringResult:
    labels: np.ndarray           # Cluster assignments
    n_clusters: int              # Number of clusters
    sample_ids: List[str]        # Sample identifiers
    probabilities: np.ndarray    # Soft cluster assignments (GMM)
    centroids: np.ndarray        # Cluster centers
    silhouette_score: float      # Clustering quality
    calinski_harabasz_score: float
    davies_bouldin_score: float
    metadata: Dict[str, Any]
```

Key methods:
- `get_samples_in_cluster(cluster_id)` - Get samples in a cluster
- `to_dataframe()` - Convert to pandas DataFrame
- `cluster_sizes` - Property returning cluster size dict

### SubtypeProfile

```python
@dataclass
class SubtypeProfile:
    subtype_id: int
    n_samples: int
    sample_ids: List[str]
    top_pathways: List[PathwaySignature]
    pathway_scores_mean: Dict[str, float]
    pathway_scores_std: Dict[str, float]
    top_genes: List[Tuple[str, float]]
    gene_frequency: Dict[str, float]
    centroid: np.ndarray
    within_cluster_variance: float
    silhouette_score: float
```

### StabilityResult

```python
@dataclass
class StabilityResult:
    mean_ari: float              # Mean adjusted Rand index
    std_ari: float
    mean_nmi: float              # Mean normalized mutual info
    std_nmi: float
    sample_stability: np.ndarray # Per-sample stability
    cluster_stability: np.ndarray
    co_clustering_matrix: np.ndarray
    ari_ci_low: float            # Confidence intervals
    ari_ci_high: float
```

## Full Pipeline Example

```python
from modules.07_pathway_scoring import PathwayScoreNormalizer
from modules.08_subtype_clustering import (
    SubtypeClusterer,
    StabilityAnalyzer,
    SubtypeCharacterizer,
    ClusteringConfig,
    StabilityConfig,
    ClusteringMethod,
)

# Load normalized pathway scores (from Module 07)
normalizer = PathwayScoreNormalizer()
normalized_scores = normalizer.normalize(pathway_scores)

# Step 1: Find optimal number of clusters
config = ClusteringConfig(method=ClusteringMethod.GMM, random_state=42)
clusterer = SubtypeClusterer(config)
optimal_k, k_scores = clusterer.find_optimal_k(
    normalized_scores.scores,
    k_range=(2, 8),
)
print(f"Optimal k: {optimal_k}")

# Step 2: Cluster with optimal k
config.n_clusters = optimal_k
result = clusterer.cluster(
    normalized_scores.scores,
    normalized_scores.samples,
)

# Step 3: Assess stability
stability_analyzer = StabilityAnalyzer(StabilityConfig(n_bootstrap=100))
stability = stability_analyzer.analyze_stability(
    normalized_scores.scores,
    clusterer,
    normalized_scores.samples,
)
print(f"Stability: {stability.stability_rating} (ARI={stability.mean_ari:.3f})")

# Step 4: Characterize subtypes
characterizer = SubtypeCharacterizer()
profiles = characterizer.characterize(
    result,
    normalized_scores.scores,
    normalized_scores.pathways,
    normalized_scores.pathway_names,
)

# Step 5: Compare subtypes
comparison = characterizer.compare_subtypes(profiles, normalized_scores.pathways)
print(f"Top discriminating pathways: {comparison['discriminating_pathways'][:5]}")

# Step 6: Generate report
report = characterizer.generate_report(profiles)
print(report)
```

## Configuration Reference

### ClusteringConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | ClusteringMethod | GMM | Clustering algorithm |
| `n_clusters` | int | 3 | Number of clusters |
| `random_state` | int | None | Random seed |
| `linkage` | str | "ward" | Linkage for hierarchical |
| `affinity` | str | "rbf" | Affinity for spectral |
| `n_init` | int | 10 | Initializations for GMM/k-means |
| `max_iter` | int | 300 | Max iterations |

### StabilityConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bootstrap` | int | 100 | Bootstrap iterations |
| `sample_fraction` | float | 0.8 | Fraction per bootstrap |
| `random_state` | int | 42 | Random seed |
| `ci_level` | float | 0.95 | Confidence interval level |
| `compute_co_clustering` | bool | True | Compute co-clustering matrix |

### CharacterizationConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_top_pathways` | int | 10 | Top pathways per subtype |
| `min_fold_change` | float | 1.5 | Min fold change |
| `p_value_threshold` | float | 0.05 | P-value cutoff |
| `use_fdr_correction` | bool | True | Apply FDR correction |
| `n_top_genes` | int | 20 | Top genes per subtype |
| `min_effect_size` | float | 0.5 | Min Cohen's d |

## Dependencies

- Module 07 (Pathway Scoring): PathwayScoreMatrix for input
- NumPy, SciPy: Numerical computations
- scikit-learn: Clustering algorithms and metrics

## Testing

```bash
# Run module tests
source autismenv/bin/activate
python -m pytest modules/08_subtype_clustering/tests/ -v

# Run specific test class
python -m pytest modules/08_subtype_clustering/tests/test_clustering.py::TestSubtypeClusterer -v

# Run with coverage
python -m pytest modules/08_subtype_clustering/tests/ -v --cov=modules/08_subtype_clustering
```

## Implementation Notes

### Choosing a Clustering Method

- **GMM**: Best for most cases; provides soft assignments (probability of belonging to each cluster); handles varying cluster sizes
- **Spectral**: Good for non-convex clusters; may struggle with very large datasets
- **Hierarchical**: Produces interpretable dendrograms; "ward" linkage works well for most cases
- **K-means**: Fastest option; assumes spherical clusters of similar size

### Stability Assessment

Stability analysis helps ensure subtypes are robust and reproducible:

- **ARI > 0.9**: Excellent stability, subtypes are highly reproducible
- **ARI 0.8-0.9**: Good stability, subtypes are reasonably robust
- **ARI 0.6-0.8**: Moderate stability, some boundary samples may shift
- **ARI < 0.6**: Poor stability, consider different k or method

### Handling Unstable Samples

Samples with low stability scores may:
- Lie on cluster boundaries
- Represent transitional phenotypes
- Require special treatment in downstream analysis

Consider:
1. Excluding unstable samples from subtype-specific analyses
2. Using soft cluster assignments for these samples
3. Creating an "unclassified" category for very unstable samples

## Integration with Pipelines

This module is a core component of the framework's integration pipelines:

### SubtypeDiscoveryPipeline

End-to-end pipeline from VCF to subtype identification:

```python
from pipelines import SubtypeDiscoveryPipeline, PipelineConfig, DataConfig

config = PipelineConfig(
    data=DataConfig(
        vcf_path="cohort.vcf.gz",
        pathway_gmt_path="reactome.gmt",
    ),
)
pipeline = SubtypeDiscoveryPipeline(config)
result = pipeline.run()

# Access clustering results
print(f"Identified {result.n_subtypes} subtypes")
for profile in result.subtype_profiles:
    print(f"  Subtype {profile.subtype_id}: {profile.n_samples} samples")
```

### TherapeuticHypothesisPipeline

Extends subtype discovery with therapeutic hypothesis generation:

```python
from pipelines import TherapeuticHypothesisPipeline, TherapeuticPipelineConfig

config = TherapeuticPipelineConfig(
    data=DataConfig(vcf_path="cohort.vcf.gz", pathway_gmt_path="reactome.gmt"),
)
pipeline = TherapeuticHypothesisPipeline(config)
result = pipeline.run()
```

See [pipelines/README.md](../../pipelines/README.md) for complete pipeline documentation
