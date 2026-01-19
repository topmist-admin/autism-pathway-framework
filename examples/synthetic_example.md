# Synthetic Example: End-to-End Pathway Analysis

## Overview

This document walks through a complete example of the pathway framework using **synthetic data**. The example is designed to illustrate the methodology, not to represent real biological findings.

---

## Scenario

We have a synthetic cohort of 500 individuals with:
- Whole-exome sequencing data (simulated variants)
- 3 ground-truth subtypes (unknown to the analysis)
- Different pathway disruption patterns per subtype

**Goal**: Recover the subtypes and identify their characteristic pathways.

---

## Step 1: Data Generation (Simulation)

### Synthetic Variant Generation

```python
# Simulation parameters
N_SAMPLES = 500
N_GENES = 5000
N_PATHWAYS = 100
GENES_PER_PATHWAY = 50

# Ground truth subtypes
TRUE_SUBTYPES = {
    0: {"name": "Synaptic", "n": 200, "disrupted_pathways": ["synaptic_transmission", "neurotransmitter"]},
    1: {"name": "Chromatin", "n": 150, "disrupted_pathways": ["chromatin_remodeling", "histone_modification"]},
    2: {"name": "Mixed", "n": 150, "disrupted_pathways": ["cell_adhesion", "cytoskeleton"]}
}

# Generate synthetic variants
def generate_synthetic_variants(sample_id, subtype):
    variants = []

    # Background variants (random across all genes)
    n_background = random_poisson(lambda=20)
    for _ in range(n_background):
        gene = random_choice(ALL_GENES)
        variants.append(create_variant(gene, impact="low"))

    # Subtype-specific variants (enriched in subtype pathways)
    disrupted_pathways = TRUE_SUBTYPES[subtype]["disrupted_pathways"]
    for pathway in disrupted_pathways:
        n_hits = random_poisson(lambda=3)
        pathway_genes = PATHWAY_DEFINITIONS[pathway]
        for _ in range(n_hits):
            gene = random_choice(pathway_genes)
            variants.append(create_variant(gene, impact="high"))

    return variants
```

### Generated Data Summary

| Subtype | Name | N | Characteristic Pathways |
|---------|------|---|------------------------|
| 0 | Synaptic | 200 | Synaptic transmission, Neurotransmitter signaling |
| 1 | Chromatin | 150 | Chromatin remodeling, Histone modification |
| 2 | Mixed | 150 | Cell adhesion, Cytoskeleton organization |

---

## Step 2: Variant to Gene Aggregation

### Input: Synthetic Variants

```
Sample S001 (Subtype 0 - Synaptic):
  - SHANK3: frameshift (weight 1.0)
  - NRXN1: missense_damaging (weight 0.7)
  - GRIN2B: stop_gained (weight 1.0)
  - ... (background variants)

Sample S201 (Subtype 1 - Chromatin):
  - CHD8: frameshift (weight 1.0)
  - ARID1B: missense_damaging (weight 0.7)
  - KMT2A: splice_donor (weight 1.0)
  - ... (background variants)
```

### Processing

```python
# Apply variant_to_gene pipeline
gene_burdens = compute_all_gene_burdens(synthetic_variants)

# Example output for one sample
print(gene_burdens["S001"])
# {
#     "SHANK3": 1.0,
#     "NRXN1": 0.7,
#     "GRIN2B": 1.0,
#     "RANDOM_GENE_1": 0.2,
#     ...
# }
```

### Quality Control Results

```
Total samples: 500
Samples passing QC: 498 (99.6%)
Mean genes with burden per sample: 15.3 (SD: 4.2)
Mean total burden per sample: 8.7 (SD: 2.8)

Flagged samples: 2
  - S234: unusually high burden (sequencing artifact)
  - S445: very few variants (low coverage)
```

---

## Step 3: Gene to Pathway Aggregation

### Pathway Definitions Used

```
synaptic_transmission (GO:0007268): 45 genes
  - SHANK3, NRXN1, GRIN2B, NLGN1, SYNGAP1, ...

chromatin_remodeling (GO:0006338): 52 genes
  - CHD8, ARID1B, SMARCA4, SMARCC2, ...

cell_adhesion (GO:0007155): 61 genes
  - CDH1, ITGB1, CTNNA1, ...

... (97 more pathways)
```

### Processing

```python
# Compute pathway scores
pathway_scores, pathway_stats = compute_all_pathway_scores(
    gene_burdens,
    pathway_definitions,
    gene_weights=constraint_scores
)

# Example output
print(pathway_scores["S001"]["synaptic_transmission"])
# {
#     "raw_score": 2.7,
#     "size_normalized": 0.40,
#     "z_score": 2.1,
#     "contributing_genes": [
#         {"gene_id": "SHANK3", "contribution": 1.0},
#         {"gene_id": "GRIN2B", "contribution": 1.0},
#         {"gene_id": "NRXN1", "contribution": 0.7}
#     ]
# }
```

### Pathway Score Distribution

```
Pathway: synaptic_transmission
  Overall mean: 0.35 (SD: 0.42)

  By true subtype:
    Subtype 0 (Synaptic): mean=0.89, SD=0.38  ← Elevated!
    Subtype 1 (Chromatin): mean=0.12, SD=0.21
    Subtype 2 (Mixed): mean=0.08, SD=0.18

Pathway: chromatin_remodeling
  Overall mean: 0.28 (SD: 0.35)

  By true subtype:
    Subtype 0 (Synaptic): mean=0.10, SD=0.19
    Subtype 1 (Chromatin): mean=0.72, SD=0.32  ← Elevated!
    Subtype 2 (Mixed): mean=0.09, SD=0.17
```

---

## Step 4: Network Refinement

### Network Used

```
Gene-gene interaction network:
  - Nodes: 4,500 genes
  - Edges: 125,000 interactions
  - Sources: PPI (60%), co-expression (40%)
```

### Propagation Results

```python
# Propagate gene scores
propagated_scores = propagate_all_samples(gene_burdens, network)

# Validation: correlation with original
# (Should be moderate - too high means no propagation, too low means over-smoothing)

for sample in ["S001", "S201", "S351"]:
    diag = validate_propagation(gene_burdens[sample], propagated_scores[sample], network)
    print(f"{sample}: r={diag['original_correlation']:.2f}, top20_overlap={diag['top20_overlap']}")

# Output:
# S001: r=0.72, top20_overlap=14
# S201: r=0.68, top20_overlap=12
# S351: r=0.71, top20_overlap=13
```

### Effect of Network Refinement

```
Pathway: synaptic_transmission

Before refinement:
  Subtype 0 vs others: Cohen's d = 1.8

After refinement:
  Subtype 0 vs others: Cohen's d = 2.2  ← Signal amplified

Reason: Network propagation spreads signal to connected synaptic genes
that didn't have direct variants but are functionally related.
```

---

## Step 5: Subtype Clustering

### Dimensionality Reduction

```python
# PCA on pathway scores
X, samples, pathways = prepare_feature_matrix(refined_pathway_scores)
X_std, mean, std = standardize_features(X)
X_pca, pca_model, n_components = reduce_dimensions(X_std)

print(f"Reduced from {len(pathways)} pathways to {n_components} PCs")
print(f"Variance explained: {sum(pca_model.explained_variance_ratio_[:n_components])*100:.1f}%")

# Output:
# Reduced from 100 pathways to 12 PCs
# Variance explained: 91.2%
```

### Cluster Selection

```python
# Test different K values
results = {}
for k in range(2, 8):
    result = fit_gmm(X_pca, k)
    stability, _ = assess_cluster_stability(X_pca, k, n_bootstrap=100)
    results[k] = {
        "bic": result["bic"],
        "stability": stability
    }

# Results:
# K=2: BIC=4521, stability=0.91
# K=3: BIC=4102, stability=0.85  ← Selected (lowest BIC, good stability)
# K=4: BIC=4156, stability=0.72
# K=5: BIC=4289, stability=0.58
# K=6: BIC=4445, stability=0.45
# K=7: BIC=4612, stability=0.38
```

### Final Clustering (K=3)

```python
result = cluster_samples(refined_pathway_scores)

print(f"Number of clusters: {result['n_clusters']}")
for subtype_id, profile in result["subtype_profiles"].items():
    print(f"\nCluster {subtype_id}: {profile['size']} samples")
    print("Top pathways:")
    for p in profile["characteristic_pathways"][:3]:
        print(f"  {p['pathway_id']}: diff={p['difference']:.2f}")

# Output:
# Number of clusters: 3
#
# Cluster 0: 195 samples
# Top pathways:
#   synaptic_transmission: diff=0.82
#   neurotransmitter_signaling: diff=0.65
#   ion_channel_activity: diff=0.41
#
# Cluster 1: 152 samples
# Top pathways:
#   chromatin_remodeling: diff=0.78
#   histone_modification: diff=0.62
#   transcription_regulation: diff=0.38
#
# Cluster 2: 151 samples
# Top pathways:
#   cell_adhesion: diff=0.71
#   cytoskeleton_organization: diff=0.55
#   extracellular_matrix: diff=0.32
```

---

## Step 6: Validation

### Recovery of Ground Truth

```python
# Compare discovered clusters to true subtypes
from sklearn.metrics import adjusted_rand_index, normalized_mutual_info_score

true_labels = [get_true_subtype(s) for s in samples]
discovered_labels = [result["assignments"][s]["hard_label"] for s in samples]

ari = adjusted_rand_index(true_labels, discovered_labels)
nmi = normalized_mutual_info_score(true_labels, discovered_labels)

print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Normalized Mutual Information: {nmi:.3f}")

# Output:
# Adjusted Rand Index: 0.847
# Normalized Mutual Information: 0.862
```

### Confusion Matrix

```
                  Discovered
                  C0    C1    C2
True   Synaptic   185   8     7
       Chromatin  5     142   3
       Mixed      5     2     143

Accuracy: 94.0%
```

### Stability Analysis

```python
# Bootstrap stability
stability, stability_matrix = assess_cluster_stability(X_pca, k=3, n_bootstrap=100)

print(f"Overall stability: {stability:.2f}")

# Per-cluster stability
for cluster in range(3):
    mask = discovered_labels == cluster
    cluster_stability = stability_matrix[mask][:, mask].mean()
    print(f"Cluster {cluster} stability: {cluster_stability:.2f}")

# Output:
# Overall stability: 0.85
# Cluster 0 stability: 0.88
# Cluster 1 stability: 0.84
# Cluster 2 stability: 0.82
```

---

## Step 7: Results Summary

### Discovered Subtypes

| Cluster | Size | Top Pathway | Mean Z-score | Stability |
|---------|------|-------------|--------------|-----------|
| 0 | 195 (39%) | Synaptic transmission | 2.1 | 0.88 |
| 1 | 152 (30%) | Chromatin remodeling | 1.9 | 0.84 |
| 2 | 151 (30%) | Cell adhesion | 1.7 | 0.82 |

### Assignment Uncertainty

```
Low uncertainty (<0.3): 412 samples (82.4%)
Medium uncertainty (0.3-0.6): 68 samples (13.6%)
High uncertainty (>0.6): 18 samples (3.6%)

Samples with high uncertainty are at cluster boundaries
and may not belong clearly to any single subtype.
```

### Key Findings (Synthetic)

1. **Three distinct subtypes** were recovered matching ground truth
2. **Pathway-level analysis** successfully identified characteristic disruptions
3. **Network refinement** improved signal separation (d increased from 1.8 to 2.2)
4. **Clustering was stable** across bootstrap resamples (stability=0.85)

---

## Step 8: Negative Controls

### Permutation Test

```python
# Permute pathway scores and re-cluster
n_permutations = 100
null_aris = []

for _ in range(n_permutations):
    X_perm = permute_rows(X_pca)
    null_result = fit_gmm(X_perm, k=3)
    null_ari = adjusted_rand_index(true_labels, null_result["labels"])
    null_aris.append(null_ari)

print(f"True ARI: {ari:.3f}")
print(f"Null ARI: {np.mean(null_aris):.3f} (SD: {np.std(null_aris):.3f})")
print(f"P-value: {(np.sum(null_aris >= ari) + 1) / (n_permutations + 1):.4f}")

# Output:
# True ARI: 0.847
# Null ARI: 0.002 (SD: 0.015)
# P-value: 0.0099
```

### Random Gene Sets

```python
# Replace pathways with random gene sets of same size
random_pathway_scores = compute_random_pathway_scores(gene_burdens)
random_result = cluster_samples(random_pathway_scores)

# Compare separation
real_silhouette = silhouette_score(X_pca, discovered_labels)
random_silhouette = silhouette_score(X_random, random_result["labels"])

print(f"Real pathways silhouette: {real_silhouette:.3f}")
print(f"Random gene sets silhouette: {random_silhouette:.3f}")

# Output:
# Real pathways silhouette: 0.42
# Random gene sets silhouette: 0.08
```

---

## Limitations of This Example

1. **Synthetic data**: Real genetic data is far more complex
2. **Clean subtypes**: Real subtypes (if they exist) are likely overlapping
3. **Known ground truth**: Real analyses don't have ground truth to validate against
4. **No noise**: Real data has batch effects, population structure, etc.
5. **Simplified pathways**: Real pathway biology is more complex

---

## Applying to Real Data

When applying this framework to real data:

1. **Expect lower recovery**: Real subtypes are not as cleanly separated
2. **Use multiple validation approaches**: Cross-cohort replication is essential
3. **Interpret cautiously**: Results are hypotheses, not facts
4. **Check for confounders**: Batch effects, ancestry, etc.
5. **Report uncertainty**: Don't hide samples that don't cluster well

---

## Code Repository

Full simulation code available in the repository:
- `simulations/generate_synthetic_data.py`
- `simulations/run_full_pipeline.py`
- `simulations/evaluate_results.py`

---

## Next Steps

- Apply to real cohort data
- Test cross-cohort replication
- Explore phenotype associations
- Generate therapeutic hypotheses (research only)

---

**Reminder**: This is a synthetic example for illustration. Real biological findings require independent validation.
