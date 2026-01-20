# Pseudocode: Subtype Clustering

## Overview

This module identifies genetically coherent subgroups (subtypes) from per-individual pathway disruption profiles using unsupervised clustering methods.

---

## Input

```
pathway_profiles: Map[sample_id → Map[pathway_id → score]]
    # Per-sample pathway disruption scores (from previous stages)

phenotype_data: Map[sample_id → Map[phenotype_id → value]] (optional)
    # Clinical/behavioral phenotypes for validation
```

## Output

```
subtype_assignments: Map[sample_id → SubtypeAssignment]

SubtypeAssignment = {
    "hard_label": int,           # Most likely subtype
    "probabilities": List[float], # Probability per subtype
    "uncertainty": float          # Entropy of assignment
}

subtype_profiles: Map[subtype_id → SubtypeProfile]

SubtypeProfile = {
    "characteristic_pathways": List[{pathway_id, importance}],
    "size": int,
    "mean_profile": Map[pathway_id → float]
}
```

---

## Constants and Configuration

```python
# Clustering parameters
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10
CLUSTER_SELECTION_METRIC = "bic"  # "bic", "silhouette", or "stability"

# Stability testing
N_BOOTSTRAP = 100
STABILITY_THRESHOLD = 0.7

# Dimensionality reduction (if needed)
MAX_DIMENSIONS = 50
VARIANCE_THRESHOLD = 0.9  # Keep components explaining 90% variance
```

---

## Algorithm

### Step 1: Prepare Feature Matrix

```python
function prepare_feature_matrix(pathway_profiles):
    """
    Convert pathway profiles to a sample × pathway feature matrix.
    """
    samples = list(pathway_profiles.keys())

    # Get all pathways across all samples
    all_pathways = set()
    for profile in pathway_profiles.values():
        all_pathways.update(profile.keys())
    pathways = sorted(all_pathways)

    # Build matrix
    n_samples = len(samples)
    n_pathways = len(pathways)
    X = zeros((n_samples, n_pathways))

    for i, sample_id in enumerate(samples):
        profile = pathway_profiles[sample_id]
        for j, pathway_id in enumerate(pathways):
            X[i, j] = profile.get(pathway_id, 0.0)

    return X, samples, pathways


function standardize_features(X):
    """
    Standardize features to zero mean and unit variance.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero

    X_standardized = (X - mean) / std

    return X_standardized, mean, std
```

### Step 2: Dimensionality Reduction (Optional)

```python
function reduce_dimensions(X, method="pca"):
    """
    Reduce dimensionality for clustering stability.
    High-dimensional data can cause clustering issues.
    """
    if method == "pca":
        return pca_reduction(X)
    elif method == "nmf":
        return nmf_reduction(X)
    elif method == "umap":
        return umap_reduction(X)
    else:
        return X  # No reduction


function pca_reduction(X):
    """
    PCA with automatic component selection.

    Selects the minimum number of components needed to explain
    VARIANCE_THRESHOLD (default 90%) of the total variance.
    """
    # Fit PCA
    pca = PCA()
    pca.fit(X)

    # Select components explaining VARIANCE_THRESHOLD of variance
    cumulative_var = cumsum(pca.explained_variance_ratio_)

    # Find first index where cumulative variance exceeds threshold
    # np.where returns indices where condition is True; take first one
    indices_above_threshold = where(cumulative_var >= VARIANCE_THRESHOLD)[0]

    if len(indices_above_threshold) > 0:
        n_components = indices_above_threshold[0] + 1  # +1 because indices are 0-based
    else:
        n_components = len(cumulative_var)  # Use all components if threshold not reached

    n_components = min(n_components, MAX_DIMENSIONS)
    n_components = max(n_components, 1)  # At least 1 component

    # Transform
    X_reduced = pca.transform(X)[:, :n_components]

    return X_reduced, pca, n_components
```

### Step 3: Gaussian Mixture Model Clustering

```python
function fit_gmm(X, n_clusters):
    """
    Fit Gaussian Mixture Model for soft clustering.

    Returns model and quality metrics.
    """
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",  # or "diag" for simpler model
        n_init=10,               # Multiple initializations
        random_state=42
    )

    gmm.fit(X)

    # Compute metrics
    bic = gmm.bic(X)
    aic = gmm.aic(X)

    # Get assignments
    labels = gmm.predict(X)
    probabilities = gmm.predict_proba(X)

    return {
        "model": gmm,
        "labels": labels,
        "probabilities": probabilities,
        "bic": bic,
        "aic": aic
    }


function select_n_clusters_gmm(X):
    """
    Select optimal number of clusters using BIC/AIC.
    """
    results = {}

    for k in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
        result = fit_gmm(X, k)
        results[k] = result

    # Select by BIC (lower is better)
    best_k = min(results.keys(), key=lambda k: results[k]["bic"])

    return best_k, results[best_k], results
```

### Step 4: Alternative Clustering Methods

```python
function fit_spectral_clustering(X, n_clusters):
    """
    Spectral clustering on similarity graph.
    Good for non-convex cluster shapes.
    """
    # Build similarity matrix
    similarity = compute_rbf_similarity(X)

    # Spectral clustering
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        n_init=10
    )
    labels = sc.fit_predict(similarity)

    return labels


function fit_hierarchical_clustering(X, n_clusters):
    """
    Agglomerative hierarchical clustering.
    Produces interpretable dendrogram.
    """
    hc = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"  # Minimize within-cluster variance
    )
    labels = hc.fit_predict(X)

    return labels


function compute_rbf_similarity(X, gamma=None):
    """
    RBF (Gaussian) similarity kernel.
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    # Pairwise squared distances
    dist_sq = pairwise_squared_distances(X)

    # RBF kernel
    similarity = exp(-gamma * dist_sq)

    return similarity
```

### Step 5: Cluster Stability Assessment

```python
function assess_cluster_stability(X, n_clusters, n_bootstrap=N_BOOTSTRAP):
    """
    Assess stability of clustering via bootstrap resampling.

    Returns stability score (proportion of pairs consistently co-clustered).
    """
    n_samples = X.shape[0]
    co_cluster_counts = zeros((n_samples, n_samples))
    co_sample_counts = zeros((n_samples, n_samples))

    for b in range(n_bootstrap):
        # Bootstrap sample
        indices = random_choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]

        # Cluster
        gmm_result = fit_gmm(X_boot, n_clusters)
        labels_boot = gmm_result["labels"]

        # Update co-clustering matrix
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Check if both i and j were sampled
                i_in = i in indices
                j_in = j in indices

                if i_in and j_in:
                    co_sample_counts[i, j] += 1
                    # Check if they co-cluster
                    i_idx = list(indices).index(i)
                    j_idx = list(indices).index(j)
                    if labels_boot[i_idx] == labels_boot[j_idx]:
                        co_cluster_counts[i, j] += 1

    # Compute stability
    mask = co_sample_counts > 0
    stability_matrix = zeros_like(co_cluster_counts)
    stability_matrix[mask] = co_cluster_counts[mask] / co_sample_counts[mask]

    # Overall stability: mean of off-diagonal
    mean_stability = stability_matrix[mask].mean()

    return mean_stability, stability_matrix


function select_k_by_stability(X):
    """
    Select number of clusters maximizing stability.
    """
    stabilities = {}

    for k in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
        stability, _ = assess_cluster_stability(X, k)
        stabilities[k] = stability

    # Select k with highest stability above threshold
    valid_k = [k for k, s in stabilities.items() if s >= STABILITY_THRESHOLD]

    if not valid_k:
        # Fall back to k with highest stability
        best_k = max(stabilities.keys(), key=lambda k: stabilities[k])
    else:
        # Among stable solutions, prefer simpler (smaller k)
        best_k = min(valid_k)

    return best_k, stabilities
```

### Step 6: Characterize Subtypes

```python
function characterize_subtypes(X, labels, pathways):
    """
    Compute characteristic pathway profiles for each subtype.
    """
    unique_labels = sorted(set(labels))
    subtype_profiles = {}

    for subtype_id in unique_labels:
        # Get samples in this subtype
        mask = labels == subtype_id
        X_subtype = X[mask]

        # Compute mean profile
        mean_profile = X_subtype.mean(axis=0)

        # Compute overall mean
        overall_mean = X.mean(axis=0)

        # Characteristic pathways: most different from overall
        pathway_importance = []
        for j, pathway_id in enumerate(pathways):
            diff = mean_profile[j] - overall_mean[j]
            pathway_importance.append({
                "pathway_id": pathway_id,
                "mean_score": mean_profile[j],
                "difference": diff,
                "importance": abs(diff)
            })

        # Sort by importance
        pathway_importance.sort(key=lambda x: -x["importance"])

        subtype_profiles[subtype_id] = {
            "size": mask.sum(),
            "fraction": mask.mean(),
            "mean_profile": {pathways[j]: mean_profile[j] for j in range(len(pathways))},
            "characteristic_pathways": pathway_importance[:20]  # Top 20
        }

    return subtype_profiles
```

### Step 7: Compute Assignment Uncertainty

```python
function compute_assignment_uncertainty(probabilities):
    """
    Compute entropy-based uncertainty for each sample's assignment.

    High entropy = uncertain assignment
    Low entropy = confident assignment
    """
    uncertainties = []

    for prob_vector in probabilities:
        # Shannon entropy
        entropy = -sum(p * log(p + 1e-10) for p in prob_vector)

        # Normalize by max entropy (uniform distribution)
        max_entropy = log(len(prob_vector))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        uncertainties.append(normalized_entropy)

    return uncertainties
```

### Step 8: Main Clustering Pipeline

```python
function cluster_samples(pathway_profiles, method="gmm"):
    """
    Main entry point for subtype discovery.
    """
    # Prepare data
    X, samples, pathways = prepare_feature_matrix(pathway_profiles)
    X_std, mean, std = standardize_features(X)

    # Optional dimensionality reduction
    X_reduced, pca_model, n_components = reduce_dimensions(X_std, method="pca")

    # Select number of clusters
    if CLUSTER_SELECTION_METRIC == "bic":
        best_k, best_result, all_results = select_n_clusters_gmm(X_reduced)
    elif CLUSTER_SELECTION_METRIC == "stability":
        best_k, stabilities = select_k_by_stability(X_reduced)
        best_result = fit_gmm(X_reduced, best_k)

    # Final clustering
    labels = best_result["labels"]
    probabilities = best_result["probabilities"]

    # Compute uncertainties
    uncertainties = compute_assignment_uncertainty(probabilities)

    # Build assignments
    assignments = {}
    for i, sample_id in enumerate(samples):
        assignments[sample_id] = {
            "hard_label": int(labels[i]),
            "probabilities": probabilities[i].tolist(),
            "uncertainty": uncertainties[i]
        }

    # Characterize subtypes
    subtype_profiles = characterize_subtypes(X_std, labels, pathways)

    return {
        "assignments": assignments,
        "subtype_profiles": subtype_profiles,
        "n_clusters": best_k,
        "model": best_result["model"],
        "metrics": {
            "bic": best_result["bic"],
            "aic": best_result["aic"]
        }
    }
```

---

## Validation with Phenotypes

```python
function validate_with_phenotypes(assignments, phenotype_data):
    """
    Assess whether genetic subtypes correspond to phenotypic differences.
    This is EXPLORATORY, not confirmatory.
    """
    results = {}

    # Get samples with both subtype and phenotype data
    common_samples = set(assignments.keys()) & set(phenotype_data.keys())

    for phenotype_id in get_all_phenotypes(phenotype_data):
        # Extract values by subtype
        subtype_values = defaultdict(list)

        for sample_id in common_samples:
            subtype = assignments[sample_id]["hard_label"]
            value = phenotype_data[sample_id].get(phenotype_id)

            if value is not None:
                subtype_values[subtype].append(value)

        # Statistical test (ANOVA for continuous, chi-square for categorical)
        if is_continuous(phenotype_id):
            stat, pvalue = kruskal_wallis_test(list(subtype_values.values()))
        else:
            stat, pvalue = chi_square_test(subtype_values)

        results[phenotype_id] = {
            "statistic": stat,
            "p_value": pvalue,
            "subtype_means": {k: mean(v) for k, v in subtype_values.items()}
        }

    # Multiple testing correction
    pvalues = [r["p_value"] for r in results.values()]
    adjusted = benjamini_hochberg(pvalues)

    for i, phenotype_id in enumerate(results.keys()):
        results[phenotype_id]["adjusted_p"] = adjusted[i]

    return results
```

---

## Cross-Cohort Replication

```python
function replicate_in_new_cohort(model, new_pathway_profiles, original_pca, original_standardization):
    """
    Apply trained clustering model to new cohort.

    IMPORTANT: All preprocessing parameters must be frozen from original cohort.
    """
    # Prepare data using FROZEN parameters
    X_new, samples, pathways = prepare_feature_matrix(new_pathway_profiles)

    mean, std = original_standardization
    X_std = (X_new - mean) / std

    # Project onto original PCA space
    X_reduced = original_pca.transform(X_std)

    # Predict cluster assignments
    labels = model.predict(X_reduced)
    probabilities = model.predict_proba(X_reduced)

    # Build assignments
    assignments = {}
    for i, sample_id in enumerate(samples):
        assignments[sample_id] = {
            "hard_label": int(labels[i]),
            "probabilities": probabilities[i].tolist()
        }

    return assignments
```

---

## Output Schema

```python
# Clustering result
ClusteringResult = {
    "n_clusters": int,
    "assignments": {
        "SAMPLE_001": {
            "hard_label": 2,
            "probabilities": [0.1, 0.15, 0.7, 0.05],
            "uncertainty": 0.45
        },
        ...
    },
    "subtype_profiles": {
        0: {
            "size": 150,
            "fraction": 0.30,
            "characteristic_pathways": [
                {"pathway_id": "GO:0007268", "difference": 1.2, "importance": 1.2},
                ...
            ]
        },
        ...
    },
    "metrics": {
        "bic": 15234.5,
        "aic": 14890.2,
        "stability": 0.82
    }
}
```

---

## Usage Example

```python
# Load pathway profiles
pathway_profiles = load_pathway_scores("pathway_scores.json")

# Cluster samples
result = cluster_samples(pathway_profiles, method="gmm")

print(f"Found {result['n_clusters']} subtypes")

# Examine subtype profiles
for subtype_id, profile in result["subtype_profiles"].items():
    print(f"\nSubtype {subtype_id}: {profile['size']} samples ({profile['fraction']*100:.1f}%)")
    print("Top characteristic pathways:")
    for p in profile["characteristic_pathways"][:5]:
        print(f"  {p['pathway_id']}: diff={p['difference']:.2f}")

# Optional: validate with phenotypes
if phenotype_data is not None:
    validation = validate_with_phenotypes(result["assignments"], phenotype_data)

    significant = [p for p, r in validation.items() if r["adjusted_p"] < 0.05]
    print(f"\nPhenotypes with subtype differences: {significant}")

# Save results
save_clustering_result(result, "subtype_clustering_result.json")
```

---

## Interpretation Guidelines

1. **Subtypes are hypotheses**, not definitive biological entities
2. **Uncertainty matters**: Samples with high uncertainty may not belong clearly to any subtype
3. **Validate externally**: Replicate in independent cohorts before drawing conclusions
4. **Avoid circular reasoning**: Don't use phenotypes to select K, then claim phenotype associations
5. **Report all K tested**: Don't hide that you tried multiple values

---

## Next Steps

- [Synthetic Example](../examples/synthetic_example.md) - Complete worked example
- [Stability and Replication](../docs/stability_replication.md) - Validation approaches
