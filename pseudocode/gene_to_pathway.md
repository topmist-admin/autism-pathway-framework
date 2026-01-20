# Pseudocode: Gene to Pathway Aggregation

## Overview

This module aggregates gene-level burden scores into pathway-level disruption scores.

---

## Input

```
gene_burdens: Map[sample_id → Map[gene_id → burden_score]]
    # Output from variant_to_gene stage

pathway_definitions: Map[pathway_id → Set[gene_id]]
    # Curated pathway gene sets (e.g., from GO, KEGG, Reactome)

gene_weights: Map[gene_id → float] (optional)
    # Gene-level importance weights (e.g., constraint scores)
```

## Output

```
pathway_scores: Map[sample_id → Map[pathway_id → PathwayScore]]

PathwayScore = {
    "raw_score": float,
    "normalized_score": float,
    "z_score": float,
    "contributing_genes": List[{gene_id, contribution}]
}
```

---

## Constants and Configuration

```python
# Minimum pathway size to include
MIN_PATHWAY_SIZE = 5

# Maximum pathway size (very large pathways are often uninformative)
MAX_PATHWAY_SIZE = 500

# Minimum genes with data required
MIN_GENES_WITH_DATA = 3

# Normalization method: "size", "zscore", or "both"
NORMALIZATION_METHOD = "both"
```

---

## Algorithm

### Step 1: Load and Filter Pathways

```python
function load_pathway_definitions(pathway_file):
    """
    Load pathway definitions and apply size filters.
    """
    pathways = load_from_file(pathway_file)  # e.g., GMT format

    filtered_pathways = {}

    for pathway_id, genes in pathways.items():
        # Apply size filters
        if len(genes) < MIN_PATHWAY_SIZE:
            continue
        if len(genes) > MAX_PATHWAY_SIZE:
            continue

        filtered_pathways[pathway_id] = set(genes)

    return filtered_pathways
```

### Step 2: Compute Raw Pathway Score

```python
function compute_raw_pathway_score(gene_burdens_for_sample, pathway_genes, gene_weights=None):
    """
    Aggregate gene burdens into a raw pathway score.

    Input:
        gene_burdens_for_sample: Map[gene_id → burden]
            - Keys are gene identifiers (e.g., HGNC symbols or Ensembl IDs)
            - Values are burden scores (typically >= 0)
            - Missing genes are treated as burden = 0
        pathway_genes: Set[gene_id] in pathway
            - Set of gene identifiers that belong to this pathway
        gene_weights: Optional Map[gene_id → weight]
            - Optional per-gene importance weights (e.g., from constraint scores)
            - If None, all genes weighted equally (weight = 1.0)

    Output:
        raw_score: float - Sum of weighted gene contributions
        contributing_genes: List of (gene, contribution) tuples, sorted by contribution

    Edge Cases Handled:
        - Empty pathway_genes: Returns (0.0, [])
        - No genes with burden > 0: Returns (0.0, [])
        - Gene in pathway but not in gene_burdens: Treated as burden = 0
        - Gene in pathway but not in gene_weights: Uses default weight = 1.0
        - All contributions are 0: Returns (0.0, [])

    Note on genes_with_data counting:
        - A gene "has data" if its burden > 0 in gene_burdens_for_sample
        - Genes with burden = 0 are NOT counted (absence of variants)
        - This differs from genes not present in gene_burdens (unknown/no coverage)
    """
    # Handle empty pathway
    if not pathway_genes:
        return 0.0, []

    total_score = 0.0
    contributions = []

    for gene in pathway_genes:
        burden = gene_burdens_for_sample.get(gene, 0.0)

        # Skip genes with no burden
        # Note: burden = 0 means no qualifying variants, not "unknown"
        if burden == 0:
            continue

        # Apply gene-level weight if provided
        if gene_weights is not None:
            weight = gene_weights.get(gene, 1.0)
        else:
            weight = 1.0

        contribution = burden * weight
        total_score += contribution

        # Only include genes with positive contribution in the list
        # This avoids cluttering output with zero-contribution genes
        if contribution > 0:
            contributions.append({
                "gene_id": gene,
                "burden": burden,
                "weight": weight,
                "contribution": contribution
            })

    # Sort by contribution (descending) for interpretability
    # Top contributors are most informative for understanding pathway scores
    contributions.sort(key=lambda x: x["contribution"], reverse=True)

    return total_score, contributions
```

### Step 3: Size Normalization

```python
function normalize_by_size(raw_score, pathway_size, method="sqrt"):
    """
    Normalize pathway score by pathway size.
    Larger pathways have more opportunity for hits by chance.

    Args:
        raw_score: Unnormalized pathway score
        pathway_size: Number of genes in the pathway
        method: Normalization method ("sqrt", "linear", "log")

    Returns:
        Size-normalized score

    Edge Cases:
        - pathway_size == 0: Returns 0.0 (avoid division by zero)
        - pathway_size == 1: Returns raw_score (no normalization needed)
        - raw_score == 0: Returns 0.0 (regardless of size)

    Common approaches:
        - linear: score / size
            Pro: Simple, intuitive
            Con: May over-penalize large pathways
        - sqrt: score / sqrt(size) [DEFAULT]
            Pro: Balances size effect, widely used in enrichment analysis
            Con: May still favor smaller pathways slightly
        - log: score / log2(size)
            Pro: Gentler normalization for large pathways
            Con: Less intuitive interpretation

    Note: The choice of normalization affects pathway ranking.
    Square root is most common in pathway enrichment literature.
    """
    # Handle edge cases
    if pathway_size == 0:
        return 0.0

    if raw_score == 0:
        return 0.0

    if pathway_size == 1:
        return raw_score

    # Apply normalization based on method
    if method == "sqrt":
        # Square root normalization (balances size effect)
        normalized = raw_score / math.sqrt(pathway_size)
    elif method == "linear":
        # Simple size normalization
        normalized = raw_score / pathway_size
    elif method == "log":
        # Logarithmic normalization (gentler for large pathways)
        normalized = raw_score / math.log2(pathway_size + 1)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized
```

### Step 4: Z-Score Normalization

```python
function compute_pathway_statistics(all_pathway_scores, pathway_id):
    """
    Compute mean and std for a pathway across all samples.
    Used for z-score normalization.
    """
    scores = [
        sample_scores[pathway_id]["raw_score"]
        for sample_scores in all_pathway_scores.values()
        if pathway_id in sample_scores
    ]

    if len(scores) < 2:
        return None, None

    mean_score = statistics.mean(scores)
    std_score = statistics.stdev(scores)

    # Avoid division by zero
    if std_score == 0:
        std_score = 1.0

    return mean_score, std_score


function compute_z_score(raw_score, mean, std):
    """
    Convert raw score to z-score.
    """
    if mean is None or std is None:
        return None

    return (raw_score - mean) / std
```

### Step 5: Main Pathway Scoring Function

```python
function compute_pathway_scores_for_sample(gene_burdens_for_sample, pathway_definitions, gene_weights=None):
    """
    Compute all pathway scores for a single sample.
    """
    pathway_scores = {}

    for pathway_id, pathway_genes in pathway_definitions.items():
        # Check minimum gene coverage
        genes_with_data = sum(
            1 for g in pathway_genes
            if gene_burdens_for_sample.get(g, 0) > 0
        )

        if genes_with_data < MIN_GENES_WITH_DATA:
            continue  # Skip pathways with insufficient data

        # Compute raw score
        raw_score, contributions = compute_raw_pathway_score(
            gene_burdens_for_sample,
            pathway_genes,
            gene_weights
        )

        # Size normalization
        size_normalized = normalize_by_size(raw_score, len(pathway_genes))

        pathway_scores[pathway_id] = {
            "raw_score": raw_score,
            "size_normalized": size_normalized,
            "pathway_size": len(pathway_genes),
            "genes_with_burden": genes_with_data,
            "contributing_genes": contributions[:10]  # Top 10
        }

    return pathway_scores
```

### Step 6: Compute Scores for All Samples

```python
function compute_all_pathway_scores(gene_burdens, pathway_definitions, gene_weights=None):
    """
    Main entry point: compute pathway scores for all samples.
    """
    # First pass: compute raw and size-normalized scores
    all_sample_scores = {}

    for sample_id, sample_gene_burdens in gene_burdens.items():
        scores = compute_pathway_scores_for_sample(
            sample_gene_burdens,
            pathway_definitions,
            gene_weights
        )
        all_sample_scores[sample_id] = scores

    # Second pass: compute z-scores (requires population statistics)
    pathway_stats = {}
    all_pathways = set()
    for scores in all_sample_scores.values():
        all_pathways.update(scores.keys())

    for pathway_id in all_pathways:
        mean, std = compute_pathway_statistics(all_sample_scores, pathway_id)
        pathway_stats[pathway_id] = {"mean": mean, "std": std}

    # Add z-scores to results
    for sample_id, sample_scores in all_sample_scores.items():
        for pathway_id, score_data in sample_scores.items():
            stats = pathway_stats.get(pathway_id, {})
            z = compute_z_score(
                score_data["raw_score"],
                stats.get("mean"),
                stats.get("std")
            )
            score_data["z_score"] = z

    return all_sample_scores, pathway_stats
```

---

## Alternative Aggregation Methods

### Enrichment-Based Scoring

```python
function compute_enrichment_score(gene_burdens_for_sample, pathway_genes, background_genes):
    """
    Compute enrichment of high-burden genes in pathway vs background.
    Similar to gene set enrichment analysis (GSEA) concept.
    """
    # Rank all genes by burden
    all_burdens = list(gene_burdens_for_sample.items())
    all_burdens.sort(key=lambda x: x[1], reverse=True)

    # Compute running enrichment score
    pathway_set = set(pathway_genes)
    n_pathway = len(pathway_set)
    n_background = len(background_genes) - n_pathway

    max_es = 0
    running_es = 0

    for gene, burden in all_burdens:
        if gene in pathway_set:
            running_es += 1.0 / n_pathway
        else:
            running_es -= 1.0 / n_background

        max_es = max(max_es, running_es)

    return max_es
```

### Weighted Mean

```python
function compute_weighted_mean_score(gene_burdens_for_sample, pathway_genes, gene_weights):
    """
    Compute weighted mean of gene burdens in pathway.
    Genes with higher importance weights contribute more.
    """
    total_weight = 0
    weighted_sum = 0

    for gene in pathway_genes:
        burden = gene_burdens_for_sample.get(gene, 0)
        weight = gene_weights.get(gene, 1.0)

        weighted_sum += burden * weight
        total_weight += weight

    if total_weight == 0:
        return 0

    return weighted_sum / total_weight
```

---

## Gene Weighting Schemes

```python
function load_gene_weights(weight_type="constraint"):
    """
    Load gene-level weights based on various metrics.
    """
    if weight_type == "constraint":
        # Use loss-of-function intolerance (pLI, LOEUF, etc.)
        # Genes under stronger constraint are more likely functional
        return load_constraint_scores()

    elif weight_type == "expression":
        # Weight by brain expression level
        return load_brain_expression()

    elif weight_type == "centrality":
        # Weight by network centrality (hub genes)
        return load_network_centrality()

    elif weight_type == "uniform":
        # No weighting
        return None

    else:
        raise ValueError(f"Unknown weight type: {weight_type}")
```

---

## Quality Control

```python
function qc_pathway_scores(pathway_scores, pathway_stats):
    """
    Quality control checks on pathway scores.
    """
    warnings = []

    # Check for pathways with no variance
    for pathway_id, stats in pathway_stats.items():
        if stats["std"] == 0 or stats["std"] is None:
            warnings.append(f"Pathway {pathway_id}: no variance across samples")

    # Check for samples with extreme scores
    for sample_id, scores in pathway_scores.items():
        extreme_pathways = [
            p for p, s in scores.items()
            if s["z_score"] is not None and abs(s["z_score"]) > 5
        ]
        if len(extreme_pathways) > 10:
            warnings.append(f"Sample {sample_id}: many extreme pathway scores")

    return warnings
```

---

## Output Schema

```python
# Per-sample pathway scores
SamplePathwayScores = {
    "sample_id": string,
    "pathways": {
        "GO:0007268": {
            "raw_score": 2.5,
            "size_normalized": 0.35,
            "z_score": 1.8,
            "pathway_size": 50,
            "genes_with_burden": 5,
            "contributing_genes": [
                {"gene_id": "SHANK3", "contribution": 1.0},
                {"gene_id": "NRXN1", "contribution": 0.8},
                ...
            ]
        },
        ...
    }
}

# Cohort-level statistics
PathwayStatistics = {
    "GO:0007268": {
        "mean": 1.2,
        "std": 0.8,
        "n_samples": 500
    },
    ...
}
```

---

## Usage Example

```python
# Load inputs
gene_burdens = load_gene_burdens("gene_burdens.json")
pathway_defs = load_pathway_definitions("pathways.gmt")
gene_weights = load_gene_weights("constraint")

# Compute pathway scores
pathway_scores, pathway_stats = compute_all_pathway_scores(
    gene_burdens,
    pathway_defs,
    gene_weights
)

# QC
warnings = qc_pathway_scores(pathway_scores, pathway_stats)

# Save output
save_pathway_scores(pathway_scores, "pathway_scores.json")
save_pathway_stats(pathway_stats, "pathway_statistics.json")
```

---

## Next Steps

- [Network Refinement](network_refinement.md) - Refine pathway scores using gene networks
- [Subtype Clustering](subtype_clustering.md) - Cluster samples by pathway profiles
