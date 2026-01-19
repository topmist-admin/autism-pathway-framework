# Pseudocode: Variant to Gene Aggregation

## Overview

This module transforms raw genetic variants into gene-level burden scores for each individual.

---

## Input

```
variants: List of annotated variants per individual
  - sample_id: string
  - chrom: string
  - pos: integer
  - ref: string
  - alt: string
  - gene: string
  - consequence: string (e.g., "missense", "frameshift", "synonymous")
  - af_population: float (allele frequency in reference population)
  - impact_score: float (predicted functional impact, e.g., CADD)
  - quality_metrics: {dp, gq, filter_status}
```

## Output

```
gene_burdens: Map[sample_id → Map[gene_id → burden_score]]
```

---

## Constants and Configuration

```python
# Consequence weights (higher = more damaging)
CONSEQUENCE_WEIGHTS = {
    "frameshift": 1.0,
    "stop_gained": 1.0,
    "splice_donor": 1.0,
    "splice_acceptor": 1.0,
    "start_lost": 0.9,
    "stop_lost": 0.9,
    "missense_damaging": 0.7,  # Based on impact predictor
    "missense_benign": 0.2,
    "inframe_indel": 0.5,
    "synonymous": 0.0,
    "intronic": 0.0,
    "intergenic": 0.0,
}

# Allele frequency thresholds
AF_ULTRA_RARE = 0.0001   # < 0.01%
AF_RARE = 0.01           # < 1%

# Quality thresholds
MIN_DEPTH = 10
MIN_GQ = 20

# Impact score threshold for "damaging" missense
DAMAGING_THRESHOLD = 20  # e.g., CADD >= 20
```

---

## Algorithm

### Step 1: Quality Control Filter

```python
function filter_variants(variants):
    """
    Remove low-quality variants that may be sequencing artifacts.
    """
    filtered = []

    for variant in variants:
        # Check quality metrics
        if variant.quality_metrics.dp < MIN_DEPTH:
            continue
        if variant.quality_metrics.gq < MIN_GQ:
            continue
        if variant.quality_metrics.filter_status != "PASS":
            continue

        # Check allele frequency (keep rare variants only)
        if variant.af_population >= AF_RARE:
            continue

        filtered.append(variant)

    return filtered
```

### Step 2: Classify Variant Impact

```python
function classify_impact(variant):
    """
    Determine the functional impact category for a variant.
    Returns a weight reflecting predicted deleteriousness.
    """
    consequence = variant.consequence

    # Handle missense variants specially
    if consequence == "missense":
        if variant.impact_score >= DAMAGING_THRESHOLD:
            return CONSEQUENCE_WEIGHTS["missense_damaging"]
        else:
            return CONSEQUENCE_WEIGHTS["missense_benign"]

    # Look up weight for other consequences
    if consequence in CONSEQUENCE_WEIGHTS:
        return CONSEQUENCE_WEIGHTS[consequence]

    # Default for unknown consequences
    return 0.0
```

### Step 3: Apply Rarity Weighting

```python
function rarity_weight(af):
    """
    Weight variants by rarity. Rarer variants get higher weight.
    """
    if af < AF_ULTRA_RARE:
        return 1.0  # Ultra-rare: full weight
    elif af < AF_RARE:
        return 0.5  # Rare: half weight
    else:
        return 0.0  # Common: exclude
```

### Step 4: Compute Gene Burden

```python
function compute_gene_burden(variants_for_sample):
    """
    Aggregate variant-level scores into gene-level burden scores.

    Input: List of variants for a single sample
    Output: Map[gene_id → burden_score]
    """
    gene_scores = defaultdict(float)

    for variant in variants_for_sample:
        gene = variant.gene

        if gene is None or gene == "":
            continue  # Skip intergenic variants

        # Compute variant weight
        impact_weight = classify_impact(variant)
        rarity = rarity_weight(variant.af_population)

        # Combined weight
        variant_weight = impact_weight * rarity

        # Aggregate to gene (additive model)
        gene_scores[gene] += variant_weight

    return gene_scores
```

### Step 5: Process All Samples

```python
function compute_all_gene_burdens(all_variants):
    """
    Main entry point: compute gene burdens for all samples.

    Input: All variants across all samples
    Output: Map[sample_id → Map[gene_id → burden_score]]
    """
    # Group variants by sample
    variants_by_sample = group_by(all_variants, key="sample_id")

    gene_burdens = {}

    for sample_id, sample_variants in variants_by_sample.items():
        # Quality filter
        filtered = filter_variants(sample_variants)

        # Compute burden
        burdens = compute_gene_burden(filtered)

        gene_burdens[sample_id] = burdens

    return gene_burdens
```

---

## Alternative Aggregation Models

### Maximum Model

Instead of summing, take the maximum variant weight per gene:

```python
function compute_gene_burden_max(variants_for_sample):
    """
    Use maximum variant weight instead of sum.
    Useful when multiple hits in same gene are redundant.
    """
    gene_scores = defaultdict(float)

    for variant in variants_for_sample:
        gene = variant.gene
        variant_weight = classify_impact(variant) * rarity_weight(variant.af_population)

        # Take maximum instead of sum
        gene_scores[gene] = max(gene_scores[gene], variant_weight)

    return gene_scores
```

### Count Model

Simple count of qualifying variants:

```python
function compute_gene_burden_count(variants_for_sample):
    """
    Count qualifying variants per gene (unweighted).
    """
    gene_counts = defaultdict(int)

    for variant in variants_for_sample:
        gene = variant.gene
        impact = classify_impact(variant)

        if impact > 0:  # Only count potentially damaging
            gene_counts[gene] += 1

    return gene_counts
```

---

## Quality Control Checks

```python
function qc_gene_burdens(gene_burdens):
    """
    Perform quality control on computed gene burdens.
    Flag samples with suspicious patterns.
    """
    warnings = []

    for sample_id, burdens in gene_burdens.items():
        total_burden = sum(burdens.values())
        num_genes_hit = len([g for g, b in burdens.items() if b > 0])

        # Flag samples with unusually high burden (possible QC issue)
        if total_burden > TOTAL_BURDEN_THRESHOLD:
            warnings.append(f"Sample {sample_id}: unusually high total burden")

        # Flag samples with too few genes (possible low coverage)
        if num_genes_hit < MIN_GENES_HIT:
            warnings.append(f"Sample {sample_id}: very few genes with variants")

    return warnings
```

---

## Output Schema

```python
# Output data structure
GeneburdenResult = {
    "sample_id": string,
    "gene_burdens": {
        "GENE1": 0.8,
        "GENE2": 1.5,
        "GENE3": 0.2,
        ...
    },
    "metadata": {
        "total_variants_input": int,
        "variants_after_qc": int,
        "genes_with_burden": int,
        "total_burden": float
    }
}
```

---

## Usage Example

```python
# Load variants (from VCF or database)
variants = load_annotated_variants("cohort_variants.vcf.gz")

# Compute gene burdens
gene_burdens = compute_all_gene_burdens(variants)

# QC check
warnings = qc_gene_burdens(gene_burdens)
if warnings:
    print("QC warnings:", warnings)

# Output for next stage
save_gene_burdens(gene_burdens, "gene_burdens.json")
```

---

## Next Steps

- [Gene to Pathway](gene_to_pathway.md) - Aggregate gene burdens into pathway scores
