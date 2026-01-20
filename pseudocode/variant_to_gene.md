# Pseudocode: Variant to Gene Aggregation

## Overview

This module transforms raw genetic variants into gene-level burden scores for each individual. It supports multiple weighting schemes including consequence-based, pathogenicity score-based (CADD, REVEL), and allele frequency-based weighting.

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
  - consequence: string (e.g., "missense_variant", "frameshift_variant", "synonymous_variant")
  - gnomad_af: float (allele frequency in gnomAD population)
  - cadd_phred: float (CADD phred-scaled score, higher = more deleterious)
  - revel_score: float (REVEL score for missense variants, 0-1 scale)
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
# These map to VEP/Ensembl consequence terms
CONSEQUENCE_WEIGHTS = {
    # Loss-of-function (highest weight)
    "frameshift_variant": 1.0,
    "stop_gained": 1.0,
    "splice_donor_variant": 1.0,
    "splice_acceptor_variant": 1.0,
    "start_lost": 1.0,

    # Moderate impact
    "missense_variant": 0.5,          # Base weight, may be adjusted by CADD/REVEL
    "inframe_insertion": 0.3,
    "inframe_deletion": 0.3,
    "protein_altering_variant": 0.5,

    # Low impact
    "splice_region_variant": 0.2,
    "synonymous_variant": 0.0,

    # Modifier (usually excluded)
    "5_prime_UTR_variant": 0.1,
    "3_prime_UTR_variant": 0.1,
    "intron_variant": 0.0,
    "intergenic_variant": 0.0,
}

# Allele frequency thresholds
AF_ULTRA_RARE = 0.0001   # < 0.01%
AF_RARE = 0.001          # < 0.1%
AF_MAX = 0.01            # < 1% (exclude common variants)

# Quality thresholds
MIN_DEPTH = 10
MIN_GQ = 20

# CADD score configuration
USE_CADD_WEIGHTING = True
CADD_THRESHOLD = 20.0        # Minimum CADD phred to include missense
CADD_WEIGHT_SCALE = 0.05     # Weight = CADD_phred * scale (e.g., CADD=30 -> 1.5)

# REVEL score configuration (for missense variants)
USE_REVEL_WEIGHTING = True
REVEL_THRESHOLD = 0.5        # Minimum REVEL score to consider damaging

# Allele frequency weighting
USE_AF_WEIGHTING = False     # If True, weight by rarity
AF_WEIGHT_BETA = 1.0         # Weight = (1 - AF)^beta

# Aggregation method
AGGREGATION_METHOD = "weighted_sum"  # Options: "weighted_sum", "max", "count"
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
    Returns a base weight reflecting predicted deleteriousness.

    For missense variants, the base weight may be overridden
    by CADD or REVEL scores in the compute_weight function.
    """
    consequence = variant.consequence

    # Look up weight for consequence type
    if consequence in CONSEQUENCE_WEIGHTS:
        return CONSEQUENCE_WEIGHTS[consequence]

    # Default for unknown consequences
    return 0.0
```

### Step 3: Compute Variant Weight (Enhanced)

```python
function compute_weight(variant):
    """
    Compute the final weight for a variant, incorporating:
    1. Base consequence weight
    2. CADD score weighting (for missense)
    3. REVEL score weighting (for missense)
    4. Allele frequency weighting (optional)

    This provides more nuanced weighting than simple consequence categories.
    """
    # Start with base consequence weight
    weight = classify_impact(variant)

    # For missense variants, apply pathogenicity score weighting
    if variant.consequence == "missense_variant":

        # CADD-based weighting: use CADD phred score directly
        if USE_CADD_WEIGHTING and variant.cadd_phred is not None:
            # Filter out missense below CADD threshold
            if variant.cadd_phred < CADD_THRESHOLD:
                return 0.0  # Exclude low-impact missense

            # Scale CADD score to weight
            # Example: CADD=30 with scale=0.05 -> weight=1.5
            weight = variant.cadd_phred * CADD_WEIGHT_SCALE

        # REVEL-based weighting: ensemble predictor for missense
        if USE_REVEL_WEIGHTING and variant.revel_score is not None:
            if variant.revel_score >= REVEL_THRESHOLD:
                # Use REVEL score directly if above threshold
                # Take maximum of CADD-derived and REVEL weight
                weight = max(weight, variant.revel_score)

    # Allele frequency weighting (optional)
    # Upweights rare variants, downweights common variants
    if USE_AF_WEIGHTING and variant.gnomad_af is not None:
        # Weight = (1 - AF)^beta
        # For beta=1: AF=0.01 -> 0.99, AF=0.001 -> 0.999
        af_weight = (1.0 - variant.gnomad_af) ** AF_WEIGHT_BETA
        weight *= af_weight

    return max(0.0, weight)
```

### Step 4: Apply Rarity Filtering

```python
function passes_rarity_filter(variant):
    """
    Check if variant passes allele frequency filters.
    We typically only consider rare variants for burden analysis.
    """
    af = variant.gnomad_af

    # If no AF data, assume rare (could be novel variant)
    if af is None:
        return True

    # Exclude common variants
    if af >= AF_MAX:
        return False

    return True
```

### Step 5: Compute Gene Burden

```python
function compute_gene_burden(variants_for_sample):
    """
    Aggregate variant-level scores into gene-level burden scores.

    Input: List of variants for a single sample
    Output: Map[gene_id → burden_score]

    Supports multiple aggregation methods:
    - weighted_sum: Sum of all variant weights (default)
    - max: Maximum variant weight per gene
    - count: Simple count of qualifying variants
    """
    gene_scores = defaultdict(float)
    gene_variant_counts = defaultdict(int)

    for variant in variants_for_sample:
        gene = variant.gene

        if gene is None or gene == "":
            continue  # Skip intergenic variants

        # Check rarity filter
        if not passes_rarity_filter(variant):
            continue

        # Compute variant weight using enhanced method
        variant_weight = compute_weight(variant)

        if variant_weight <= 0:
            continue

        # Aggregate to gene based on configured method
        if AGGREGATION_METHOD == "weighted_sum":
            gene_scores[gene] += variant_weight
        elif AGGREGATION_METHOD == "max":
            gene_scores[gene] = max(gene_scores[gene], variant_weight)
        elif AGGREGATION_METHOD == "count":
            gene_scores[gene] += 1

        gene_variant_counts[gene] += 1

    return gene_scores
```

### Step 6: Process All Samples

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
