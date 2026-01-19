# Pathway Scoring Concept

## Overview

This document describes the conceptual approach to computing pathway disruption scores from genetic variation data.

---

## The Aggregation Hierarchy

```
Variants → Gene Burden → Pathway Score → Network-Refined Score
```

Each level aggregates information from the level below, progressively moving from raw genetic data to biologically interpretable signals.

---

## Level 1: Variant Annotation

### Functional Categories

Variants are classified by predicted functional impact:

| Category | Description | Typical Weight |
|----------|-------------|----------------|
| Loss-of-function (LoF) | Stop-gain, frameshift, splice-site | Highest |
| Damaging missense | Predicted deleterious amino acid changes | High |
| Other missense | Missense with uncertain impact | Moderate |
| Synonymous | No amino acid change | Low/None |
| Non-coding | Regulatory, intronic, intergenic | Context-dependent |

### Rarity Filtering

- **Ultra-rare**: Allele frequency < 0.01% in reference populations
- **Rare**: Allele frequency < 1%
- **Common**: Allele frequency ≥ 1% (typically excluded from burden analysis)

Rarer variants generally receive higher weights due to stronger expected effect sizes.

---

## Level 2: Gene Burden Scores

### Basic Formula

For each individual `i` and gene `g`:

```
GeneBurden(i, g) = Σ w(v) × I(v ∈ variants(i, g))
```

Where:
- `w(v)` = weight for variant `v` based on functional impact and rarity
- `I()` = indicator function
- `variants(i, g)` = variants in gene `g` carried by individual `i`

### Weighting Considerations

| Factor | Rationale |
|--------|-----------|
| Functional impact | LoF variants more likely to disrupt function |
| Allele frequency | Rare variants under stronger selection |
| Constraint metrics | Genes intolerant to variation more likely functional |
| Inheritance | De novo variants have higher prior probability |

### Example Weighting Scheme

```
LoF (ultra-rare):           1.0
LoF (rare):                 0.8
Damaging missense (ultra-rare): 0.6
Damaging missense (rare):   0.4
Other missense:             0.2
```

**Note**: These are illustrative values. Actual weights should be calibrated empirically.

---

## Level 3: Pathway Disruption Scores

### Pathway Definitions

Pathways are sourced from curated databases:
- Gene Ontology (GO) biological processes
- KEGG pathways
- Reactome pathways
- Custom curated gene sets

### Basic Aggregation

For each individual `i` and pathway `p`:

```
PathwayScore(i, p) = Σ GeneBurden(i, g) × w(g, p)
                     g ∈ pathway(p)
```

Where `w(g, p)` = gene-level weight within pathway (optional).

### Normalization

Raw scores must be normalized to account for:

1. **Pathway size**: Larger pathways have more opportunities for hits
2. **Sequencing depth**: More variants called = higher raw scores
3. **Background variation**: Some pathways are more variable in general

#### Size Normalization

```
NormalizedScore(i, p) = PathwayScore(i, p) / sqrt(|pathway(p)|)
```

#### Z-score Normalization

```
Z(i, p) = (PathwayScore(i, p) - μ(p)) / σ(p)
```

Where μ(p) and σ(p) are estimated from the cohort or reference population.

---

## Level 4: Network Refinement

### Rationale

Pathway membership alone may miss:
- Genes with functional relationships not captured in pathway definitions
- Weak signals that become significant through network connectivity
- Hub genes that connect multiple pathways

### Network Propagation

Using Random Walk with Restart (RWR):

```
p(t+1) = (1 - α) × W × p(t) + α × p(0)
```

Where:
- `p(0)` = initial gene scores (from burden analysis)
- `W` = normalized adjacency matrix of gene-gene network
- `α` = restart probability (typically 0.3-0.7)
- Iterate until convergence

### Network Sources

| Source | Edge Type |
|--------|-----------|
| Protein-protein interactions | Physical binding |
| Co-expression networks | Expression correlation |
| Regulatory networks | TF-target relationships |
| Pathway co-membership | Shared pathway annotation |

### Avoiding Pitfalls

- **Hub bias**: High-degree genes can dominate; apply degree correction
- **Over-smoothing**: Too much propagation loses specificity; tune α carefully
- **Network incompleteness**: Networks are biased toward well-studied genes

---

## Practical Considerations

### Missing Data

- Genes with no variants receive score = 0
- Pathways with no covered genes are excluded
- Imputation is generally not recommended

### Quality Control

- Remove individuals with outlier total burden (sequencing artifacts)
- Exclude pathways with < N genes covered
- Check for batch effects in pathway scores

### Interpretation

Pathway scores represent **relative enrichment** of genetic perturbation, not absolute disruption. High scores indicate:
- More rare, damaging variants in pathway genes
- Relative to other pathways in that individual
- Relative to other individuals in the cohort

---

## Summary Table

| Level | Input | Output | Key Choices |
|-------|-------|--------|-------------|
| Variant | VCF/gVCF | Annotated variants | Impact predictors, frequency thresholds |
| Gene | Annotated variants | Gene burden scores | Weighting scheme, aggregation method |
| Pathway | Gene burdens | Pathway scores | Pathway database, normalization |
| Network | Pathway scores | Refined scores | Network source, propagation parameters |

---

## Next Steps

- [Stability and Replication](stability_replication.md) - Validating pathway scores
- [Pseudocode: Gene to Pathway](../pseudocode/gene_to_pathway.md) - Implementation details
