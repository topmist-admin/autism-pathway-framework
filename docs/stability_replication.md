# Stability and Replication

## Overview

Robust research findings must demonstrate both **internal stability** (consistent under perturbation) and **external validity** (replicable in independent data). This document describes validation approaches for the pathway framework.

---

## Why Stability Matters

In high-dimensional biological data:
- Many analytical choices are arbitrary
- Small changes can produce different results
- Overfitting to specific cohorts is common
- Publication bias favors novel findings

Stability testing and replication protect against false discoveries and overinterpretation.

---

## Internal Stability Testing

### 1. Bootstrap Resampling

**Procedure**:
1. Resample individuals with replacement (B times, typically B=1000)
2. Recompute pathway scores and clustering for each bootstrap
3. Assess consistency of:
   - Pathway rankings
   - Subtype assignments
   - Key associations

**Metrics**:
- Bootstrap confidence intervals for pathway scores (95% CI)
- Cluster stability: proportion of times pairs co-cluster (threshold: ≥0.80 for stable clusters)
- Rank correlation of pathway importance across bootstraps (Spearman ρ ≥ 0.70 indicates stability)

### 2. Subsample Stability

**Procedure**:
1. Randomly hold out X% of samples (e.g., 20%)
2. Refit models on remaining 80%
3. Repeat K times
4. Measure consistency across subsamples

**Useful for**:
- Detecting results driven by small subsets
- Identifying influential observations
- Estimating effective sample size

### 3. Parameter Sensitivity

**Procedure**:
1. Vary key parameters within reasonable ranges:
   - Variant weighting schemes
   - Network propagation α
   - Number of clusters K
2. Assess stability of main conclusions

**Stable results**: Key findings should persist when parameters vary by ±20% from defaults. Specifically:
- Variant weighting: α ± 0.2 should not change top-10 pathway rankings
- Network propagation: α ± 0.1 should maintain >80% overlap in enriched genes
- Cluster number K: results should be interpretable for K ± 1

### 4. Initialization Sensitivity

For methods with random initialization (clustering, VAE):
1. Run multiple times with different seeds
2. Compare solutions across runs
3. Report only findings consistent across initializations

---

## Cross-Cohort Replication

### Replication Philosophy

Replication is assessed at the **pathway and subtype level**, not by exact gene overlap.

| Level | Expectation |
|-------|-------------|
| Individual variants | Will differ |
| Specific genes | May differ |
| Pathway enrichments | Should be directionally consistent (Spearman ρ ≥ 0.60) |
| Subtype structure | Should show similar patterns (ARI ≥ 0.50, Jaccard ≥ 0.40) |

### Replication Procedure

1. **Train**: Develop model on discovery cohort
2. **Freeze**: Fix all parameters and thresholds
3. **Apply**: Run frozen pipeline on replication cohort
4. **Compare**: Assess consistency of findings

### What to Compare

**Pathway-level**:
- Rank correlation of pathway disruption scores (Spearman ρ ≥ 0.60 for strong replication)
- Overlap of top-K pathways (Jaccard index ≥ 0.50 for top-20 pathways)
- Directional consistency: ≥75% of significantly disrupted pathways show same direction of effect

**Subtype-level**:
- Similar number of clusters (±1 cluster tolerance)
- Relative size consistency: largest cluster size differs by <15% between cohorts
- Consistent pathway signatures per subtype (cosine similarity ≥ 0.70)
- Comparable phenotype associations (if available): correlation r ≥ 0.50

### Partial Replication

Full replication is rare. Consider:
- **Directional consistency**: Same direction of effect (≥70% agreement on top pathways)
- **Biological coherence**: Same biological themes even if specific pathways differ (semantic similarity ≥ 0.60 using GO hierarchy)
- **Effect size attenuation**: Smaller effects in replication (expected 20-40% attenuation due to winner's curse)

---

## Negative Controls

### 1. Permuted Labels

**Procedure**:
1. Randomly permute case/control labels (or outcome variables)
2. Rerun full analysis
3. Results should be null (no significant pathways)

**Detects**: Confounding, data leakage, analysis bugs

### 2. Randomized Gene Sets

**Procedure**:
1. Replace pathway definitions with random gene sets of same size
2. Rerun pathway scoring
3. Real pathways should outperform random sets

**Detects**: Spurious aggregation effects, pathway size artifacts

### 3. Batch-Only Predictors

**Procedure**:
1. Build model using only batch/technical variables
2. Should not predict biological outcomes

**Detects**: Batch effects masquerading as biology

### 4. Held-Out Chromosomes

**Procedure**:
1. Exclude one chromosome from training
2. Test if predictions generalize

**Detects**: Overfitting to specific genomic regions

---

## Reporting Standards

### Required for Main Findings

| Element | Description |
|---------|-------------|
| Bootstrap CIs | 95% confidence intervals from resampling |
| Stability metrics | Cluster stability, rank correlation |
| Effect sizes | Not just p-values |
| Sample sizes | Per group, after QC |

### Required for Replication

| Element | Description |
|---------|-------------|
| Cohort description | Demographics, ascertainment, sequencing |
| Frozen parameters | All settings fixed from discovery |
| Comparison metrics | Quantitative replication assessment |
| Discrepancies | Honest reporting of differences |

### Required Negative Controls

| Control | Outcome |
|---------|---------|
| Permutation test | P-value from null distribution |
| Random gene sets | Comparison to real pathways |
| Batch prediction | Should be non-predictive |

---

## Common Pitfalls

### 1. Circular Analysis

**Problem**: Using the same data for discovery and validation

**Solution**: Strict separation of training/test sets; pre-registration of hypotheses

### 2. P-Hacking

**Problem**: Testing many comparisons, reporting only significant ones

**Solution**: Correct for multiple testing; report all tests performed

### 3. HARKing

**Problem**: Hypothesizing After Results are Known

**Solution**: Distinguish exploratory vs. confirmatory analyses; pre-register when possible

### 4. Survivorship Bias

**Problem**: Only reporting successful replications

**Solution**: Publish failures; use registered reports

### 5. Overfitting

**Problem**: Complex models that don't generalize

**Solution**: Regularization; cross-validation; simpler models

---

## Checklist for Robust Results

- [ ] Bootstrap confidence intervals computed
- [ ] Cluster stability assessed (if clustering used)
- [ ] Parameter sensitivity tested
- [ ] At least one negative control performed
- [ ] Cross-cohort replication attempted (or acknowledged as limitation)
- [ ] All analytical choices documented
- [ ] Effect sizes reported (not just p-values)
- [ ] Multiple testing correction applied
- [ ] Limitations explicitly stated

---

## Quantitative Thresholds Summary

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Cluster stability (co-clustering) | ≥ 0.80 | Pair of samples clusters together ≥80% of bootstraps |
| Rank correlation (pathways) | Spearman ρ ≥ 0.70 | Stable pathway importance across bootstraps |
| Cross-cohort pathway correlation | Spearman ρ ≥ 0.60 | Strong replication of pathway scores |
| Top-K pathway Jaccard | ≥ 0.50 | At least half of top-20 pathways overlap |
| Directional consistency | ≥ 75% | Same enrichment direction in replication |
| Adjusted Rand Index (ARI) | ≥ 0.50 | Moderate cluster agreement across cohorts |
| Subtype Jaccard overlap | ≥ 0.40 | Acceptable sample assignment agreement |
| Pathway signature similarity | Cosine ≥ 0.70 | Similar subtype pathway profiles |
| Phenotype correlation | r ≥ 0.50 | Comparable clinical associations |
| Effect size attenuation | 20-40% | Expected reduction in replication cohort |
| Parameter sensitivity | ±20% | Results stable within this range |

---

## Summary

| Validation Type | Purpose | Key Methods |
|-----------------|---------|-------------|
| Bootstrap | Uncertainty quantification | Resampling with replacement |
| Stability | Robustness to perturbation | Subsampling, parameter variation |
| Negative controls | Detect artifacts | Permutation, random gene sets |
| Replication | External validity | Independent cohorts |

---

## Next Steps

- [Limitations](limitations.md) - Known constraints of the framework
- [Synthetic Example](../examples/synthetic_example.md) - Worked example with validation
