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
- Bootstrap confidence intervals for pathway scores
- Cluster stability (proportion of times pairs co-cluster)
- Rank correlation of pathway importance across bootstraps

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

**Stable results** should be robust to modest parameter changes.

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
| Pathway enrichments | Should be directionally consistent |
| Subtype structure | Should show similar patterns |

### Replication Procedure

1. **Train**: Develop model on discovery cohort
2. **Freeze**: Fix all parameters and thresholds
3. **Apply**: Run frozen pipeline on replication cohort
4. **Compare**: Assess consistency of findings

### What to Compare

**Pathway-level**:
- Rank correlation of pathway disruption scores
- Overlap of top-K pathways
- Directional consistency (same pathways enriched/depleted)

**Subtype-level**:
- Similar number and relative size of clusters
- Consistent pathway signatures per subtype
- Comparable phenotype associations (if available)

### Partial Replication

Full replication is rare. Consider:
- **Directional consistency**: Same direction of effect
- **Biological coherence**: Same biological themes even if specific pathways differ
- **Effect size attenuation**: Smaller effects in replication (expected due to winner's curse)

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
