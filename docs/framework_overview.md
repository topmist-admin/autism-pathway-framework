# Framework Overview

## Introduction

This document provides a conceptual overview of the pathway- and network-based framework for analyzing genetic heterogeneity in Autism Spectrum Disorder (ASD).

---

## The Problem: Genetic Heterogeneity

ASD is diagnosed based on behavioral criteria, yet its underlying biology is highly heterogeneous:

- **Hundreds of genes** have been implicated
- Most variants confer **small or context-dependent effects**
- Gene lists **differ across cohorts** and often fail to replicate
- Individuals meeting diagnostic criteria may share **few or no genetic variants**

This mismatch between diagnostic categories and biological mechanisms complicates interpretation and limits translational progress.

---

## The Solution: Pathway-Level Analysis

Rather than focusing on individual genes, this framework aggregates genetic variation at the **biological pathway level**:

```
Variants → Genes → Pathways → Subtypes
```

### Why Pathways?

1. **Biological convergence**: Multiple genetic perturbations may disrupt shared pathways even when specific genes differ
2. **Improved signal detection**: Aggregation increases statistical power
3. **Better interpretability**: Pathways align with biological processes and drug mechanisms
4. **Enhanced replication**: Pathway-level signals are more robust across cohorts

---

## Core Framework Components

### 1. Variant-to-Gene Aggregation

Raw genetic variants are:
- Annotated for predicted functional impact
- Filtered by rarity and quality
- Aggregated into **gene-level burden scores**

### 2. Gene-to-Pathway Aggregation

Gene burdens are:
- Mapped to curated biological pathways
- Aggregated into **pathway disruption scores**
- Normalized for pathway size and background variation

### 3. Network Refinement

Pathway scores are refined using:
- Gene-gene interaction networks (protein interactions, co-expression)
- Network propagation algorithms
- Connectivity-based signal amplification

### 4. Subtype Discovery

Per-individual pathway profiles are used for:
- Unsupervised clustering (GMM, spectral, hierarchical)
- Identification of **genetically coherent subgroups**
- Probabilistic subtype membership assignment

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
│  WES/WGS variants • CNVs • Optional phenotypes • Family data    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PROCESSING LAYER                             │
│  QC • Annotation • Population structure • Batch correction       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AGGREGATION LAYER                             │
│  Variant → Gene burdens → Pathway scores → Network refinement    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ANALYSIS LAYER                               │
│  Subtype clustering • Pathway enrichment • Association testing   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                                │
│  Subtype definitions • Pathway profiles • Hypothesis reports     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Principles

### Transparency

- All analytical choices are documented
- Weighting schemes are explicit and interpretable
- Uncertainty is quantified and reported

### Robustness

- Internal stability via bootstrap and resampling
- External validity via cross-cohort replication
- Negative controls to detect spurious signals

### Interpretability

- Pathway-level results map to biology
- Subtype definitions include characteristic signatures
- Evidence trails link outputs to inputs

---

## Relationship to Therapeutic Hypothesis Generation

Disrupted pathways can be mapped to:
- Known drug targets
- Mechanism categories
- Candidate interventions

This enables **research-only** therapeutic hypothesis generation, prioritizing:
- Biological plausibility
- Mechanistic alignment
- Evidence strength

All hypotheses require independent experimental and clinical validation.

---

## Next Steps

- [Pathway Scoring Concept](pathway_scoring_concept.md) - Detailed scoring methodology
- [Stability and Replication](stability_replication.md) - Validation approaches
- [Limitations](limitations.md) - Known constraints and caveats
