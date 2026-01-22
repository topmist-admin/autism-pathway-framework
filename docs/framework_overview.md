# Framework Overview

## Implementation Status

| Phase | Description | Status | Modules |
|-------|-------------|--------|---------|
| 1A | Data Foundation | ✅ Complete | 01 Data Loaders |
| 1B | Variant Processing | ✅ Complete | 02 Variant Processing |
| 2A | Knowledge Graph | ✅ Complete | 03 Knowledge Graph |
| 2B | Graph Embeddings | ✅ Complete | 04 Graph Embeddings (TransE, RotatE) |
| 2C | Pretrained Embeddings | 🔲 Not Started | 05 Geneformer, ESM-2 |
| 3A | Ontology GNN | 🔲 Not Started | 06 GNN Models |
| 3B | Pathway Scoring | 🔲 Not Started | 07 Pathway Scoring |
| 3C | Subtype Clustering | 🔲 Not Started | 08 Clustering |
| 4 | Reasoning Layer | 🔲 Not Started | 09-11 Rules, Neurosymbolic, Hypotheses |
| 5 | Causal Inference | 🔲 Not Started | 12 Causal Analysis |

> See [Implementation Plan](implementation_plan.md) for detailed module status and development roadmap.

---

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

### 4. Biological Context Integration

Domain knowledge is embedded intrinsically through:
- **Developmental expression**: BrainSpan prenatal/postnatal expression patterns
- **Cell-type specificity**: Single-cell atlas integration (Allen Brain)
- **Gene constraint**: gnomAD pLI/LOEUF scores, SFARI gene annotations
- **Foundation model embeddings**: Geneformer, ESM-2, and literature-based representations

### 5. Subtype Discovery

Per-individual pathway profiles are used for:
- Unsupervised clustering (GMM, spectral, hierarchical)
- Identification of **genetically coherent subgroups**
- Probabilistic subtype membership assignment

### 6. Neuro-Symbolic Reasoning

Combines neural predictions with biological rules:
- **R1**: Constrained LoF in developing cortex → high-confidence disruption
- **R2**: Multiple pathway hits → convergence signal
- **R3**: CHD8 cascade → chromatin regulation subtype
- **R4**: Synaptic gene + excitatory neuron expression → synaptic subtype
- **R5**: Intact paralog → potential compensation
- **R6**: Drug-pathway targeting → therapeutic hypothesis

### 7. Causal Inference

Enables mechanistic reasoning:
- **Structural causal models**: Explicit causal chains from variant to phenotype
- **Intervention queries**: "What if we target this pathway?"
- **Counterfactual reasoning**: "Would phenotype differ if gene were intact?"
- **Mediation analysis**: Direct vs. indirect effects through pathways

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
│                  BIOLOGICAL CONTEXT LAYER                        │
│  BrainSpan expression • Single-cell atlas • Gene constraints     │
│  Foundation model embeddings (Geneformer, ESM-2)                 │
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
│                    REASONING LAYER                               │
│  Neuro-symbolic rules (R1-R6) • Causal inference • Explanations  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                                │
│  Subtype definitions • Pathway profiles • Hypothesis reports     │
│  Causal analysis • Reasoning chains • Evidence trails            │
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
- **Causal support**: Intervention queries validate that targeting a pathway would plausibly affect outcome

### Causal Validation of Hypotheses

The framework uses causal inference to strengthen therapeutic hypotheses:

1. **Intervention queries**: Estimate P(phenotype | do(pathway_targeted))
2. **Mediation analysis**: Quantify how much of a gene's effect flows through the target pathway
3. **Counterfactual reasoning**: "Would phenotype differ if this pathway were restored?"

Hypotheses with strong causal support are prioritized over those based on correlation alone.

All hypotheses require independent experimental and clinical validation.

---

---

## Architecture Summary

The framework implements a **layered hybrid architecture**:

| Layer | Components | Purpose |
|-------|------------|---------|
| **Data Foundation** | VCF loaders, expression loaders, constraint loaders | Standardized data ingestion |
| **Biological Context** | BrainSpan, single-cell atlas, gnomAD, SFARI | Developmental and cell-type specificity |
| **Knowledge Representation** | Knowledge graph, graph embeddings, foundation models | Structured biological relationships |
| **Neural Models** | Ontology-aware GNN, pathway scoring, clustering | Pattern learning and aggregation |
| **Symbolic Reasoning** | Rule engine (R1-R6), neuro-symbolic integration | Explainable biological inference |
| **Causal Inference** | SCM, do-calculus, counterfactuals, mediation | Mechanistic validation |

This design makes domain knowledge **intrinsic** rather than external, improving biological plausibility and cross-cohort generalization.

---

## Next Steps

- [Pathway Scoring Concept](pathway_scoring_concept.md) - Detailed scoring methodology
- [Stability and Replication](stability_replication.md) - Validation approaches
- [Limitations](limitations.md) - Known constraints and caveats
- [Domain Knowledge Integration](domain_knowledge_integration_analysis.md) - Technical deep-dive on knowledge embedding
- [Implementation Plan](implementation_plan.md) - Session-by-session development guide
