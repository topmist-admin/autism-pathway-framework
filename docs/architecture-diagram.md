# Pipeline Architecture Diagram

> **RESEARCH USE ONLY** — This framework is for research purposes only.

## Overview

The Autism Pathway Framework implements a **layered hybrid architecture** that transforms genetic variants into biologically interpretable subtypes.

---

## Pipeline Flow Diagram

```mermaid
flowchart TB
    subgraph INPUT["INPUT LAYER"]
        VCF[("VCF File<br/>Variants")]
        PHENO[("Phenotypes<br/>(optional)")]
        PATHWAY_DB[("Pathway DB<br/>GMT file")]
    end

    subgraph PROCESSING["PROCESSING LAYER"]
        direction TB
        QC["QC Filtering<br/>(Module 02)"]
        ANNOT["Annotation<br/>(Module 02)"]
        BURDEN["Gene Burden<br/>Calculation"]
    end

    subgraph KNOWLEDGE["KNOWLEDGE LAYER"]
        direction TB
        KG["Knowledge Graph<br/>(Module 03)"]
        EMBED["Graph Embeddings<br/>(Module 04)"]
        PRETRAIN["Foundation Models<br/>(Module 05)"]
    end

    subgraph ANALYSIS["ANALYSIS LAYER"]
        direction TB
        PATHWAY["Pathway Scoring<br/>(Module 07)"]
        NETWORK["Network<br/>Propagation"]
        CLUSTER["GMM Clustering<br/>(Module 08)"]
    end

    subgraph VALIDATION["VALIDATION LAYER"]
        direction TB
        NEG1["Negative Control 1<br/>Label Shuffle"]
        NEG2["Negative Control 2<br/>Random Genes"]
        STAB["Stability Test<br/>Bootstrap"]
    end

    subgraph OUTPUT["OUTPUT LAYER"]
        SCORES[("Pathway<br/>Scores CSV")]
        ASSIGN[("Subtype<br/>Assignments")]
        REPORT[("Validation<br/>Report")]
        FIGURE[("Summary<br/>Figure")]
    end

    VCF --> QC
    PHENO -.-> CLUSTER
    PATHWAY_DB --> PATHWAY

    QC --> ANNOT --> BURDEN
    BURDEN --> PATHWAY

    KG -.-> PATHWAY
    EMBED -.-> PATHWAY
    PRETRAIN -.-> PATHWAY

    PATHWAY --> NETWORK --> CLUSTER

    CLUSTER --> NEG1
    CLUSTER --> NEG2
    CLUSTER --> STAB

    CLUSTER --> ASSIGN
    PATHWAY --> SCORES
    NEG1 & NEG2 & STAB --> REPORT
    CLUSTER --> FIGURE

    style INPUT fill:#e1f5fe
    style PROCESSING fill:#fff3e0
    style KNOWLEDGE fill:#f3e5f5
    style ANALYSIS fill:#e8f5e9
    style VALIDATION fill:#ffebee
    style OUTPUT fill:#fafafa
```

---

## Simplified ASCII Diagram

For environments without Mermaid rendering:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                                 │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐              │
│   │ VCF File │    │ Phenotypes   │    │ Pathway DB   │              │
│   │ Variants │    │  (optional)  │    │  (GMT file)  │              │
│   └────┬─────┘    └──────┬───────┘    └──────┬───────┘              │
└────────┼─────────────────┼───────────────────┼──────────────────────┘
         │                 │                   │
         ▼                 │                   │
┌─────────────────────────────────────────────────────────────────────┐
│                      PROCESSING LAYER                                │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐                  │
│   │    QC    │───▶│ Annotate │───▶│ Gene Burden  │                  │
│   │ Filtering│    │          │    │ Calculation  │                  │
│   └──────────┘    └──────────┘    └──────┬───────┘                  │
└──────────────────────────────────────────┼──────────────────────────┘
                                           │
         ┌─────────────────────────────────┼──────────────────┐
         │        KNOWLEDGE LAYER          │                  │
         │  ┌────────────┐ ┌───────────┐   │ ┌─────────────┐  │
         │  │ Knowledge  │ │   Graph   │   │ │ Foundation  │  │
         │  │   Graph    │ │ Embeddings│   │ │   Models    │  │
         │  └─────┬──────┘ └─────┬─────┘   │ └──────┬──────┘  │
         └────────┼──────────────┼─────────┼────────┼─────────┘
                  │              │         │        │
                  └──────────────┴────┬────┴────────┘
                                      │ (optional enrichment)
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ANALYSIS LAYER                                │
│   ┌──────────────┐    ┌────────────────┐    ┌────────────────┐      │
│   │   Pathway    │───▶│    Network     │───▶│  GMM Clustering│      │
│   │   Scoring    │    │  Propagation   │    │   (Subtypes)   │      │
│   └──────────────┘    └────────────────┘    └───────┬────────┘      │
└─────────────────────────────────────────────────────┼───────────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       VALIDATION LAYER                               │
│   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │
│   │ Neg Control 1  │  │ Neg Control 2  │  │ Stability Test │        │
│   │ Label Shuffle  │  │ Random Genes   │  │   Bootstrap    │        │
│   └───────┬────────┘  └───────┬────────┘  └───────┬────────┘        │
└───────────┼───────────────────┼───────────────────┼─────────────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                 │
│   ┌──────────────┐  ┌──────────────┐  ┌────────┐  ┌────────────┐    │
│   │   Pathway    │  │   Subtype    │  │ Report │  │  Summary   │    │
│   │  Scores CSV  │  │ Assignments  │  │ JSON/MD│  │   Figure   │    │
│   └──────────────┘  └──────────────┘  └────────┘  └────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Mapping

| Layer | Modules | Description |
|-------|---------|-------------|
| **Input** | - | VCF, phenotypes, pathway database |
| **Processing** | 01, 02 | Data loading, QC, annotation, gene burden |
| **Knowledge** | 03, 04, 05 | Knowledge graph, embeddings, foundation models |
| **Analysis** | 06, 07, 08 | GNN, pathway scoring, clustering |
| **Validation** | - | Negative controls, stability testing |
| **Output** | - | CSV tables, reports, figures |

---

## Data Flow Summary

```
Variants (VCF)
    │
    ▼
Gene-level burden scores (per sample × gene)
    │
    ▼
Pathway disruption scores (per sample × pathway)
    │
    ▼
Clustered subtypes (per sample → cluster assignment)
    │
    ▼
Validated outputs (with reproducibility checks)
```

---

## Key Design Principles

1. **Layered Architecture**: Each layer has well-defined inputs/outputs
2. **Modularity**: Components can be used independently or combined
3. **Validation-First**: Built-in negative controls and stability testing
4. **Reproducibility**: Deterministic with seed control and hash verification
5. **Research-Only**: Clear boundaries on appropriate use

---

*See [framework_overview.md](framework_overview.md) for conceptual details.*
