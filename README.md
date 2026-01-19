# Autism Pathway Framework

A research framework for pathway- and network-based analysis of genetic heterogeneity in Autism Spectrum Disorder (ASD).

---

## Overview

Autism Spectrum Disorder is genetically complex and biologically heterogeneous, involving hundreds of genes and diverse molecular mechanisms. Traditional gene-centric analyses often fail to generalize across cohorts, limiting biological interpretation and translational impact.

This framework shifts the unit of analysis from **individual genes** to **biological pathways and interaction networks**, enabling:

- Integration of genetic variation into pathway-level disruption scores
- Network-based signal refinement using gene-gene interactions
- Unsupervised and semi-supervised learning to identify biologically coherent subgroups
- Improved cross-cohort replication at the pathway level

---

## Repository Structure

```
autism-pathway-framework/
│
├── README.md                 # This file
├── DISCLAIMER.md             # Research-only usage disclaimer
├── LICENSE                   # MIT License
│
├── docs/                     # Conceptual documentation
│   ├── framework_overview.md
│   ├── pathway_scoring_concept.md
│   ├── stability_replication.md
│   └── limitations.md
│
├── pseudocode/               # Algorithm pseudocode
│   ├── variant_to_gene.md
│   ├── gene_to_pathway.md
│   ├── network_refinement.md
│   └── subtype_clustering.md
│
└── examples/                 # Worked examples
    └── synthetic_example.md
```

---

## Key Concepts

### 1. Pathway-First Analysis

Rather than focusing on individual "autism genes," this framework emphasizes **biological convergence**—the observation that diverse genetic variants often disrupt shared biological pathways.

### 2. Network Refinement

Gene-gene interaction networks (protein interactions, co-expression, regulatory relationships) are used to propagate and refine pathway signals, reducing dependence on exact gene membership.

### 3. Subtype Discovery

Per-individual pathway disruption profiles serve as input to clustering methods, identifying **genetically coherent subgroups** that may respond differently to interventions.

### 4. Stability and Replication

All results are validated through:
- Internal stability testing (bootstrap, resampling)
- Cross-cohort replication at the pathway/subtype level
- Negative controls (permuted labels, randomized gene sets)

---

## Intended Use

This framework is designed for:

- **Hypothesis generation** in autism genetics research
- **Study design** for clinical and translational studies
- **Educational purposes** in computational biology

---

## Not Intended For

This framework is **NOT** intended for:

- Clinical diagnosis or screening
- Treatment recommendations
- Individual-level predictions or decision-making

See [DISCLAIMER.md](DISCLAIMER.md) for full details.

---

## Getting Started

1. Review the [Framework Overview](docs/framework_overview.md)
2. Understand the [Pathway Scoring Concept](docs/pathway_scoring_concept.md)
3. Explore the [Pseudocode](pseudocode/) for implementation details
4. Walk through the [Synthetic Example](examples/synthetic_example.md)

---

## Contributing

Contributions are welcome. Please ensure all contributions:

- Maintain the research-only focus
- Include appropriate uncertainty and limitations
- Follow scientific rigor and reproducibility standards

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this framework in your research, please cite appropriately and acknowledge its research-only nature.

---

## Contact

For questions or collaboration inquiries, please open an issue in this repository.
