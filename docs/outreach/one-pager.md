# Autism Pathway Framework

**A Research Tool for Pathway-Based Analysis of Genetic Heterogeneity in ASD**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18403844.svg)](https://doi.org/10.5281/zenodo.18403844)

---

## The Problem

Autism Spectrum Disorder is genetically complex—hundreds of genes, diverse mechanisms, and significant heterogeneity across individuals. Traditional gene-centric analyses often fail to generalize across cohorts.

## Our Approach

The **Autism Pathway Framework** shifts the unit of analysis from individual genes to **biological pathways**, enabling:

- Integration of rare variants into pathway-level disruption scores
- Network-based signal refinement using protein-protein interactions
- Unsupervised clustering to identify biologically coherent subgroups
- Built-in validation gates to prevent overfitting

## Key Features

| Feature | Description |
|---------|-------------|
| **Pathway Scoring** | Aggregate gene burdens across 15+ biological pathways |
| **Subtype Discovery** | GMM clustering with automatic cluster selection |
| **Validation Gates** | Negative controls + bootstrap stability testing |
| **Reproducibility** | Deterministic execution with pinned dependencies |
| **Colab-Ready** | Run the demo in your browser—no installation |

## What We're Looking For

We're seeking **collaborators with ASD cohort data** (N ≥ 100) to:

1. **Validate** the framework on independent cohorts
2. **Extend** pathway definitions with domain expertise
3. **Co-author** publications on subtype discovery

## Data Requirements

| Input | Format | Notes |
|-------|--------|-------|
| Variants | VCF | Annotated with gene symbols |
| Phenotypes | CSV | Sample IDs + clinical features |
| (Optional) Expression | CSV | For multi-omic integration |

**Your data stays on your infrastructure.** The framework runs locally or in your cloud environment.

## Getting Started

```bash
# 5-minute setup
git clone https://github.com/topmist-admin/autism-pathway-framework
cd autism-pathway-framework
pip install -r requirements.lock && pip install -e .
make demo  # Run on synthetic data
```

Or try the **[Colab notebook](https://colab.research.google.com/github/topmist-admin/autism-pathway-framework/blob/main/examples/notebooks/01_demo_end_to_end.ipynb)** (no installation required).

## Links

- **GitHub**: https://github.com/topmist-admin/autism-pathway-framework
- **DOI**: https://doi.org/10.5281/zenodo.18403844
- **Preprint**: https://doi.org/10.13140/RG.2.2.25221.41441
- **Documentation**: https://github.com/topmist-admin/autism-pathway-framework/tree/main/docs

## Contact

**Rohit Chauhan**
Email: info@topmist.com
GitHub: [@topmist-admin](https://github.com/topmist-admin)

---

> **RESEARCH USE ONLY** — This framework is for hypothesis generation. Not for clinical diagnosis or treatment decisions.
