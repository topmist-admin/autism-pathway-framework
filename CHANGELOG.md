# Changelog

All notable changes to the Autism Pathway Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note:** This framework is for RESEARCH USE ONLY. Not for clinical decision-making.

## [0.1.0] - 2026-01-28

### Added

#### Core Framework
- **12 core modules** implementing the complete pathway analysis pipeline:
  - Module 01: Data Loaders (VCF, pathway, expression, constraint data)
  - Module 02: Variant Processing (QC filtering, annotation, gene burden)
  - Module 03: Knowledge Graph (biological relationship encoding)
  - Module 04: Graph Embeddings (TransE, RotatE)
  - Module 05: Pretrained Embeddings (Geneformer, ESM-2, PubMedBERT integration)
  - Module 06: Ontology GNN (ontology-aware graph neural network)
  - Module 07: Pathway Scoring (multi-evidence pathway disruption scoring)
  - Module 08: Subtype Clustering (GMM-based clustering with validation)
  - Module 09: Symbolic Rules (biological rule engine R1-R7)
  - Module 10: Neurosymbolic (GNN + symbolic rule integration)
  - Module 11: Therapeutic Hypotheses (drug-pathway mapping and ranking)
  - Module 12: Causal Inference (SCM, do-calculus, counterfactuals)

- **3 integration pipelines**:
  - Subtype Discovery Pipeline: VCF → pathway scores → subtypes
  - Therapeutic Hypothesis Pipeline: subtypes + rules + drug hypotheses
  - Causal Analysis Pipeline: individual case mechanistic analysis

#### Reproducibility & Validation
- **Validation gates** with automatic pass/fail reporting:
  - Negative Control 1: Label shuffle test (expect ARI ~0)
  - Negative Control 2: Random gene sets test
  - Stability Test: Bootstrap resampling (ARI ≥ 0.8 threshold)
- **Golden outputs** for cross-machine reproducibility verification
- **Deterministic execution** with seed control (`PYTHONHASHSEED=42`)
- **CI/CD pipeline** (GitHub Actions) with automated reproducibility checks

#### Demo & Documentation
- **Synthetic demo dataset** (50 samples, 20 variants, 15 pathways)
- **Colab-ready notebook** (`examples/notebooks/01_demo_end_to_end.ipynb`)
- **One-command demo**: `make demo` or `python -m autism_pathway_framework --config configs/demo.yaml`
- **Comprehensive documentation**:
  - Quickstart guide (`docs/quickstart.md`)
  - Architecture diagram (`docs/architecture-diagram.md`)
  - Outputs dictionary (`docs/outputs_dictionary.md`)
  - Troubleshooting guide (`docs/troubleshooting.md`)
  - Start Here for Researchers (`docs/start-here-researchers.md`)

#### Infrastructure
- **Docker support** for containerized execution
- **Makefile** with convenient commands (`make demo`, `make test`, `make verify`)
- **Locked dependencies** (`requirements.lock`) for exact reproducibility

### Research Guardrails
- "Research Use Only" disclaimers in all outputs and documentation
- Clear separation of hypothesis generation vs. clinical claims
- Validation gates prevent over-interpretation of results

### Known Limitations

#### Data Limitations
- Demo dataset is synthetic and small (N=50)
- Real cohorts may require parameter tuning
- Pathway databases have incomplete coverage

#### Algorithmic Limitations
- GMM clustering assumes Gaussian mixture structure
- Network propagation depends on PPI network quality
- Causal inference requires strong assumptions

#### Scope Limitations
- Not validated for clinical use
- Individual-level predictions not recommended
- Requires computational biology expertise to interpret

### Dependencies
- Python 3.10+ (3.11 recommended)
- NumPy, Pandas, SciPy, scikit-learn
- NetworkX, PyTorch (optional for GNN)
- See `requirements.lock` for exact versions

## [Unreleased]

### Planned for v0.2
- GPU-accelerated GNN training
- Therapeutic hypothesis ranking with clinical validation
- Multi-cohort federation support
- Real cohort validation studies

---

[0.1.0]: https://github.com/topmist-admin/autism-pathway-framework/releases/tag/v0.1.0
[Unreleased]: https://github.com/topmist-admin/autism-pathway-framework/compare/v0.1.0...HEAD
