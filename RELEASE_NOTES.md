# Autism Pathway Framework v0.1.0

> **RESEARCH USE ONLY** — Not for clinical decision-making

The first reproducibility release of the Autism Pathway Framework.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18403844.svg)](https://doi.org/10.5281/zenodo.18403844)

---

## Quickstart

```bash
# Clone and setup
git clone https://github.com/topmist-admin/autism-pathway-framework.git
cd autism-pathway-framework
python3 -m venv autismenv && source autismenv/bin/activate
pip install -r requirements.lock && pip install -e .

# Run demo
make demo
```

**Or try in Google Colab** (no installation):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/topmist-admin/autism-pathway-framework/blob/main/examples/notebooks/01_demo_end_to_end.ipynb)

---

## What's Included

### Core Framework (12 Modules)

| Module | Description |
|--------|-------------|
| 01 Data Loaders | VCF, pathway, expression, constraint data |
| 02 Variant Processing | QC filtering, annotation, gene burden |
| 03 Knowledge Graph | Biological relationship encoding |
| 04 Graph Embeddings | TransE, RotatE embeddings |
| 05 Pretrained Embeddings | Geneformer, ESM-2, PubMedBERT |
| 06 Ontology GNN | Ontology-aware graph neural network |
| 07 Pathway Scoring | Multi-evidence pathway disruption |
| 08 Subtype Clustering | GMM clustering with validation |
| 09 Symbolic Rules | Biological rule engine (R1-R7) |
| 10 Neurosymbolic | GNN + symbolic integration |
| 11 Therapeutic Hypotheses | Drug-pathway mapping |
| 12 Causal Inference | SCM, do-calculus, counterfactuals |

### Integration Pipelines

- **Subtype Discovery**: VCF → pathway scores → molecular subtypes
- **Therapeutic Hypothesis**: subtypes → rules → drug candidates
- **Causal Analysis**: individual mechanistic reasoning

---

## Demo Runtime

| Environment | Runtime | Notes |
|-------------|---------|-------|
| Local (M1 Mac) | ~30 seconds | Synthetic 50-sample dataset |
| Google Colab | ~2 minutes | Including setup |
| Docker | ~45 seconds | Containerized execution |

---

## Outputs

The demo pipeline generates:

```
outputs/demo_run/
├── pathway_scores.csv      # Pathway disruption scores per sample
├── subtype_assignments.csv # Cluster assignments with confidence
├── report.json             # Machine-readable summary
├── report.md               # Human-readable report
├── run_metadata.yaml       # Reproducibility metadata
├── pipeline.log            # Execution log
└── figures/
    └── summary.png         # Visualization (optional)
```

See [docs/outputs_dictionary.md](docs/outputs_dictionary.md) for interpretation guide.

---

## Validation Gates

Every run includes automatic validation:

| Gate | Description | Threshold |
|------|-------------|-----------|
| Negative Control 1 | Label shuffle test | ARI < 0.15 |
| Negative Control 2 | Random gene sets | ARI < 0.15 |
| Stability Test | Bootstrap resampling | ARI ≥ 0.8 |

Results appear in `report.json` and `report.md`:

```json
{
  "validation_gates": {
    "all_passed": true,
    "tests": [...]
  }
}
```

---

## Reproducibility

This release ensures deterministic execution:

- **Pinned dependencies**: `requirements.lock`
- **Seed control**: `PYTHONHASHSEED=42`
- **Golden outputs**: `tests/golden/expected_outputs.yaml`
- **CI verification**: GitHub Actions runs on every push

To verify reproducibility:
```bash
make reproducibility-test
```

---

## Requirements

- Python 3.10+ (3.11 recommended)
- 16 GB RAM recommended
- 5 GB disk space

---

## Citation

```bibtex
@software{chauhan_2026_autism_pathway,
  author       = {Chauhan, Rohit},
  title        = {Autism Pathway Framework},
  version      = {0.1.0},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18403844},
  url          = {https://doi.org/10.5281/zenodo.18403844}
}
```

---

## Known Limitations

- Demo dataset is synthetic (N=50)
- GMM clustering assumes Gaussian mixture structure
- Network propagation depends on PPI network quality
- Not validated for clinical use

See [CHANGELOG.md](CHANGELOG.md) for full details.

---

## Documentation

- [Quickstart Guide](docs/quickstart.md)
- [Architecture Diagram](docs/architecture-diagram.md)
- [Outputs Dictionary](docs/outputs_dictionary.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Start Here for Researchers](docs/start-here-researchers.md)

---

## Links

- **GitHub**: https://github.com/topmist-admin/autism-pathway-framework
- **DOI**: https://doi.org/10.5281/zenodo.18403844
- **Preprint**: https://doi.org/10.13140/RG.2.2.25221.41441

---

**RESEARCH USE ONLY** — This framework is for hypothesis generation. Not for clinical diagnosis or treatment decisions.
