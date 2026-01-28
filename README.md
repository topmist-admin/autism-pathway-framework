# Autism Pathway Framework

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/topmist-admin/autism-pathway-framework/blob/main/examples/notebooks/01_demo_end_to_end.ipynb)

> **⚠️ RESEARCH USE ONLY — NOT FOR CLINICAL DECISION-MAKING**
>
> This framework is for research and hypothesis generation purposes only. Outputs must NOT be used for clinical diagnosis, treatment decisions, or medical advice. All findings require independent experimental and clinical validation. See [DISCLAIMER.md](DISCLAIMER.md) and [docs/outputs_dictionary.md](docs/outputs_dictionary.md) for details.

A comprehensive research framework for pathway- and network-based analysis of genetic heterogeneity in Autism Spectrum Disorder (ASD).

> **Implementation Status**: FRAMEWORK COMPLETE - 12 modules + 3 integration pipelines | Last Updated: January 2026

---

## Overview

Autism Spectrum Disorder is genetically complex and biologically heterogeneous, involving hundreds of genes and diverse molecular mechanisms. Traditional gene-centric analyses often fail to generalize across cohorts, limiting biological interpretation and translational impact.

This framework shifts the unit of analysis from **individual genes** to **biological pathways and interaction networks**, enabling:

- Integration of genetic variation into pathway-level disruption scores
- Network-based signal refinement using gene-gene interactions
- Graph neural networks with ontology-aware architecture
- Neuro-symbolic reasoning combining ML with biological rules
- Causal inference for mechanistic hypothesis validation
- Unsupervised learning to identify biologically coherent subgroups

---

## Module Status

### Core Modules

| Module | Name | Description | Status |
|--------|------|-------------|--------|
| 01 | Data Loaders | VCF, pathway, expression, constraint data loading | ✅ Complete |
| 02 | Variant Processing | QC filtering, annotation, gene burden calculation | ✅ Complete |
| 03 | Knowledge Graph | Biological relationship graph construction | ✅ Complete |
| 04 | Graph Embeddings | TransE, RotatE knowledge graph embeddings | ✅ Complete |
| 05 | Pretrained Embeddings | Geneformer, ESM-2, PubMedBERT integration | ✅ Complete |
| 06 | Ontology GNN | Ontology-aware graph neural network | ✅ Complete |
| 07 | Pathway Scoring | Multi-evidence pathway disruption scoring | ✅ Complete |
| 08 | Subtype Clustering | GMM-based clustering with validation | ✅ Complete |
| 09 | Symbolic Rules | Biological rule engine (R1-R7) | ✅ Complete |
| 10 | Neurosymbolic | GNN + symbolic rule integration | ✅ Complete |
| 11 | Therapeutic Hypotheses | Drug-pathway mapping and ranking | ✅ Complete |
| 12 | Causal Inference | SCM, do-calculus, counterfactuals | ✅ Complete |

### Integration Pipelines

| Pipeline | Description | Status |
|----------|-------------|--------|
| Subtype Discovery | End-to-end VCF → pathway scores → subtypes | ✅ Complete |
| Therapeutic Hypothesis | Subtype discovery + rules + drug hypotheses + causal validation | ✅ Complete |
| Causal Analysis | Standalone causal reasoning for individual cases | ✅ Complete |

---

## Repository Structure

```
autism-pathway-framework/
│
├── README.md                 # This file
├── DISCLAIMER.md             # Research-only usage disclaimer
├── LICENSE                   # MIT License
│
├── modules/                  # Core implementation modules
│   ├── 01_data_loaders/      # VCF, pathway, expression loaders
│   ├── 02_variant_processing/# QC, annotation, gene burden
│   ├── 03_knowledge_graph/   # Biological knowledge graph
│   ├── 04_graph_embeddings/  # TransE, RotatE embeddings
│   ├── 05_pretrained_embeddings/ # Foundation model integration
│   ├── 06_ontology_gnn/      # Ontology-aware GNN
│   ├── 07_pathway_scoring/   # Pathway disruption scoring
│   ├── 08_subtype_clustering/# GMM clustering + validation
│   ├── 09_symbolic_rules/    # Biological rule engine
│   ├── 10_neurosymbolic/     # GNN + rules integration
│   ├── 11_therapeutic_hypotheses/ # Drug mapping + ranking
│   └── 12_causal_inference/  # Causal reasoning framework
│
├── pipelines/                # End-to-end integration pipelines
│   ├── subtype_discovery.py  # VCF → subtypes pipeline
│   ├── therapeutic_hypothesis.py # Full therapeutic pipeline
│   ├── causal_analysis.py    # Standalone causal reasoning
│   └── tests/                # Pipeline tests
│
├── autism_pathway_framework/ # CLI and utilities
│   ├── cli.py                # Command-line interface
│   ├── pipeline.py           # Demo pipeline orchestrator
│   ├── validation.py         # Validation gates
│   └── utils/                # Reproducibility utilities
│
├── configs/                  # Configuration files
│   ├── demo.yaml             # Demo pipeline config
│   └── default.yaml          # Default settings
│
├── examples/                 # Demo data and examples
│   ├── demo_data/            # Synthetic 50-sample dataset
│   └── notebooks/            # Jupyter notebooks (Colab-ready)
│
├── outputs/                  # Pipeline outputs (gitignored)
│
├── tests/                    # Test suites
│   ├── fixtures/             # Test data
│   └── golden/               # Golden outputs for reproducibility
│
├── docs/                     # Documentation
│   ├── v0.1_scope.md         # Release scope
│   ├── quickstart.md         # Getting started
│   ├── troubleshooting.md    # Common issues + solutions
│   ├── reproducibility.md    # Deterministic execution guide
│   └── outputs_dictionary.md # Output interpretation
│
├── .github/                  # GitHub configuration
│   └── workflows/ci.yml      # CI/CD pipeline
│
└── Makefile                  # Build and run commands
```

---

## Architecture

The framework implements a **layered hybrid architecture**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                                    │
│   WES/WGS variants • CNVs • Optional phenotypes • Family data           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       PROCESSING LAYER (Modules 01-02)                   │
│   QC • Annotation • Gene Burden Calculation • Batch Correction          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE LAYER (Modules 03-05)                       │
│   Knowledge Graph • Graph Embeddings • Foundation Model Embeddings      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ANALYSIS LAYER (Modules 06-08)                      │
│   Ontology GNN • Pathway Scoring • Subtype Clustering                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     REASONING LAYER (Modules 09-12)                      │
│   Symbolic Rules • Neurosymbolic • Therapeutic Hypotheses • Causal      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT LAYER                                    │
│   Subtype Definitions • Pathway Profiles • Hypothesis Reports           │
│   Causal Analysis • Reasoning Chains • Evidence Trails                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Pathway-First Analysis
Rather than focusing on individual "autism genes," the framework emphasizes **biological convergence**—the observation that diverse genetic variants often disrupt shared biological pathways.

### 2. Ontology-Aware GNN
Graph neural network architecture that respects the hierarchical structure of biological ontologies (GO, HPO), enabling structured biological inference.

### 3. Neuro-Symbolic Reasoning
Combines neural network predictions with explicit biological rules:
- **R1**: Constrained LoF in developing cortex → high-confidence disruption
- **R2**: Multiple pathway hits → convergence signal
- **R3**: CHD8 cascade → chromatin regulation subtype
- **R4**: Synaptic gene + excitatory neuron expression → synaptic subtype
- **R5**: Intact paralog → potential compensation
- **R6**: Drug-pathway targeting → therapeutic hypothesis

### 4. Causal Inference
Enables mechanistic validation of hypotheses:
- **Structural Causal Models**: Explicit causal chains from variant to phenotype
- **Do-Calculus**: Intervention queries ("What if we target this pathway?")
- **Counterfactual Reasoning**: "Would phenotype differ if gene were intact?"
- **Mediation Analysis**: Direct vs. indirect effects through pathways

### 5. Therapeutic Hypothesis Generation
Disrupted pathways are mapped to drug targets with evidence-based ranking:
- Biological plausibility scoring
- Safety flag assessment
- Causal support validation
- Diversity constraints for hypothesis portfolios

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/autism-pathway-framework.git
cd autism-pathway-framework

# Create virtual environment
python3 -m venv autismenv
source autismenv/bin/activate

# Install dependencies (using locked versions for reproducibility)
pip install -r requirements.lock
pip install -e .

# Or use make for convenience
make setup

# Verify environment
make verify

# Run tests
make test
```

### Requirements

- Python 3.10+ (3.11 recommended)
- 16 GB RAM recommended
- 5 GB free disk space

---

## Quick Start

### Try it in Google Colab (Fastest)

No installation required - run the demo directly in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/topmist-admin/autism-pathway-framework/blob/main/examples/notebooks/01_demo_end_to_end.ipynb)

The notebook walks through the complete pipeline step-by-step with visualizations.

### Using Integration Pipelines (Recommended)

```python
# Subtype Discovery Pipeline: VCF → Pathway Scores → Subtypes
from pipelines import SubtypeDiscoveryPipeline, PipelineConfig, DataConfig

config = PipelineConfig(
    data=DataConfig(
        vcf_path="cohort.vcf.gz",
        pathway_gmt_path="reactome.gmt",
    ),
)
pipeline = SubtypeDiscoveryPipeline(config)
result = pipeline.run()

print(result.summary)
print(f"Identified {result.n_subtypes} subtypes")

# Therapeutic Hypothesis Pipeline: Subtypes + Rules + Drug Hypotheses
from pipelines import TherapeuticHypothesisPipeline, TherapeuticPipelineConfig

config = TherapeuticPipelineConfig(
    data=DataConfig(vcf_path="cohort.vcf.gz", pathway_gmt_path="reactome.gmt"),
)
pipeline = TherapeuticHypothesisPipeline(config)
result = pipeline.run()

for hyp in result.ranking_result.top_hypotheses:
    print(hyp.summary())

# Causal Analysis Pipeline: Individual Case Analysis
from pipelines import CausalAnalysisPipeline, CausalAnalysisConfig

config = CausalAnalysisConfig(
    sample_id="PATIENT_001",
    variant_genes=["SHANK3", "CHD8"],
    disrupted_pathways=["synaptic_transmission"],
)
pipeline = CausalAnalysisPipeline(config)
result = pipeline.run()

# Query interventions
effect = pipeline.query_intervention("synaptic_transmission", "asd_phenotype")
```

### Using Individual Modules

```python
# Example: Load variants and compute gene burdens
from modules.01_data_loaders import VCFLoader
from modules.02_variant_processing import QCFilter, GeneBurdenCalculator

loader = VCFLoader()
variants = loader.load("variants.vcf.gz")

qc = QCFilter()
filtered = qc.filter_variants(variants)

calculator = GeneBurdenCalculator()
burdens = calculator.compute(filtered)

# Example: Run causal analysis directly
from modules.12_causal_inference import (
    StructuralCausalModel,
    DoCalculusEngine,
)

scm = StructuralCausalModel()
do_engine = DoCalculusEngine(scm)
ate = do_engine.average_treatment_effect("SHANK3_function", "asd_phenotype")
```

---

## Intended Use

This framework is designed for:

- **Hypothesis generation** in autism genetics research
- **Study design** for clinical and translational studies
- **Educational purposes** in computational biology
- **Research exploration** of pathway-based genetic analysis

---

## Not Intended For

This framework is **NOT** intended for:

- Clinical diagnosis or screening
- Treatment recommendations
- Individual-level predictions or decision-making

See [DISCLAIMER.md](DISCLAIMER.md) for full details.

---

## Documentation

### Getting Started

- **[Demo Notebook](examples/notebooks/01_demo_end_to_end.ipynb)** - Interactive walkthrough (Colab-ready)
- **[Quickstart Guide](docs/quickstart.md)** - Get running in 30 minutes
- **[Start Here for Researchers](docs/start-here-researchers.md)** - Overview and golden path
- [Architecture Diagram](docs/architecture-diagram.md) - Pipeline flow visualization

### Reference

- [Framework Overview](docs/framework_overview.md) - Conceptual architecture
- [API Reference](docs/api_reference.md) - Detailed API documentation
- [Outputs Dictionary](docs/outputs_dictionary.md) - How to interpret results
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

### Deep Dives

- [Pathway Scoring Concept](docs/pathway_scoring_concept.md) - Scoring methodology
- [Stability and Replication](docs/stability_replication.md) - Validation approaches
- [Limitations](docs/limitations.md) - Known constraints and caveats
- [Implementation Plan](docs/implementation_plan.md) - Module development guide

---

## Testing & Reproducibility

```bash
# Run all tests
make test

# Run specific module tests
python -m pytest modules/12_causal_inference/tests/ -v

# Run with coverage
make test-cov
```

### Demo Pipeline

Run the full demo pipeline on synthetic data:

```bash
# Run demo (produces outputs in outputs/demo_run/)
make demo

# Or directly
python -m autism_pathway_framework --config configs/demo.yaml
```

### Reproducibility Verification

Verify outputs match expected golden reference:

```bash
# Verify against golden outputs
make verify-reproducibility

# Full reproducibility test (runs pipeline twice, compares outputs)
make reproducibility-test
```

See [docs/troubleshooting.md](docs/troubleshooting.md) for common issues.

---

## Contributing

Contributions are welcome. Please ensure all contributions:

- Maintain the research-only focus
- Include appropriate uncertainty and limitations
- Follow scientific rigor and reproducibility standards
- Include comprehensive tests

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this framework in your research, please cite appropriately and acknowledge its research-only nature.

---

## Contact

For questions or collaboration inquiries, please open an issue in this repository.
