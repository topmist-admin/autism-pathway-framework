# Autism Pathway Framework

A comprehensive research framework for pathway- and network-based analysis of genetic heterogeneity in Autism Spectrum Disorder (ASD).

> **Implementation Status**: All 12 core modules complete | Last Updated: January 2026

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
| 09 | Symbolic Rules | Biological rule engine (R1-R6) | ✅ Complete |
| 10 | Neurosymbolic | GNN + symbolic rule integration | ✅ Complete |
| 11 | Therapeutic Hypotheses | Drug-pathway mapping and ranking | ✅ Complete |
| 12 | Causal Inference | SCM, do-calculus, counterfactuals | ✅ Complete |

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
├── docs/                     # Documentation
│   ├── implementation_plan.md
│   ├── framework_overview.md
│   ├── api_reference.md
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
├── tests/                    # Test suites
│   └── fixtures/             # Test data
│
└── examples/                 # Worked examples
    └── synthetic_example.md
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

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest modules/ -v
```

---

## Quick Start

```python
# Example: Load variants and compute gene burdens
from modules.01_data_loaders import VCFLoader
from modules.02_variant_processing import QCFilter, GeneBurdenCalculator

# Load variants
loader = VCFLoader()
variants = loader.load("variants.vcf.gz")

# Apply QC
qc = QCFilter()
filtered = qc.filter_variants(variants)

# Compute gene burdens
calculator = GeneBurdenCalculator()
burdens = calculator.compute(filtered)

# Example: Build knowledge graph and compute embeddings
from modules.03_knowledge_graph import KnowledgeGraph
from modules.04_graph_embeddings import TransEModel

kg = KnowledgeGraph()
kg.load_pathways("reactome.gmt")
kg.load_interactions("string.tsv")

embeddings = TransEModel(kg).train()

# Example: Run causal analysis
from modules.12_causal_inference import (
    StructuralCausalModel,
    DoCalculusEngine,
    CausalEffectEstimator
)

scm = StructuralCausalModel()
# ... build causal graph ...
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

- [Framework Overview](docs/framework_overview.md) - Conceptual architecture
- [Implementation Plan](docs/implementation_plan.md) - Module development guide
- [API Reference](docs/api_reference.md) - Detailed API documentation
- [Pathway Scoring Concept](docs/pathway_scoring_concept.md) - Scoring methodology
- [Stability and Replication](docs/stability_replication.md) - Validation approaches
- [Limitations](docs/limitations.md) - Known constraints and caveats

---

## Testing

```bash
# Run all tests
python -m pytest modules/ -v

# Run specific module tests
python -m pytest modules/12_causal_inference/tests/ -v

# Run with coverage
python -m pytest modules/ --cov=modules --cov-report=html
```

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
