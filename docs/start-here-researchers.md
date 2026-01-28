# Start Here for Researchers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/topmist-admin/autism-pathway-framework/blob/main/examples/notebooks/01_demo_end_to_end.ipynb)

> **RESEARCH USE ONLY** — This framework is for hypothesis generation only. Not for clinical decision-making. All findings require independent experimental and clinical validation.

Welcome to the Autism Pathway Framework. This guide will help you understand what the framework does, run the demo, and evaluate whether it's useful for your research.

**Fastest way to try it:** Click the "Open in Colab" button above to run the demo interactively in your browser - no installation required.

---

## What This Framework Does

The Autism Pathway Framework shifts genetic analysis from **individual genes** to **biological pathways**:

```
Traditional approach:  Variant → Gene → "Is this gene associated?"
This framework:        Variant → Gene → Pathway → Subtype → "What biology is disrupted?"
```

### Why Pathways Matter

- **Genetic heterogeneity**: Hundreds of genes implicated in ASD, few replicate across cohorts
- **Biological convergence**: Different variants often disrupt shared pathways
- **Better signal**: Pathway-level aggregation improves statistical power
- **Interpretability**: Pathways map to biological processes and drug targets

### What You Get

| Output | Description |
|--------|-------------|
| **Pathway scores** | Per-sample disruption scores across biological pathways |
| **Subtype assignments** | Unsupervised clustering into genetically coherent groups |
| **Validation report** | Negative controls and stability metrics |

---

## The Golden Path: 30-Minute Demo

### 1. Setup (5-10 minutes)

```bash
# Clone repository
git clone https://github.com/topmist-admin/autism-pathway-framework.git
cd autism-pathway-framework

# Create environment
python3 -m venv autismenv
source autismenv/bin/activate

# Install
pip install -r requirements.lock
pip install -e .

# Verify
make verify
```

### 2. Run Demo (20-40 minutes)

```bash
make demo
```

This runs the full pipeline on a **synthetic 50-sample dataset** with planted ground truth subtypes.

### 3. Review Results

Open `outputs/demo_run/report.md` for a human-readable summary.

Key outputs:
- `pathway_scores.csv` - Sample × pathway matrix
- `subtype_assignments.csv` - Cluster assignments
- `figures/summary.png` - Visual overview

### 4. Verify Reproducibility

```bash
make verify-reproducibility
```

If this passes, the framework is working correctly on your machine.

---

## Understanding the Demo Results

### Expected Behavior

The demo dataset contains **planted subtypes** based on:
- Synaptic transmission genes
- Chromatin remodeling genes
- Ion channel genes

The framework should recover these subtypes with reasonable accuracy.

### Validation Gates

| Gate | What It Tests | Demo Expectation |
|------|---------------|------------------|
| **Label Shuffle** | Clustering shouldn't work on random labels | PASS (ARI ~0) |
| **Random Genes** | Clustering shouldn't work with random pathways | PASS/WARN |
| **Bootstrap Stability** | Clusters should be robust to resampling | PASS/WARN |

**Note:** On small datasets, validation gates may show WARN. This is expected.

---

## Is This Framework Right for Your Research?

### Good Fit

- You have **WES/WGS variant data** from autism cohorts
- You want to explore **pathway-level heterogeneity**
- You're interested in **hypothesis generation**, not clinical prediction
- You value **reproducibility** and **validation**

### Not a Good Fit

- You need **clinical diagnostic tools** (this is research-only)
- You have **small sample sizes** (<50 samples may be unstable)
- You want **individual-level predictions** (framework is cohort-focused)
- You need **real-time analysis** (batch processing only)

---

## Next Steps

### Run on Your Data

1. **Prepare your data:**
   - VCF file (multi-sample or merged)
   - Phenotype file (optional, CSV with sample_id column)
   - Pathway database (GMT format, or use provided defaults)

2. **Create config file:**
   ```yaml
   # your_config.yaml
   pipeline:
     name: "my_cohort"
     output_dir: "outputs/my_cohort"
     seed: 42

   data:
     vcf_path: "path/to/your/variants.vcf.gz"
     phenotype_path: "path/to/phenotypes.csv"  # optional
     pathway_db: "examples/demo_data/demo_pathways.gmt"
   ```

3. **Run:**
   ```bash
   python -m autism_pathway_framework --config your_config.yaml
   ```

See [data_formats.md](data_formats.md) for input specifications.

### Collaborate

If you're interested in:
- Testing on your cohort
- Contributing improvements
- Discussing methodology

Open an issue or start a discussion on GitHub.

---

## Key Resources

| Document | Purpose |
|----------|---------|
| [quickstart.md](quickstart.md) | Step-by-step setup guide |
| [architecture-diagram.md](architecture-diagram.md) | Pipeline flow visualization |
| [outputs_dictionary.md](outputs_dictionary.md) | How to interpret outputs |
| [troubleshooting.md](troubleshooting.md) | Common issues and solutions |
| [framework_overview.md](framework_overview.md) | Conceptual architecture |
| [limitations.md](limitations.md) | Known constraints |

---

## Important Disclaimers

### Research Use Only

This framework is designed for:
- Hypothesis generation
- Study design
- Educational purposes
- Research exploration

This framework is **NOT** designed for:
- Clinical diagnosis
- Treatment recommendations
- Individual predictions
- Medical decision-making

### Validation Requirements

All findings from this framework require:
- Independent replication in external cohorts
- Experimental validation of biological hypotheses
- Clinical validation before any translational application

### Uncertainty

Outputs include confidence scores and validation metrics. Always consider:
- Sample size limitations
- Cohort-specific biases
- Pathway database completeness
- Algorithm assumptions

---

## Contact

- **GitHub Issues:** Bug reports and feature requests
- **GitHub Discussions:** Questions and methodology discussions

---

*Version 0.1.0 | January 2026*
