# Quickstart Guide

> **RESEARCH USE ONLY** — This framework is for research purposes only. Not for clinical decision-making. See [DISCLAIMER.md](../DISCLAIMER.md).

Get the Autism Pathway Framework running in under 30 minutes.

---

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- **Git**
- **16 GB RAM** recommended
- **5 GB disk space**

---

## Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/topmist-admin/autism-pathway-framework.git
cd autism-pathway-framework

# Create virtual environment
python3 -m venv autismenv
source autismenv/bin/activate  # Linux/macOS
# or: autismenv\Scripts\activate  # Windows

# Install dependencies (pinned versions for reproducibility)
pip install -r requirements.lock
pip install -e .
```

Or use the convenience script:

```bash
make setup
```

---

## Step 2: Verify Installation

```bash
make verify
```

Expected output:
```
1. Python version:
   Python 3.11.x

2. Key packages:
   numpy: 1.26.x
   pandas: 2.2.x
   torch: 2.x.x
   networkx: 3.x

Verification complete!
```

---

## Step 3: Run the Demo Pipeline

The demo uses a synthetic 50-sample dataset with planted ground truth.

```bash
make demo
```

Or directly:
```bash
python -m autism_pathway_framework --config configs/demo.yaml
```

**Expected runtime:** 20-40 minutes on a standard laptop.

---

## Step 4: Check Outputs

After the pipeline completes, outputs are in `outputs/demo_run/`:

```
outputs/demo_run/
├── pathway_scores.csv      # Sample × Pathway disruption scores
├── subtype_assignments.csv # Cluster assignments with confidence
├── report.json             # Machine-readable results
├── report.md               # Human-readable report
├── run_metadata.yaml       # Reproducibility metadata
├── figures/
│   └── summary.png         # 3-panel visualization
└── pipeline.log            # Execution log
```

### Key Files to Review

1. **`report.md`** - Start here for a summary of results
2. **`subtype_assignments.csv`** - Sample cluster assignments
3. **`figures/summary.png`** - Visual overview

---

## Step 5: Verify Reproducibility

Confirm outputs match the expected golden reference:

```bash
make verify-reproducibility
```

Expected output:
```
Reproducibility verification: PASSED
Checks passed: 5/5
```

---

## Understanding the Results

### Pipeline Flow

```
VCF Variants → Gene Burdens → Pathway Scores → Clustering → Validation
```

### Validation Gates

The pipeline runs three validation checks:

| Gate | Purpose | Expected |
|------|---------|----------|
| **Negative Control 1** | Label shuffle should yield ARI ~0 | PASS |
| **Negative Control 2** | Random gene sets should not cluster | PASS/WARN |
| **Stability Test** | Bootstrap ARI ≥ 0.8 | PASS/WARN |

**Note:** On the small demo dataset, some validation gates may show WARN or FAIL. This is expected and does not indicate a problem with the framework.

### Output Interpretation

See [outputs_dictionary.md](outputs_dictionary.md) for detailed guidance on:
- What each output file contains
- How to interpret pathway scores
- What NOT to infer from results

---

## Next Steps

### Run on Your Own Data

1. Prepare your data:
   - VCF file with variants
   - Phenotype CSV (optional)
   - Pathway GMT file (or use provided defaults)

2. Create a config file based on `configs/demo.yaml`

3. Run:
   ```bash
   python -m autism_pathway_framework --config your_config.yaml
   ```

See [data_formats.md](data_formats.md) for input specifications.

### Explore the Codebase

| Resource | Description |
|----------|-------------|
| [framework_overview.md](framework_overview.md) | Conceptual architecture |
| [api_reference.md](api_reference.md) | Module API documentation |
| [troubleshooting.md](troubleshooting.md) | Common issues + solutions |

### Use Individual Modules

```python
# Example: Load variants and compute gene burdens
from modules.01_data_loaders import VCFLoader
from modules.02_variant_processing import QCFilter, GeneBurdenCalculator

loader = VCFLoader()
variants = loader.load("your_variants.vcf.gz")

qc = QCFilter()
filtered = qc.filter_variants(variants)

calculator = GeneBurdenCalculator()
burdens = calculator.compute(filtered)
```

---

## Common Issues

### Pipeline fails to start

```bash
# Ensure you're in the virtual environment
source autismenv/bin/activate

# Verify installation
make verify
```

### Memory errors

The demo dataset is small (50 samples). For larger cohorts:
- Use a machine with 16+ GB RAM
- Consider batch processing

### Different results across runs

Ensure reproducibility settings:
```bash
export PYTHONHASHSEED=42
make demo
```

See [troubleshooting.md](troubleshooting.md) for more solutions.

---

## Getting Help

- **Issues:** [GitHub Issues](https://github.com/topmist-admin/autism-pathway-framework/issues)
- **Discussions:** [GitHub Discussions](https://github.com/topmist-admin/autism-pathway-framework/discussions)

---

## Quick Command Reference

| Command | Description |
|---------|-------------|
| `make setup` | Install dependencies |
| `make verify` | Check environment |
| `make demo` | Run demo pipeline |
| `make verify-reproducibility` | Validate outputs |
| `make test` | Run unit tests |
| `make clean` | Remove build artifacts |

---

*Last updated: January 2026*
