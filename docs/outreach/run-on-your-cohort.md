# How to Run on Your Cohort

This guide explains how to run the Autism Pathway Framework on your own cohort data.

---

## Prerequisites

- Python 3.10+ (3.11 recommended)
- 16 GB RAM
- VCF file with annotated variants
- Phenotype CSV file

---

## Step 1: Prepare Your Data

### 1.1 VCF File Requirements

Your VCF must include:

```
##fileformat=VCFv4.2
##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol">
##INFO=<ID=CONSEQUENCE,Number=1,Type=String,Description="Variant consequence">
##INFO=<ID=CADD,Number=1,Type=Float,Description="CADD score">
#CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO    FORMAT  SAMPLE1 SAMPLE2 ...
```

**Required INFO fields:**
| Field | Description | Example |
|-------|-------------|---------|
| GENE | Gene symbol | `SHANK3` |
| CONSEQUENCE | Variant effect | `missense_variant` |
| CADD | Deleteriousness score | `28.5` |

**If your VCF lacks annotations**, use VEP or ANNOVAR:

```bash
# Using VEP
vep -i your_cohort.vcf -o annotated.vcf --symbol --canonical --pick

# Using ANNOVAR
table_annovar.pl your_cohort.vcf humandb/ -buildver hg38 -out annotated
```

### 1.2 Phenotype CSV Requirements

```csv
sample_id,diagnosis,age,sex
SAMPLE1,ASD,8,M
SAMPLE2,ASD,12,F
SAMPLE3,Control,10,M
```

**Required columns:**
- `sample_id` — Must match VCF sample names exactly

**Optional columns:**
- `diagnosis`, `age`, `sex`, `iq`, `severity`, etc.
- Any clinical features you want to correlate with subtypes

### 1.3 Pathway File (Optional)

Use the default pathways or provide your own GMT file:

```
SYNAPTIC_TRANSMISSION    http://example.org    SHANK3    NRXN1    NLGN1    ...
CHROMATIN_REMODELING     http://example.org    CHD8      ARID1B   ...
```

---

## Step 2: Create Configuration

Copy and modify the demo config:

```bash
cp configs/demo.yaml configs/my_cohort.yaml
```

Edit `configs/my_cohort.yaml`:

```yaml
pipeline:
  name: "my_cohort_analysis"
  output_dir: "outputs/my_cohort"
  seed: 42
  verbose: true

data:
  vcf_path: "path/to/your/cohort.vcf"
  phenotype_path: "path/to/your/phenotypes.csv"
  pathway_db: "examples/demo_data/demo_pathways.gmt"  # or your own

clustering:
  n_clusters: null  # Auto-select via BIC
  n_clusters_range: [2, 8]

output:
  disclaimer: "Research use only. Not medical advice."
```

---

## Step 3: Run the Pipeline

```bash
# Activate environment
source autismenv/bin/activate

# Run with your config
python -m autism_pathway_framework --config configs/my_cohort.yaml
```

**Expected runtime:**
| Cohort Size | Approximate Time |
|-------------|------------------|
| N = 100 | 1-2 minutes |
| N = 500 | 5-10 minutes |
| N = 1000 | 15-30 minutes |

---

## Step 4: Review Outputs

Outputs are saved to `outputs/my_cohort/`:

```
outputs/my_cohort/
├── pathway_scores.csv       # Pathway disruption scores
├── subtype_assignments.csv  # Cluster assignments
├── report.json              # Machine-readable summary
├── report.md                # Human-readable report
├── run_metadata.yaml        # Reproducibility info
├── pipeline.log             # Execution log
└── figures/
    └── summary.png          # Visualization
```

### Key Files to Check

**1. Validation Gates (report.json)**

```json
{
  "validation_gates": {
    "all_passed": true,
    "tests": [
      {"name": "label_shuffle", "status": "PASS"},
      {"name": "random_genes", "status": "PASS"},
      {"name": "bootstrap_stability", "status": "PASS"}
    ]
  }
}
```

⚠️ **If validation fails:** Your cohort may be too small, or subtypes may not be well-separated. See [Troubleshooting](../troubleshooting.md).

**2. Cluster Assignments (subtype_assignments.csv)**

```csv
sample_id,cluster_id,cluster_label,confidence
SAMPLE1,0,synaptic,0.92
SAMPLE2,1,chromatin,0.87
...
```

**3. Pathway Scores (pathway_scores.csv)**

Z-normalized pathway disruption scores for each sample.

---

## Step 5: Interpret Results

### What the Subtypes Mean

| Subtype | Interpretation |
|---------|----------------|
| `synaptic` | Enriched for synaptic transmission pathway disruption |
| `chromatin` | Enriched for chromatin remodeling pathway disruption |
| `ion_channel` | Enriched for ion channel pathway disruption |

### Confidence Scores

- **> 0.8**: High confidence assignment
- **0.6 - 0.8**: Moderate confidence
- **< 0.6**: Low confidence (sample may be transitional)

### Correlating with Phenotypes

```python
import pandas as pd

# Load outputs
assignments = pd.read_csv("outputs/my_cohort/subtype_assignments.csv")
phenotypes = pd.read_csv("path/to/your/phenotypes.csv")

# Merge
merged = assignments.merge(phenotypes, on="sample_id")

# Analyze
print(merged.groupby("cluster_label")["iq"].mean())
print(merged.groupby("cluster_label")["severity"].value_counts())
```

---

## Troubleshooting

### "Not enough variants in pathways"

Your cohort may have different variant annotations. Check:
```bash
grep -o "GENE=[^;]*" your_cohort.vcf | sort | uniq -c | sort -rn | head
```

### "Validation gates failed"

- **Label shuffle failed**: May indicate spurious clusters
- **Bootstrap failed**: Clusters are unstable—try increasing N or reducing n_clusters

### "Memory error"

For large cohorts (N > 1000), increase available RAM or use chunked processing.

---

## Getting Help

- **Issues**: https://github.com/topmist-admin/autism-pathway-framework/issues
- **Documentation**: https://github.com/topmist-admin/autism-pathway-framework/tree/main/docs
- **Email**: info@topmist.com

---

> **RESEARCH USE ONLY** — Results are hypotheses requiring independent validation.
