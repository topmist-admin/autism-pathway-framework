# Data Formats Specification

> This document specifies the input and output data formats used in the Autism Pathway Framework.

## Input Formats

### VCF (Variant Call Format)

Standard VCF format (v4.2+) with required fields:

```
##fileformat=VCFv4.2
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SAMPLE1	SAMPLE2
chr1	12345	rs123	A	G	50	PASS	DP=30;AF=0.001	GT:DP:GQ	0/1:15:30	0/0:20:40
```

**Required columns**: CHROM, POS, REF, ALT, QUAL, FILTER, FORMAT, sample genotypes
**Required FORMAT fields**: GT (genotype)
**Recommended FORMAT fields**: DP (depth), GQ (genotype quality)
**Compression**: `.vcf.gz` with tabix index recommended

### GMT (Gene Matrix Transposed)

Pathway definition format:

```
PATHWAY_ID<TAB>DESCRIPTION<TAB>GENE1<TAB>GENE2<TAB>GENE3<TAB>...
```

Example:
```
GO:0007268	synaptic_transmission	SHANK3	NRXN1	NLGN1	GRIN2B	SCN2A
REACT:12345	Neurotransmitter_release	SYT1	SNAP25	STX1A	VAMP2
```

**Gene identifiers**: HGNC symbols recommended (e.g., SHANK3, NRXN1)

### GO OBO Format

Gene Ontology structure file:

```
[Term]
id: GO:0007268
name: chemical synaptic transmission
namespace: biological_process
def: "The vesicular release of classical neurotransmitter..."
is_a: GO:0099537 ! trans-synaptic signaling
```

### GAF (Gene Association Format)

GO gene associations:

```
!gaf-version: 2.1
UniProtKB	Q9UPX8	SHANK3	GO:0007268	PMID:12345	IDA	P	SH3 and multiple ankyrin repeat domains protein 3	gene	taxon:9606	20200101	UniProt
```

**Required columns**: DB, DB_Object_ID, DB_Object_Symbol, GO_ID, Reference, Evidence

### gnomAD Constraint Format

Tab-separated constraint scores:

```
gene	transcript	pLI	LOEUF	mis_z
SHANK3	ENST00000262795	1.00	0.15	3.45
NRXN1	ENST00000353564	1.00	0.12	4.21
```

**Required columns**: gene, pLI (or LOEUF)

### SFARI Gene Format

CSV format with gene annotations:

```csv
gene-symbol,gene-name,gene-score,syndromic,chromosome
SHANK3,SH3 and multiple ankyrin repeat domains 3,1,1,22
CHD8,Chromodomain helicase DNA binding protein 8,1,0,14
```

**Scores**: 1 (high confidence), 2 (strong candidate), 3 (suggestive evidence)

### BrainSpan Expression Format

Matrix format with genes × samples:

```
gene_id	sample1_prenatal	sample2_prenatal	sample3_postnatal
ENSG00000251322	5.2	4.8	2.1
ENSG00000185097	3.1	3.5	6.2
```

With accompanying metadata:
```csv
sample_id,age,structure,sex
sample1_prenatal,12 pcw,DFC,M
sample2_prenatal,16 pcw,VFC,F
```

### h5ad (Single-Cell Format)

AnnData HDF5 format for single-cell data:

```python
# Structure:
# adata.X          - Expression matrix (cells × genes)
# adata.obs        - Cell metadata (DataFrame)
# adata.var        - Gene metadata (DataFrame)
# adata.uns        - Unstructured annotations

import anndata
adata = anndata.read_h5ad("atlas.h5ad")
```

**Required metadata**:
- `adata.obs['cell_type']` - Cell type annotations
- `adata.var.index` - Gene identifiers (symbols or Ensembl IDs)

---

## Output Formats

### Gene Burden Matrix

**Parquet format** (recommended):

```
Schema:
- sample_id: string (partition key)
- gene_id: string
- burden_score: float
- variant_count: int
- contributing_variants: list<string>
```

**JSON format** (for smaller datasets):

```json
{
  "SAMPLE_001": {
    "SHANK3": {
      "burden_score": 1.5,
      "variant_count": 2,
      "contributing_variants": ["chr22:51135990:G>A", "chr22:51140231:C>T"]
    },
    "NRXN1": {
      "burden_score": 1.0,
      "variant_count": 1,
      "contributing_variants": ["chr2:50145678:A>G"]
    }
  }
}
```

### Pathway Scores

**JSON Schema**:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "sample_id": {"type": "string"},
    "pathways": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "raw_score": {"type": "number"},
          "normalized_score": {"type": "number"},
          "z_score": {"type": ["number", "null"]},
          "pathway_size": {"type": "integer"},
          "genes_with_burden": {"type": "integer"},
          "contributing_genes": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "gene_id": {"type": "string"},
                "contribution": {"type": "number"}
              }
            }
          }
        }
      }
    }
  }
}
```

**Example**:

```json
{
  "sample_id": "SAMPLE_001",
  "pathways": {
    "GO:0007268": {
      "raw_score": 2.5,
      "normalized_score": 0.35,
      "z_score": 1.8,
      "pathway_size": 50,
      "genes_with_burden": 5,
      "contributing_genes": [
        {"gene_id": "SHANK3", "contribution": 1.0},
        {"gene_id": "NRXN1", "contribution": 0.8},
        {"gene_id": "GRIN2B", "contribution": 0.4}
      ]
    }
  }
}
```

### Cluster Assignments

**JSON Schema**:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "n_clusters": {"type": "integer"},
    "assignments": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "hard_label": {"type": "integer"},
          "probabilities": {
            "type": "array",
            "items": {"type": "number"}
          },
          "uncertainty": {"type": "number"}
        }
      }
    },
    "subtype_profiles": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "size": {"type": "integer"},
          "fraction": {"type": "number"},
          "characteristic_pathways": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "pathway_id": {"type": "string"},
                "importance": {"type": "number"}
              }
            }
          }
        }
      }
    }
  }
}
```

**Example**:

```json
{
  "n_clusters": 4,
  "assignments": {
    "SAMPLE_001": {
      "hard_label": 2,
      "probabilities": [0.1, 0.15, 0.7, 0.05],
      "uncertainty": 0.45
    }
  },
  "subtype_profiles": {
    "0": {
      "size": 150,
      "fraction": 0.30,
      "characteristic_pathways": [
        {"pathway_id": "GO:0007268", "importance": 1.2}
      ]
    }
  }
}
```

### Validation Report

```json
{
  "validation_id": "val_20240115_001",
  "timestamp": "2024-01-15T10:30:00Z",
  "input_file": "cohort_variants.vcf.gz",
  "summary": {
    "total_variants": 125000,
    "passed_qc": 98500,
    "failed_qc": 26500,
    "total_samples": 500,
    "passed_samples": 495
  },
  "qc_breakdown": {
    "low_quality": 15000,
    "low_depth": 5000,
    "high_allele_frequency": 6500
  },
  "warnings": [
    "5 samples have unusually high variant counts"
  ]
}
```

---

## Matrix Formats

### Dense Matrix (NumPy/Pandas)

For small-medium datasets:

```python
import numpy as np
import pandas as pd

# Gene burden matrix: samples × genes
burden_matrix = pd.DataFrame(
    data=np.random.rand(500, 20000),  # 500 samples, 20k genes
    index=sample_ids,
    columns=gene_ids
)

# Save formats
burden_matrix.to_parquet("burden_matrix.parquet")
burden_matrix.to_csv("burden_matrix.csv")
np.save("burden_matrix.npy", burden_matrix.values)
```

### Sparse Matrix (SciPy)

For large, sparse datasets (most gene burdens are 0):

```python
from scipy.sparse import csr_matrix, save_npz

# Create sparse matrix
sparse_burden = csr_matrix(burden_matrix.values)

# Save
save_npz("burden_matrix_sparse.npz", sparse_burden)

# Save row/column labels separately
np.save("samples.npy", sample_ids)
np.save("genes.npy", gene_ids)
```

### HDF5 Format

For large datasets with metadata:

```python
import h5py

with h5py.File("burdens.h5", "w") as f:
    # Data
    f.create_dataset("scores", data=burden_matrix.values)

    # Metadata
    f.create_dataset("samples", data=sample_ids.astype("S"))
    f.create_dataset("genes", data=gene_ids.astype("S"))

    # Attributes
    f.attrs["n_samples"] = len(sample_ids)
    f.attrs["n_genes"] = len(gene_ids)
    f.attrs["created"] = "2024-01-15"
```

---

## File Naming Conventions

```
# Input files
variants_{cohort}_hg38.vcf.gz
pathways_go_bp.gmt
pathways_reactome.gmt
gnomad_v2.1.1_constraints.tsv
brainspan_expression.parquet

# Output files
{cohort}_gene_burdens.parquet
{cohort}_pathway_scores.json
{cohort}_cluster_assignments.json
{cohort}_validation_report.json

# Intermediate files
{cohort}_qc_filtered.vcf.gz
{cohort}_annotated_variants.parquet
```

---

## Compression

| Format | Compression | Typical Ratio | Use Case |
|--------|-------------|---------------|----------|
| VCF | bgzip + tabix | 10-20x | Random access, standard |
| Parquet | snappy/gzip | 5-10x | Columnar analytics |
| JSON | gzip | 5-10x | Human-readable output |
| HDF5 | gzip/lzf | 3-5x | Large matrices |

---

## See Also

- [Configuration Guide](configuration.md) - File path configuration
- [API Reference](api_reference.md) - Loading functions
- [Terminology](terminology.md) - Field definitions
