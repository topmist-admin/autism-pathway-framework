# Data Schema Documentation

> **⚠️ RESEARCH USE ONLY** — All data and outputs are for research purposes only. Not for clinical decision-making. See [DISCLAIMER.md](../DISCLAIMER.md).

This document describes the data flow through the Autism Pathway Framework: input formats, intermediate representations, and output artifacts.

## Overview

```
INPUT                    INTERMEDIATE               OUTPUT
─────                    ────────────               ──────
VCF files          →     Variant matrix       →     Subtype assignments
Phenotype CSV      →     Feature tensors      →     Pathway enrichments
Pathway GMT        →     Knowledge graph      →     Embeddings (h5)
Reference DBs      →     Annotated variants   →     Visualization files
```

## Input Data Formats

### 1. Variant Data (VCF)

**Format**: VCF v4.2+
**Location**: `data/raw/` or `examples/demo_data/`
**Required fields**:

| Field | Description | Example |
|-------|-------------|---------|
| CHROM | Chromosome | chr1 |
| POS | Position (1-based) | 155234822 |
| REF | Reference allele | G |
| ALT | Alternate allele | A |
| QUAL | Quality score | 99 |
| FILTER | Filter status | PASS |
| INFO | Annotations (see below) | GENE=CHD8;CADD=28.5 |
| FORMAT | Genotype format | GT:DP:GQ |
| SAMPLE_* | Per-sample genotypes | 0/1:45:99 |

**Required INFO annotations**:
- `GENE`: Gene symbol (HGNC)
- `CONSEQUENCE`: Variant effect (missense_variant, frameshift, etc.)
- `CADD`: CADD phred score (pathogenicity)

**Optional INFO annotations**:
- `AF`: Allele frequency (gnomAD)
- `CLNSIG`: ClinVar significance
- `SIFT`, `PolyPhen`: Functional predictions

### 2. Phenotype Data (CSV)

**Format**: CSV with header
**Location**: `data/raw/` or `examples/demo_data/`
**Required columns**:

| Column | Type | Description |
|--------|------|-------------|
| sample_id | string | Unique sample identifier (matches VCF) |
| diagnosis | string | Diagnosis category (ASD, Control, etc.) |

**Recommended columns**:

| Column | Type | Description |
|--------|------|-------------|
| age_at_diagnosis | float | Age in years |
| sex | string | M/F |
| iq_score | int | IQ assessment |
| language_delay | int | 0/1 binary |
| motor_delay | int | 0/1 binary |
| seizure_history | int | 0/1 binary |
| family_history_asd | int | 0/1 binary |
| cohort | string | Study cohort identifier |

### 3. Pathway Definitions (GMT)

**Format**: Gene Matrix Transposed (GMT)
**Location**: `data/raw/pathways/` or `examples/demo_data/`
**Structure**: Tab-separated, one pathway per line

```
PATHWAY_NAME<TAB>DESCRIPTION<TAB>GENE1<TAB>GENE2<TAB>...
```

**Example**:
```
SYNAPTIC_TRANSMISSION	Synaptic signaling	SHANK3	NRXN1	SYNGAP1	GRIN2B
```

### 4. Reference Databases

See [data_versions.md](data_versions.md) for version specifications.

| Database | Format | Purpose |
|----------|--------|---------|
| Reactome | BioPAX/GMT | Pathway definitions |
| Gene Ontology | OBO/GAF | Functional annotations |
| STRING | TSV | Protein interactions |
| gnomAD | VCF/TSV | Population frequencies |
| SFARI Gene | CSV | ASD gene scores |
| ClinVar | VCF | Clinical significance |

---

## Intermediate Representations

### 1. Variant Matrix

**Format**: NumPy array / Pandas DataFrame
**Shape**: `(n_samples, n_variants)`
**Values**: 0 (ref/ref), 1 (het), 2 (hom alt), -1 (missing)

```python
# Access via Module 01
from modules.module_01_data_loaders import vcf_loader
variant_matrix = vcf_loader.load_vcf_to_matrix("path/to/file.vcf")
```

### 2. Annotated Variant Table

**Format**: Pandas DataFrame
**Columns**:

| Column | Type | Source |
|--------|------|--------|
| variant_id | string | Generated |
| chrom | string | VCF |
| pos | int | VCF |
| ref | string | VCF |
| alt | string | VCF |
| gene | string | INFO field |
| consequence | string | INFO field |
| cadd_score | float | INFO field |
| gnomad_af | float | gnomAD lookup |
| sfari_score | int | SFARI lookup |
| is_lof | bool | Computed |
| is_missense | bool | Computed |

### 3. Knowledge Graph

**Format**: NetworkX DiGraph / PyTorch Geometric Data
**Node types**:
- `gene`: Gene entities
- `variant`: Variant entities
- `pathway`: Pathway entities
- `phenotype`: Phenotype entities

**Edge types**:
- `variant_in_gene`: Variant → Gene
- `gene_in_pathway`: Gene → Pathway
- `gene_interacts`: Gene ↔ Gene (STRING)
- `sample_has_variant`: Sample → Variant
- `sample_has_phenotype`: Sample → Phenotype

```python
# Access via Module 03
from modules.module_03_knowledge_graph import kg_builder
graph = kg_builder.build_kg(variants_df, pathways_gmt, interactions_tsv)
```

### 4. Embedding Tensors

**Format**: HDF5 / PyTorch tensors
**Contents**:

| Key | Shape | Description |
|-----|-------|-------------|
| gene_embeddings | (n_genes, embed_dim) | Gene representations |
| variant_embeddings | (n_variants, embed_dim) | Variant representations |
| pathway_embeddings | (n_pathways, embed_dim) | Pathway representations |
| sample_embeddings | (n_samples, embed_dim) | Sample representations |

**Default embed_dim**: 128

---

## Output Artifacts

### 1. Subtype Assignments

**Format**: CSV
**Location**: `outputs/<run_id>/subtype_assignments.csv`

| Column | Type | Description |
|--------|------|-------------|
| sample_id | string | Sample identifier |
| cluster_id | int | Assigned cluster (0-indexed) |
| cluster_label | string | Biological interpretation |
| confidence | float | Assignment confidence [0,1] |
| top_pathways | string | Semicolon-separated pathway names |

### 2. Pathway Enrichment Results

**Format**: CSV
**Location**: `outputs/<run_id>/pathway_enrichment.csv`

| Column | Type | Description |
|--------|------|-------------|
| cluster_id | int | Cluster identifier |
| pathway | string | Pathway name |
| genes_in_cluster | int | Count of mutated genes |
| genes_in_pathway | int | Total genes in pathway |
| fold_enrichment | float | Observed/expected ratio |
| p_value | float | Fisher's exact test |
| q_value | float | FDR-corrected p-value |

### 3. Embeddings Archive

**Format**: HDF5
**Location**: `outputs/<run_id>/embeddings.h5`

```python
import h5py
with h5py.File("outputs/run_001/embeddings.h5", "r") as f:
    gene_emb = f["gene_embeddings"][:]
    sample_emb = f["sample_embeddings"][:]
```

### 4. Visualization Files

**Format**: PNG/SVG/HTML
**Location**: `outputs/<run_id>/figures/`

| File | Description |
|------|-------------|
| umap_clusters.png | 2D UMAP of sample embeddings |
| pathway_heatmap.png | Cluster × pathway enrichment |
| network_viz.html | Interactive knowledge graph |
| phenotype_distribution.png | Phenotype by cluster |

### 5. Run Metadata

**Format**: YAML
**Location**: `outputs/<run_id>/run_metadata.yaml`

```yaml
run_id: "demo_20240115_143022"
config: "configs/demo.yaml"
timestamp: "2024-01-15T14:30:22Z"
git_commit: "abc123def"
random_seed: 42
input_files:
  vcf: "examples/demo_data/demo_variants.vcf"
  phenotypes: "examples/demo_data/demo_phenotypes.csv"
  pathways: "examples/demo_data/demo_pathways.gmt"
parameters:
  n_clusters: 3
  embedding_dim: 128
  min_cadd_score: 15.0
runtime_seconds: 847.3
validation:
  silhouette_score: 0.72
  pathway_recovery: 0.85
```

---

## Data Flow by Module

| Module | Input | Output |
|--------|-------|--------|
| 01 Data Loaders | VCF, CSV, GMT | Variant matrix, phenotype df |
| 02 Variant Processing | Variant matrix | Annotated variants |
| 03 Knowledge Graph | Annotated variants, pathways | NetworkX graph |
| 04 Embeddings | Knowledge graph | Embedding tensors |
| 05 Foundation Models | Sequences, embeddings | Enhanced embeddings |
| 06 GNN | Graph, features | Node embeddings |
| 07 Clustering | Sample embeddings | Cluster assignments |
| 08 Subtype Clustering | Clusters, pathways | Subtype profiles |
| 09 Statistical Analysis | Clusters, phenotypes | Enrichment stats |
| 10 Causal Inference | Variants, phenotypes | Causal estimates |
| 11 Interpretability | All embeddings | SHAP values, reports |
| 12 Visualization | All outputs | Figures, dashboards |

---

## Validation Schema

For automated validation, see `modules/module_01_data_loaders/validators/`:

```python
from modules.module_01_data_loaders.validators import validate_vcf, validate_phenotypes

# Returns (is_valid: bool, errors: list[str])
is_valid, errors = validate_vcf("path/to/file.vcf")
is_valid, errors = validate_phenotypes("path/to/phenotypes.csv")
```
