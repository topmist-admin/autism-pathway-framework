# External Data Sources

> This document tracks the external biological databases used in the Autism Pathway Framework, including download URLs, file names, and download history.

---

## Required Databases

### 1. Gene Ontology (GO)

**Purpose**: Provides hierarchical functional annotations for genes

| File | URL | Local Path |
|------|-----|------------|
| go-basic.obo | http://purl.obolibrary.org/obo/go/go-basic.obo | `data/raw/go-basic.obo` |
| goa_human.gaf.gz | http://geneontology.org/gene-associations/goa_human.gaf.gz | `data/raw/goa_human.gaf.gz` |

**Download method**: Automated via `scripts/download_databases.sh`

---

### 2. gnomAD Constraint Scores

**Purpose**: Gene-level constraint metrics (pLI, LOEUF) indicating tolerance to loss-of-function variants

| File | URL | Local Path |
|------|-----|------------|
| gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz | https://storage.googleapis.com/gcp-public-data--gnomad/release/2.1.1/constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz | `data/raw/gnomad.v2.1.1.lof_metrics.by_gene.txt` |

**Download method**: Automated via `scripts/download_databases.sh`

**Key columns used**:
- `gene` - Gene symbol
- `pLI` - Probability of loss-of-function intolerance
- `oe_lof_upper` - LOEUF (loss-of-function observed/expected upper bound)
- `mis_z` - Missense Z-score

---

### 3. Reactome Pathways

**Purpose**: Curated biological pathway definitions

| File | URL | Local Path |
|------|-----|------------|
| ReactomePathways.gmt | https://reactome.org/download/current/ReactomePathways.gmt.zip | `data/raw/ReactomePathways.gmt` |

**Download method**: Automated via `scripts/download_databases.sh`

---

### 4. STRING Protein-Protein Interaction Network

**Purpose**: Protein interaction network for network-based analysis

| File | URL | Local Path |
|------|-----|------------|
| 9606.protein.links.v12.0.txt.gz | https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz | `data/raw/9606.protein.links.v12.0.txt` |

**Download method**: Automated via `scripts/download_databases.sh --all`
**Note**: Large file (~300MB compressed), only downloaded with `--all` flag

---

### 5. SFARI Gene Database

**Purpose**: Curated list of autism-associated genes with confidence scores

| File | Source | Local Path |
|------|--------|------------|
| SFARI-Gene_genes_01-20-2025export.csv | https://gene.sfari.org/ | `data/raw/SFARI-Gene_genes.csv` |

**Download method**: Manual (requires website registration)

**Dataset selected**: **Gene scoring** (from options: HumanGene, Gene scoring, CNV, Mouse Models)

**Key columns used**:
- `gene-symbol` - Gene symbol (e.g., CHD8, SHANK3)
- `gene-score` - SFARI confidence score (1=high confidence, 2=strong candidate, 3=suggestive)
- `syndromic` - Whether associated with syndromic autism (0 or 1)
- `genetic-category` - Evidence category

**Download instructions**:
1. Go to https://gene.sfari.org/
2. Register/login for access
3. Navigate to "Gene scoring" dataset
4. Export as CSV
5. Rename to `SFARI-Gene_genes.csv` and place in `data/raw/`

---

### 6. BrainSpan Developmental Transcriptome

**Purpose**: Human brain gene expression across developmental stages

| File | Source | Local Path |
|------|--------|------------|
| RNA-Seq Gencode v10 summarized to genes | https://www.brainspan.org/ | `data/raw/brainspan/` |

**Download method**: Manual (requires website registration)

**Dataset selected**: **RNA-Seq Gencode v10 summarized to genes** (from Developmental Transcriptome Dataset)

Other available datasets (not required):
- RNA-Seq Gencode v10 summarized to exons
- Exon microarray summarized to probe sets
- Exon microarray summarized to genes
- Prenatal LMD Microarray Dataset
- 3-D Fiber Tract annotations
- MRI/DTI, Methylation, MicroRNA data

**Download instructions**:
1. Go to https://www.brainspan.org/
2. Register for access
3. Navigate to "Developmental Transcriptome Dataset"
4. Download "RNA-Seq Gencode v10 summarized to genes"
5. Extract contents to `data/raw/brainspan/`

**Expected files after extraction**:
- Expression matrix (genes × samples)
- Sample metadata (brain region, age, donor info)
- Gene annotation file

---

## Validation Summary

### SFARI Gene Database Validation (2025-01-20)

| Metric | Value |
|--------|-------|
| Total genes | 1,267 |
| Score 1 (High confidence) | 244 |
| Score 2 (Strong candidate) | 708 |
| Score 3 (Suggestive) | 221 |
| Unscored | 94 |
| Syndromic genes | 312 |

**Key ASD genes verified**: SHANK3, CHD8, SCN2A, NRXN1, GRIN2B (all Score 1)

### BrainSpan Developmental Transcriptome Validation (2025-01-20)

| Metric | Value |
|--------|-------|
| Total genes | 52,376 |
| Total samples | 524 |
| Brain regions | 26 |
| Donors (Male) | 298 |
| Donors (Female) | 226 |
| Developmental stages | Prenatal (8-37 pcw) to Adult (40 yrs) |

**Files validated**:
- `expression_matrix.csv` (183 MB) - RPKM values, genes × samples
- `columns_metadata.csv` - Sample info (donor, age, gender, brain region)
- `rows_metadata.csv` - Gene info (gene_id, ensembl_id, gene_symbol, entrez_id)

**Key ASD genes found in expression data**: SHANK3, CHD8, SCN2A, NRXN1, GRIN2B

---

## Download History

| Date | Database | Version/File | Status | Notes |
|------|----------|--------------|--------|-------|
| 2025-01-20 | Gene Ontology | go-basic.obo | Downloaded | Via download script |
| 2025-01-20 | GO Annotations | goa_human.gaf.gz | Downloaded | Via download script |
| 2025-01-20 | gnomAD Constraints | v2.1.1 | Downloaded | Decompressed from .bgz |
| 2025-01-20 | Reactome Pathways | Current | Downloaded | Via download script |
| 2025-01-20 | STRING PPI | v12.0 | ✅ GCS + BigQuery | 13.7M interactions in BigQuery |
| 2025-01-20 | SFARI Genes | 01-20-2025 export | ✅ Validated | 1,267 genes; Score 1: 244, Score 2: 708, Score 3: 221 |
| 2025-01-20 | BrainSpan | Gencode v10 genes | ✅ Validated | 52,376 genes × 524 samples; 26 brain regions |

---

## Automated Download Script

The `scripts/download_databases.sh` script automates downloading of most databases:

```bash
# Minimal download (excludes large files)
bash scripts/download_databases.sh --minimal

# Full download (includes STRING PPI ~300MB)
bash scripts/download_databases.sh --all
```

---

## Data Update Schedule

| Database | Update Frequency | Last Checked |
|----------|------------------|--------------|
| Gene Ontology | Monthly | 2025-01-20 |
| gnomAD | Major releases (~yearly) | 2025-01-20 |
| Reactome | Quarterly | 2025-01-20 |
| STRING | Major releases (~yearly) | 2025-01-20 |
| SFARI | Continuous updates | 2025-01-20 |
| BrainSpan | Static dataset | 2025-01-20 |

---

## Google Cloud Platform Storage

All data files are stored on Google Cloud Platform for scalability and collaboration.

### GCS Bucket

**Bucket**: `gs://autism-pathway-data`
**Region**: us-central1

| Path | Contents | Size |
|------|----------|------|
| `raw/go-basic.obo` | Gene Ontology structure | 31 MB |
| `raw/goa_human.gaf` | GO annotations | 190 MB |
| `raw/ReactomePathways.gmt` | Reactome pathways | 1 MB |
| `raw/SFARI-Gene_genes.csv` | SFARI genes | 123 KB |
| `raw/gnomad_constraints.txt` | gnomAD constraints | 13 MB |
| `raw/9606.protein.links.v12.0.txt` | STRING PPI | 631 MB |
| `raw/brainspan/` | BrainSpan expression | 186 MB |

### BigQuery Dataset

**Project**: `autism-pathway-framework`
**Dataset**: `autism_genetics`

| Table | Description | Rows |
|-------|-------------|------|
| `string_ppi` | STRING protein interactions | 13,715,404 |
| `gnomad_constraints` | Gene constraint scores | 19,704 |

### Authentication

Uses Application Default Credentials (ADC):

```bash
gcloud auth application-default login --project=autism-pathway-framework
```

### Configuration

See [configs/gcp_config.yaml](../configs/gcp_config.yaml) for full GCP configuration.

---

## See Also

- [Data Formats](data_formats.md) - File format specifications
- [Configuration](configuration.md) - Environment variable setup
- [GCP Config](../configs/gcp_config.yaml) - Google Cloud configuration
- [download_databases.sh](../scripts/download_databases.sh) - Automated download script
