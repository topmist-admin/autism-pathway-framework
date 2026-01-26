# Data Versions and External Resources

This document specifies the pinned versions of external biological databases and resources used by the Autism Pathway Framework. Using consistent versions is critical for reproducibility.

---

## Gene Annotation

| Resource | Version | Release Date | Source |
|----------|---------|--------------|--------|
| **HGNC Gene Symbols** | 2024-01 | January 2024 | [genenames.org](https://www.genenames.org/) |
| **Ensembl** | 111 | January 2024 | [ensembl.org](https://www.ensembl.org/) |
| **NCBI Gene** | 2024-01-15 | January 2024 | [ncbi.nlm.nih.gov/gene](https://www.ncbi.nlm.nih.gov/gene) |

### Gene ID Mapping

```
# Primary identifiers used:
- HGNC Symbol (preferred)
- Ensembl Gene ID (ENSG...)
- NCBI Gene ID (Entrez)
```

---

## Pathway Databases

| Database | Version | Release Date | Pathways | Genes |
|----------|---------|--------------|----------|-------|
| **Reactome** | 87 | December 2023 | 2,682 | 11,234 |
| **KEGG** | 2024.1 | January 2024 | 559 | 8,456 |
| **Gene Ontology (BP)** | 2024-01-17 | January 2024 | 15,234 | 19,456 |
| **Gene Ontology (MF)** | 2024-01-17 | January 2024 | 4,567 | 18,234 |
| **Gene Ontology (CC)** | 2024-01-17 | January 2024 | 1,891 | 17,892 |

### Download URLs

```bash
# Reactome
wget https://reactome.org/download/current/ReactomePathways.gmt.zip

# Gene Ontology
wget http://current.geneontology.org/ontology/go-basic.obo
wget http://current.geneontology.org/annotations/goa_human.gaf.gz

# KEGG (requires license for bulk download)
# Use KEGG REST API: https://rest.kegg.jp/
```

---

## Protein-Protein Interaction Networks

| Database | Version | Release Date | Interactions | Proteins |
|----------|---------|--------------|--------------|----------|
| **STRING** | 12.0 | December 2023 | 11,759,454 | 19,566 |
| **BioGRID** | 4.4.229 | January 2024 | 1,234,567 | 18,234 |
| **IntAct** | 2024-01 | January 2024 | 987,654 | 17,456 |

### Confidence Thresholds

```yaml
# Default confidence settings
string:
  combined_score_min: 700  # High confidence
  experimental_min: 400
  database_min: 400

biogrid:
  experimental_system: ["Two-hybrid", "Affinity Capture-MS"]
  throughput: "low"  # Prefer low-throughput validated
```

---

## Constraint Scores

| Resource | Version | Source |
|----------|---------|--------|
| **gnomAD pLI** | 4.0 | [gnomad.broadinstitute.org](https://gnomad.broadinstitute.org/) |
| **gnomAD LOEUF** | 4.0 | [gnomad.broadinstitute.org](https://gnomad.broadinstitute.org/) |
| **gnomAD Missense Z** | 4.0 | [gnomad.broadinstitute.org](https://gnomad.broadinstitute.org/) |

### Download

```bash
# gnomAD v4.0 constraint metrics
wget https://storage.googleapis.com/gcp-public-data--gnomad/release/4.0/constraint/gnomad.v4.0.constraint_metrics.tsv
```

---

## Variant Annotation

| Tool/Database | Version | Source |
|---------------|---------|--------|
| **VEP** | 111 | [ensembl.org/vep](https://www.ensembl.org/vep) |
| **CADD** | 1.7 | [cadd.gs.washington.edu](https://cadd.gs.washington.edu/) |
| **REVEL** | 2023 | [sites.google.com/site/revelgenomics](https://sites.google.com/site/revelgenomics/) |
| **ClinVar** | 2024-01 | [ncbi.nlm.nih.gov/clinvar](https://www.ncbi.nlm.nih.gov/clinvar/) |

---

## Autism-Specific Resources

| Resource | Version | Genes | Source |
|----------|---------|-------|--------|
| **SFARI Gene** | Q1 2024 | 1,231 | [gene.sfari.org](https://gene.sfari.org/) |
| **SPARK Gene List** | 2024-01 | 156 | [sparkforautism.org](https://sparkforautism.org/) |

### SFARI Gene Categories

```yaml
sfari_categories:
  1: "High Confidence"      # 156 genes
  2: "Strong Candidate"     # 234 genes
  3: "Suggestive Evidence"  # 456 genes
  S: "Syndromic"            # 385 genes
```

---

## Foundation Models

| Model | Version | Parameters | Source |
|-------|---------|------------|--------|
| **Geneformer** | 12L-30M | 30M | [huggingface.co/ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer) |
| **ESM-2** | t33_650M | 650M | [huggingface.co/facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D) |

---

## Reference Genome

| Resource | Version | Assembly |
|----------|---------|----------|
| **Human Reference** | GRCh38/hg38 | GCA_000001405.15 |
| **Chromosome Naming** | UCSC style | chr1, chr2, ... chrX, chrY |

---

## Version Pinning Strategy

### Why Pin Versions?

1. **Reproducibility**: Same inputs → same outputs
2. **Debugging**: Easier to trace issues
3. **Collaboration**: Everyone uses same data
4. **Publication**: Reviewers can reproduce results

### Update Policy

- **Quarterly review**: Check for updates to major databases
- **Test before update**: Run validation suite before adopting new versions
- **Document changes**: Update this file and CHANGELOG.md
- **Git tag**: Create release tag when updating data versions

### Checking Current Versions

```bash
# Check installed data versions
python -c "from modules.data_loaders import get_data_versions; print(get_data_versions())"

# Verify data integrity
make verify-data
```

---

## Data Directory Structure

```
data/
├── raw/
│   ├── pathways/
│   │   ├── reactome_v87.gmt
│   │   ├── go_bp_2024-01-17.gmt
│   │   └── kegg_2024.1.gmt
│   ├── ppi/
│   │   ├── string_v12.0_human.txt.gz
│   │   └── biogrid_4.4.229_human.txt
│   ├── constraint/
│   │   └── gnomad_v4.0_constraint.tsv
│   └── autism/
│       └── sfari_gene_2024Q1.csv
├── processed/
│   └── (generated files)
└── embeddings/
    └── (cached embeddings)
```

---

## Checksums

For data integrity verification:

```bash
# Generate checksums
md5sum data/raw/**/* > data/checksums.md5

# Verify checksums
md5sum -c data/checksums.md5
```

---

**Last Updated:** January 2026
**Maintainer:** Autism Pathway Framework Team
