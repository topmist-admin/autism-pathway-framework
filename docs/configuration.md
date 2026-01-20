# Configuration Guide

> This document describes all configurable parameters in the Autism Pathway Framework and provides example configuration files.

## Configuration File Format

The framework uses YAML configuration files. The default configuration is loaded from `configs/default.yaml` and can be overridden with environment-specific configs.

## Configuration Sections

### 1. Data Loading Configuration

```yaml
data_loading:
  # VCF Loading
  vcf:
    min_quality: 20.0           # Minimum QUAL score to include variant
    min_depth: 10               # Minimum read depth (DP)
    min_genotype_quality: 20    # Minimum genotype quality (GQ)
    filter_pass_only: true      # Only include PASS variants
    include_chromosomes:        # Chromosomes to include (null = all)
      - "chr1"
      - "chr2"
      # ... chr1-22, chrX, chrY
    exclude_chromosomes:        # Chromosomes to exclude
      - "chrM"
      - "chrY"                  # Often excluded for consistency

  # Pathway Databases
  pathways:
    min_pathway_size: 5         # Minimum genes per pathway
    max_pathway_size: 500       # Maximum genes per pathway
    sources:                    # Which databases to load
      - "GO_BP"                 # GO Biological Process
      - "GO_MF"                 # GO Molecular Function
      - "Reactome"
      - "KEGG"
    go_evidence_codes:          # GO evidence codes to include
      - "EXP"                   # Experimental
      - "IDA"                   # Direct assay
      - "IMP"                   # Mutant phenotype
      - "IGI"                   # Genetic interaction
      # Exclude: IEA (electronic annotation)

  # Expression Data
  expression:
    brainspan_stages:           # Developmental stages to use
      - "prenatal"
      - "early_postnatal"
      - "adolescent"
      - "adult"
    expression_threshold: 1.0   # Log2(TPM) threshold for "expressed"

  # Gene Constraints
  constraints:
    pli_threshold: 0.9          # pLI score for "constrained"
    loeuf_threshold: 0.35       # LOEUF for "constrained"
```

### 2. Variant Processing Configuration

```yaml
variant_processing:
  # Quality Control
  qc:
    max_allele_frequency: 0.01    # Max gnomAD AF (1%)
    min_allele_frequency: 0.0     # Min AF (usually 0)
    max_missing_rate: 0.1         # Max sample missingness
    max_variant_missing_rate: 0.1 # Max variant missingness

  # Annotation
  annotation:
    consequence_source: "VEP"     # VEP or SnpEff
    canonical_only: true          # Use only canonical transcripts
    include_consequences:         # Consequences to include
      - "frameshift_variant"
      - "stop_gained"
      - "splice_acceptor_variant"
      - "splice_donor_variant"
      - "start_lost"
      - "missense_variant"
      - "inframe_insertion"
      - "inframe_deletion"

  # Gene Burden Calculation
  gene_burden:
    aggregation_method: "weighted_sum"  # weighted_sum, max, count

    # Consequence weights
    consequence_weights:
      frameshift_variant: 1.0
      stop_gained: 1.0
      splice_acceptor_variant: 1.0
      splice_donor_variant: 1.0
      start_lost: 1.0
      missense_variant: 0.5
      inframe_insertion: 0.3
      inframe_deletion: 0.3

    # CADD weighting
    use_cadd_weighting: true
    cadd_threshold: 20.0          # Min CADD phred for missense
    cadd_weight_scale: 0.05       # Weight = CADD * scale

    # REVEL weighting
    use_revel_weighting: true
    revel_threshold: 0.5          # Min REVEL for "damaging"

    # Allele frequency weighting
    use_af_weighting: false
    af_weight_beta: 1.0           # Weight = (1-AF)^beta
```

### 3. Pathway Scoring Configuration

```yaml
pathway_scoring:
  # Score Calculation
  scoring:
    method: "weighted_sum"        # weighted_sum, enrichment, mean
    size_normalization: "sqrt"    # sqrt, linear, log, none
    min_genes_with_data: 3        # Min genes with burden > 0

  # Gene Weighting
  gene_weighting:
    type: "constraint"            # constraint, expression, uniform
    constraint_metric: "pli"      # pli, loeuf, mis_z

  # Normalization
  normalization:
    method: "zscore"              # zscore, minmax, rank
    across_samples: true          # Normalize across cohort
```

### 4. Network Propagation Configuration

```yaml
network_propagation:
  enabled: true

  # Network Construction
  network:
    sources:
      - name: "STRING_PPI"
        weight: 1.0
        min_confidence: 0.7
      - name: "BioGRID"
        weight: 0.5
      - name: "coexpression"
        weight: 0.3
    min_edge_confidence: 0.4
    min_node_degree: 1

  # Propagation Parameters
  propagation:
    method: "rwr"                 # rwr (random walk with restart)
    restart_probability: 0.5      # Alpha parameter (0.3-0.7)
    max_iterations: 100
    convergence_threshold: 1.0e-6

  # Hub Correction
  hub_correction:
    enabled: true
    penalty_power: 0.5            # Degree^(-power)
```

### 5. Clustering Configuration

```yaml
clustering:
  # Dimensionality Reduction
  dimensionality_reduction:
    method: "pca"                 # pca, umap, nmf
    max_dimensions: 50
    variance_threshold: 0.9       # For PCA: keep 90% variance

  # Clustering Algorithm
  clustering:
    method: "gmm"                 # gmm, spectral, hierarchical
    min_clusters: 2
    max_clusters: 10
    selection_metric: "bic"       # bic, silhouette, stability

  # GMM-specific
  gmm:
    covariance_type: "full"       # full, diag, tied, spherical
    n_init: 10                    # Number of initializations

  # Stability Assessment
  stability:
    n_bootstrap: 100
    min_stability: 0.7            # Threshold for stable solution
```

### 6. Symbolic Rules Configuration

```yaml
symbolic_rules:
  enabled: true

  # Rule Selection
  rules:
    - name: "R1_synaptic_relevance"
      enabled: true
      weight: 1.5
    - name: "R2_prenatal_expression"
      enabled: true
      weight: 1.3
    - name: "R3_constraint_priority"
      enabled: true
      weight: 1.4
    - name: "R4_sfari_evidence"
      enabled: true
      weight: 1.2
    - name: "R5_cell_type_specificity"
      enabled: true
      weight: 1.1
    - name: "R6_network_centrality"
      enabled: true
      weight: 1.0

  # Combination Method
  combination:
    method: "learned"             # learned, weighted_average, max
    neural_weight: 0.7            # Weight for neural component
    symbolic_weight: 0.3          # Weight for symbolic component
```

### 7. Output Configuration

```yaml
output:
  # File Formats
  formats:
    gene_burdens: "parquet"       # parquet, csv, json
    pathway_scores: "parquet"
    cluster_assignments: "json"

  # Directories
  directories:
    processed: "data/processed"
    results: "results"
    figures: "results/figures"
    logs: "logs"

  # Logging
  logging:
    level: "INFO"                 # DEBUG, INFO, WARNING, ERROR
    file: "logs/pipeline.log"
    console: true
```

## Example Configuration Files

### Default Configuration (`configs/default.yaml`)

```yaml
# Default configuration for production use
data_loading:
  vcf:
    min_quality: 20.0
    min_depth: 10
    filter_pass_only: true
    exclude_chromosomes: ["chrM"]

variant_processing:
  qc:
    max_allele_frequency: 0.01
  gene_burden:
    aggregation_method: "weighted_sum"
    use_cadd_weighting: true
    cadd_threshold: 20.0

pathway_scoring:
  scoring:
    size_normalization: "sqrt"
    min_genes_with_data: 3

clustering:
  clustering:
    method: "gmm"
    min_clusters: 2
    max_clusters: 10
```

### Test Configuration (`configs/small_test.yaml`)

```yaml
# Configuration for small test runs
data_loading:
  vcf:
    min_quality: 10.0             # More permissive for test data
    min_depth: 5
  pathways:
    min_pathway_size: 3           # Smaller pathways OK for testing
    max_pathway_size: 100

variant_processing:
  qc:
    max_allele_frequency: 0.05    # More permissive

clustering:
  stability:
    n_bootstrap: 10               # Fewer bootstraps for speed
```

### High-Stringency Configuration (`configs/stringent.yaml`)

```yaml
# High-stringency configuration for confident results
data_loading:
  vcf:
    min_quality: 30.0
    min_depth: 20
    min_genotype_quality: 30

variant_processing:
  qc:
    max_allele_frequency: 0.001   # Ultra-rare only
  gene_burden:
    cadd_threshold: 25.0          # Higher CADD threshold
    revel_threshold: 0.7          # Higher REVEL threshold

pathway_scoring:
  scoring:
    min_genes_with_data: 5        # Require more genes
```

## Environment Variables

Configuration values can be overridden with environment variables:

```bash
# Override max allele frequency
export APF_VARIANT_PROCESSING__QC__MAX_ALLELE_FREQUENCY=0.005

# Override clustering method
export APF_CLUSTERING__CLUSTERING__METHOD=spectral

# Override output directory
export APF_OUTPUT__DIRECTORIES__RESULTS=/custom/output/path
```

Environment variable format: `APF_<SECTION>__<SUBSECTION>__<PARAMETER>`

## Loading Configuration in Code

```python
from autism_pathway_framework.config import load_config

# Load default configuration
config = load_config()

# Load specific configuration file
config = load_config("configs/stringent.yaml")

# Override with environment variables
config = load_config(use_env=True)

# Access configuration values
max_af = config.variant_processing.qc.max_allele_frequency
```

## Configuration Validation

The framework validates configuration on load:

```python
from autism_pathway_framework.config import load_config, validate_config

config = load_config("my_config.yaml")
errors = validate_config(config)

if errors:
    for error in errors:
        print(f"Configuration error: {error}")
```

## See Also

- [Testing Guide](testing.md) - Using test configurations
- [Data Formats](data_formats.md) - Input/output file formats
- [API Reference](api_reference.md) - Programmatic configuration access
