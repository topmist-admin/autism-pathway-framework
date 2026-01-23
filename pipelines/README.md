# Pipelines

End-to-end analysis pipelines for the Autism Pathway Framework.

## Overview

This module provides complete workflows that integrate multiple framework modules
to perform complex analyses from raw genetic data to interpretable results.

## Available Pipelines

### SubtypeDiscoveryPipeline (Session 17)

Identifies autism subtypes from genetic variation data through pathway-based clustering.

**Pipeline Steps:**
1. **Data Loading** - Load VCF and pathway databases
2. **Variant Processing** - QC filtering, annotation, gene burden calculation
3. **Pathway Scoring** - Aggregate gene burdens to pathway scores
4. **Clustering** - Identify subtypes via GMM/spectral/hierarchical clustering
5. **Stability Analysis** - Bootstrap validation of cluster stability
6. **Characterization** - Generate subtype profiles with top pathways
7. **Validation** - Confound analysis and negative controls (optional)

## Quick Start

```python
from pipelines import (
    SubtypeDiscoveryPipeline,
    PipelineConfig,
    DataConfig,
    run_subtype_discovery,
)

# Option 1: Convenience function (minimal configuration)
result = run_subtype_discovery(
    vcf_path="cohort.vcf.gz",
    pathway_path="reactome.gmt",
    n_clusters=5,
)
print(result.summary)

# Option 2: Full configuration
config = PipelineConfig(
    data=DataConfig(
        vcf_path="cohort.vcf.gz",
        pathway_gmt_path="reactome.gmt",
        gnomad_constraints_path="gnomad_constraints.tsv",
    ),
    processing=ProcessingConfig(
        min_quality=30.0,
        max_allele_freq=0.01,
        use_constraint_weighting=True,
    ),
    clustering=ClusteringPipelineConfig(
        n_clusters=5,
        run_stability=True,
        n_bootstrap=100,
    ),
    output_dir="results/",
    save_intermediate=True,
)

pipeline = SubtypeDiscoveryPipeline(config)
result = pipeline.run()
```

## Configuration Reference

### DataConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vcf_path` | str | Required | Path to input VCF file |
| `pathway_gmt_path` | str | None | Path to GMT pathway file |
| `pathway_go_obo_path` | str | None | Path to GO OBO file |
| `pathway_go_gaf_path` | str | None | Path to GO GAF annotations |
| `gnomad_constraints_path` | str | None | Path to gnomAD constraints |
| `knowledge_graph_path` | str | None | Path to pre-built knowledge graph |
| `ppi_network_path` | str | None | Path to PPI network (STRING) |
| `ppi_min_score` | float | 700.0 | Minimum PPI confidence score |
| `gene_id_type` | str | "symbol" | Gene ID type (symbol, ensembl, entrez) |

### ProcessingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_quality` | float | 30.0 | Minimum variant quality score |
| `min_depth` | int | 10 | Minimum read depth |
| `filter_pass_only` | bool | True | Only keep PASS variants |
| `max_allele_freq` | float | 0.01 | Maximum allele frequency (rare variants) |
| `use_cadd_weighting` | bool | False | Weight by CADD scores |
| `cadd_threshold` | float | 20.0 | CADD score threshold |
| `use_constraint_weighting` | bool | True | Weight by gene constraint (pLI) |
| `burden_aggregation` | str | "weighted_sum" | Aggregation method |

### PathwayScoringConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aggregation_method` | AggregationMethod | WEIGHTED_SUM | Pathway aggregation method |
| `min_pathway_size` | int | 5 | Minimum genes per pathway |
| `max_pathway_size` | int | 500 | Maximum genes per pathway |
| `normalize_by_pathway_size` | bool | True | Normalize by pathway size |
| `use_network_propagation` | bool | False | Apply network propagation |
| `propagation_method` | PropagationMethod | RANDOM_WALK | Propagation algorithm |
| `restart_prob` | float | 0.5 | Random walk restart probability |
| `normalization_method` | NormalizationMethod | ZSCORE | Score normalization |

### ClusteringPipelineConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | ClusteringMethod | GMM | Clustering algorithm |
| `n_clusters` | int | None | Number of clusters (auto if None) |
| `min_clusters` | int | 2 | Minimum clusters for auto-detection |
| `max_clusters` | int | 10 | Maximum clusters for auto-detection |
| `run_stability` | bool | True | Run bootstrap stability analysis |
| `n_bootstrap` | int | 100 | Number of bootstrap iterations |
| `n_top_pathways` | int | 10 | Top pathways per subtype |
| `min_fold_change` | float | 1.5 | Minimum fold change for significance |
| `use_fdr_correction` | bool | True | Apply FDR correction |
| `run_confound_analysis` | bool | False | Test for confounding variables |
| `run_negative_controls` | bool | False | Run permutation tests |
| `n_permutations` | int | 100 | Number of permutations |

## Result Object

`SubtypeDiscoveryResult` contains:

```python
@dataclass
class SubtypeDiscoveryResult:
    # Core results
    clustering_result: ClusteringResult  # Cluster assignments and metrics
    subtype_profiles: List[SubtypeProfile]  # Characterized subtypes
    n_subtypes: int  # Number of identified subtypes

    # Intermediate data
    pathway_scores: PathwayScoreMatrix  # Sample x pathway scores
    gene_burdens: GeneBurdenMatrix  # Sample x gene burdens

    # Validation (optional)
    stability_result: Optional[StabilityResult]
    confound_report: Optional[ConfoundReport]
    negative_control_report: Optional[NegativeControlReport]

    # Reports
    qc_report: Optional[QCReport]
    validation_report: Optional[ValidationReport]

    # Metadata
    provenance: Optional[ProvenanceRecord]
    config: Optional[PipelineConfig]
    runtime_seconds: float
    timestamp: str
```

### Useful Methods

```python
# Get summary report
print(result.summary)

# Get subtype for a sample
subtype = result.get_subtype_assignment("SAMPLE001")

# Get all samples in a subtype
samples = result.get_samples_by_subtype(subtype_id=0)

# Access pathway scores as DataFrame
df = result.pathway_scores.to_dataframe()

# Access subtype profiles
for profile in result.subtype_profiles:
    print(f"Subtype {profile.subtype_id}: {profile.n_samples} samples")
    for pathway in profile.top_pathways[:3]:
        print(f"  - {pathway.pathway_name} (FC={pathway.fold_change:.2f})")
```

## Command Line Interface

```bash
python -m pipelines.subtype_discovery \
    --vcf cohort.vcf.gz \
    --pathways reactome.gmt \
    --n-clusters 5 \
    --output results/ \
    --verbose
```

## Module Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                  SubtypeDiscoveryPipeline                    │
└─────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
    ▼                         ▼                         ▼
┌─────────┐           ┌─────────────┐           ┌─────────────┐
│Module 01│           │  Module 02  │           │  Module 03  │
│  Data   │──────────▶│  Variant    │◀──────────│  Knowledge  │
│ Loaders │           │ Processing  │           │    Graph    │
└─────────┘           └─────────────┘           └─────────────┘
                              │
                              ▼
                      ┌─────────────┐
                      │  Module 07  │
                      │   Pathway   │
                      │   Scoring   │
                      └─────────────┘
                              │
                              ▼
                      ┌─────────────┐
                      │  Module 08  │
                      │   Subtype   │
                      │  Clustering │
                      └─────────────┘
```

## Validation and Research Integrity

The pipeline includes optional validation steps to ensure research integrity:

### Stability Analysis
- Bootstrap resampling to assess cluster reproducibility
- Adjusted Rand Index (ARI) across bootstrap samples
- Identifies unstable samples that shouldn't be interpreted

### Confound Analysis
- Tests for correlation between clusters and confounding variables
- Supports: ancestry, batch, sex, age, sequencing site
- Returns risk assessment and recommendations

### Negative Controls
- Permutation tests to validate non-random clustering
- Random gene set baselines
- Label shuffle tests

```python
# Enable validation
config = PipelineConfig(
    data=DataConfig(...),
    clustering=ClusteringPipelineConfig(
        run_stability=True,
        run_confound_analysis=True,
        run_negative_controls=True,
        n_permutations=1000,
    ),
)

result = pipeline.run(
    confounds={
        "ancestry": ancestry_array,
        "batch": batch_array,
        "sex": sex_array,
    }
)

# Check results
if result.stability_result.is_stable:
    print("Clusters are stable")

if result.confound_report.overall_risk == "low":
    print("No confounding detected")

if result.negative_control_report.permutation_result.p_value < 0.05:
    print("Clustering is significant")
```

---

## Session 18 Pipelines

### TherapeuticHypothesisPipeline

Extends SubtypeDiscoveryPipeline with therapeutic hypothesis generation.

**Pipeline Steps:**
1. **Subtype Discovery** - Full Session 17 pipeline
2. **Symbolic Rules** - Apply rules R1-R7 to each individual
3. **Hypothesis Generation** - Map pathways to drug candidates
4. **Hypothesis Ranking** - Rank by evidence and relevance
5. **Causal Validation** - Validate with do-calculus

**IMPORTANT**: All hypotheses are RESEARCH HYPOTHESES only.

```python
from pipelines import (
    TherapeuticHypothesisPipeline,
    TherapeuticPipelineConfig,
    TherapeuticConfig,
    DataConfig,
)

config = TherapeuticPipelineConfig(
    data=DataConfig(
        vcf_path="cohort.vcf.gz",
        pathway_gmt_path="reactome.gmt",
    ),
    therapeutic=TherapeuticConfig(
        enable_rules=True,
        enable_causal_validation=True,
        min_pathway_zscore=1.5,
        max_hypotheses=50,
    ),
)

pipeline = TherapeuticHypothesisPipeline(config)
result = pipeline.run()

# View results
print(result.summary)

# Access hypotheses
for hyp in result.ranking_result.top_hypotheses:
    print(hyp.summary())

# View per-individual analysis
for sample_id, analysis in result.individual_analyses.items():
    print(f"{sample_id}: {analysis.n_rules_fired} rules fired")
```

#### TherapeuticConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_rules` | bool | True | Apply symbolic rules (R1-R7) |
| `enable_causal_validation` | bool | True | Validate with causal inference |
| `min_pathway_zscore` | float | 1.5 | Min pathway score for hypotheses |
| `max_hypotheses` | int | 50 | Max hypotheses to generate |
| `weight_evidence` | float | 0.4 | Weight for evidence score |
| `weight_pathway_score` | float | 0.3 | Weight for pathway disruption |

---

### CausalAnalysisPipeline

Standalone causal reasoning for individual case analysis.

**Capabilities:**
- Intervention queries (do-calculus)
- Counterfactual reasoning ("What if gene wasn't mutated?")
- Mediation analysis (gene -> pathway -> phenotype)
- Key causal driver identification

```python
from pipelines import (
    CausalAnalysisPipeline,
    CausalAnalysisConfig,
)

config = CausalAnalysisConfig(
    sample_id="PATIENT_001",
    variant_genes=["SHANK3", "CHD8", "SCN2A"],
    disrupted_pathways=["synaptic_transmission", "chromatin_remodeling"],
    gene_effects={"SHANK3": 0.8, "CHD8": 0.7},
)

pipeline = CausalAnalysisPipeline(config)
result = pipeline.run()

# View summary
print(result.summary)

# Query specific intervention
effect = pipeline.query_intervention(
    treatment="synaptic_transmission",
    outcome="asd_phenotype",
)

# Counterfactual query
cf_result = pipeline.query_counterfactual(
    factual={"SHANK3_function": 0.3},
    counterfactual={"SHANK3_function": 1.0},
    outcome="asd_phenotype",
)

# Mediation analysis
mediation = pipeline.query_mediation(
    treatment="SHANK3_function",
    outcome="asd_phenotype",
    mediator="synaptic_transmission",
)
print(f"Proportion mediated: {mediation.proportion_mediated:.1%}")
```

#### CausalAnalysisConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_id` | str | "individual" | Sample identifier |
| `variant_genes` | List[str] | [] | Genes with variants |
| `disrupted_pathways` | List[str] | [] | Disrupted pathways |
| `gene_effects` | Dict[str, float] | {} | Gene effect sizes |
| `use_sample_model` | bool | True | Use built-in ASD causal model |
| `run_intervention_analysis` | bool | True | Run do-calculus queries |
| `run_counterfactual_analysis` | bool | True | Run counterfactual queries |
| `run_mediation_analysis` | bool | True | Run mediation analysis |

---

## Module Dependencies (Complete)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TherapeuticHypothesisPipeline                          │
│                    (extends SubtypeDiscoveryPipeline)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Module 09     │         │   Module 11     │         │   Module 12     │
│ Symbolic Rules  │         │   Therapeutic   │         │    Causal       │
│   (R1-R7)       │         │   Hypotheses    │         │   Inference     │
└─────────────────┘         └─────────────────┘         └─────────────────┘
         │                           │                           │
         └───────────────────────────┴───────────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │ SubtypeDiscovery    │
                          │ Pipeline (Session 17)│
                          └─────────────────────┘
                                     │
    ┌─────────────┬──────────────────┼──────────────────┬─────────────┐
    │             │                  │                  │             │
    ▼             ▼                  ▼                  ▼             ▼
┌───────┐   ┌───────────┐    ┌─────────────┐    ┌───────────┐   ┌───────────┐
│Mod 01 │   │  Mod 02   │    │   Mod 03    │    │  Mod 07   │   │  Mod 08   │
│ Data  │   │ Variant   │    │ Knowledge   │    │ Pathway   │   │ Subtype   │
│Loaders│   │Processing │    │   Graph     │    │ Scoring   │   │Clustering │
└───────┘   └───────────┘    └─────────────┘    └───────────┘   └───────────┘
```

---

## CLI Reference

```bash
# Session 17: Subtype Discovery
python -m pipelines.subtype_discovery \
    --vcf cohort.vcf.gz \
    --pathways reactome.gmt \
    --n-clusters 5 \
    --output results/

# Session 18: Therapeutic Hypothesis
python -m pipelines.therapeutic_hypothesis \
    --vcf cohort.vcf.gz \
    --pathways reactome.gmt \
    --output results/ \
    --no-causal  # Disable causal validation

# Session 18: Causal Analysis
python -m pipelines.causal_analysis \
    --sample-id PATIENT_001 \
    --genes SHANK3 CHD8 SCN2A \
    --pathways synaptic_transmission chromatin_remodeling
```
