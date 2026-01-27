# Demo Data

> **⚠️ RESEARCH USE ONLY** — This is synthetic data for demonstration purposes. All outputs are for research only. Not for clinical decision-making.

This folder contains synthetic demonstration data for the Autism Pathway Framework.

## Purpose

- Enable reproducible end-to-end pipeline testing
- Provide a "golden path" for new users
- Validate outputs without requiring sensitive genomic data

## Contents

| File | Description | Records |
|------|-------------|---------|
| `demo_variants.vcf` | Synthetic variant calls across 20 ASD-associated genes | 20 variants |
| `demo_phenotypes.csv` | Sample metadata with phenotype features | 50 samples |
| `demo_pathways.gmt` | Pathway definitions for subtype clustering | 15 pathways |

## File Details

### demo_variants.vcf

VCF 4.2 format file containing synthetic variants in known ASD-associated genes:

**Genes included:**
- Synaptic: SHANK3, NRXN1, SYNGAP1, GRIN2B, ANK2
- Chromatin: CHD8, ARID1B, ASH1L, KMT2A, ADNP, MECP2
- Ion channel: SCN2A, KCNQ2, CACNA1A, TSC1, TSC2
- Other: FOXP1, PTEN, DYRK1A, NAA15

**Variant types:**
- Missense variants (CADD scores 20-35)
- Frameshift variants (CADD scores 30-40)
- Stop gained variants (CADD scores 35-45)

**Sample genotypes:** 50 synthetic samples (SAMPLE_001 to SAMPLE_050) with realistic allele frequencies.

### demo_phenotypes.csv

CSV file with phenotype data for each sample:

| Column | Type | Values |
|--------|------|--------|
| sample_id | string | SAMPLE_001 - SAMPLE_050 |
| diagnosis | string | ASD |
| age_at_diagnosis | float | 2.5 - 4.3 years |
| sex | string | M (80%), F (20%) |
| iq_score | int | 74 - 98 |
| language_delay | int | 0 or 1 |
| motor_delay | int | 0 or 1 |
| seizure_history | int | 0 or 1 |
| family_history_asd | int | 0 or 1 |
| cohort | string | DEMO_COHORT |
| planted_subtype | string | synaptic, chromatin, or ion_channel |

**Planted subtypes distribution:**
- synaptic: ~18 samples
- chromatin: ~17 samples
- ion_channel: ~15 samples

### demo_pathways.gmt

GMT (Gene Matrix Transposed) format with 15 pathways:

| Pathway | Category | Genes |
|---------|----------|-------|
| SYNAPTIC_TRANSMISSION | Synaptic | SHANK3, NRXN1, SYNGAP1, GRIN2B, ANK2 |
| CHROMATIN_REMODELING | Chromatin | CHD8, ARID1B, ASH1L, KMT2A, ADNP, MECP2 |
| ION_CHANNEL_REGULATION | Ion channel | SCN2A, KCNQ2, CACNA1A, TSC1, TSC2 |
| SYNAPTIC_SCAFFOLDING | Synaptic | SHANK3, SYNGAP1, ANK2, NRXN1 |
| TRANSCRIPTION_REGULATION | Chromatin | FOXP1, CHD8, ADNP, MECP2, ARID1B |
| MTOR_SIGNALING | Ion channel | PTEN, TSC1, TSC2, DYRK1A |
| NEURONAL_DEVELOPMENT | General | CHD8, FOXP1, ADNP, DYRK1A, NAA15 |
| GLUTAMATERGIC_SIGNALING | Synaptic | GRIN2B, SHANK3, SYNGAP1, NRXN1 |
| SODIUM_CHANNEL_COMPLEX | Ion channel | SCN2A, ANK2 |
| POTASSIUM_CHANNEL_COMPLEX | Ion channel | KCNQ2, CACNA1A |
| HISTONE_MODIFICATION | Chromatin | KMT2A, ASH1L, CHD8, ARID1B |
| PROTEIN_ACETYLATION | Chromatin | NAA15, ADNP |
| ASD_HIGH_CONFIDENCE | Meta | CHD8, SCN2A, SHANK3, SYNGAP1, ADNP, ARID1B, ASH1L, DYRK1A, GRIN2B, PTEN |
| ASD_SYNDROMIC | Meta | TSC1, TSC2, MECP2, PTEN, CACNA1A |
| NEUREXIN_NEUROLIGIN | Synaptic | NRXN1, SHANK3, ANK2 |

## Privacy Notice

This is entirely synthetic data generated for demonstration purposes.
It does not contain any real patient information.

## Usage

```bash
# Run demo pipeline with this data
python -m autism_pathway_framework --config configs/demo.yaml
```

## Validation

The demo data includes **planted ground truth** to validate clustering:

1. Samples are assigned to one of three subtypes based on their variant profiles
2. The `planted_subtype` column provides expected cluster assignments
3. A successful run should achieve ARI > 0.7 when comparing predicted vs planted subtypes

```python
from sklearn.metrics import adjusted_rand_score

# After running the pipeline
ari = adjusted_rand_score(
    phenotypes['planted_subtype'],
    results['cluster_label']
)
print(f"Adjusted Rand Index: {ari:.3f}")  # Expected: > 0.7
```

## Deterministic Generation

The demo data was generated with `seed=42` for full reproducibility. See [docs/reproducibility.md](../../docs/reproducibility.md) for the seed strategy.
