# SFARI Data Access Application

Copy-paste these sections into the SFARI Base application form.

---

## Project Title

```
Validation of Pathway-Based Molecular Subtyping in Autism Spectrum Disorder
```

---

## Principal Investigator

```
Name: Rohit Chauhan
Email: info@topmist.com
```

---

## Project Summary (250 words max)

```
Autism Spectrum Disorder (ASD) is genetically heterogeneous, with hundreds of implicated genes making it difficult to identify clinically meaningful subtypes. Traditional gene-level analyses often fail to replicate across cohorts due to allelic and locus heterogeneity.

We have developed the Autism Pathway Framework, an open-source computational tool that aggregates rare variant burden at the biological pathway level rather than individual genes. This approach captures convergent biology across different genetic variants, potentially improving replicability and biological interpretability.

The framework has been validated on synthetic data and is publicly available (DOI: 10.5281/zenodo.18403844). We now seek to validate it on real ASD cohorts.

Our objectives are to:
1. Apply the framework to SPARK and SSC exome data to identify molecular subtypes
2. Test cross-cohort reproducibility of identified subtypes
3. Correlate subtypes with clinical phenotypes (IQ, adaptive behavior, language)
4. Release validated parameters as a community resource

This work will determine whether pathway-level analysis can identify reproducible, clinically meaningful ASD subtypes. All analysis code is open-source, and validated model parameters will be released publicly to enable other researchers to apply the same approach to their cohorts.

The framework includes built-in validation gates (negative controls, stability testing) to prevent overfitting and ensure robust findings. Results will be submitted for peer-reviewed publication with full reproducibility materials.
```

---

## Specific Aims

```
Aim 1: Validate Pathway-Based Subtype Discovery Across Cohorts

We will apply the Autism Pathway Framework to SPARK and SSC exome data independently. For each cohort, we will:
- Compute gene-level burden scores from rare damaging variants
- Aggregate burdens into pathway disruption scores across 15 biological pathways
- Identify molecular subtypes via Gaussian Mixture Model clustering
- Validate clusters using negative controls (label shuffle, random gene sets) and bootstrap stability testing

Success criterion: Subtypes identified in SSC replicate in SPARK with Adjusted Rand Index > 0.7.


Aim 2: Characterize Clinical Features of Molecular Subtypes

We will correlate identified subtypes with available phenotype data:
- Cognitive measures (IQ, verbal/nonverbal)
- Adaptive behavior (Vineland scales)
- Language development milestones
- Medical comorbidities (seizures, GI issues)
- ASD severity (ADI-R, ADOS scores)

We will test for significant associations using appropriate statistical tests with multiple testing correction.


Aim 3: Release Validated Framework as Community Resource

Upon successful validation, we will:
- Publish peer-reviewed manuscript describing methods and findings
- Release validated model parameters for SPARK/SSC
- Provide documentation for applying the framework to new cohorts
- Engage with other research groups for independent validation
```

---

## Data Requested

```
From SPARK:
- Exome sequencing variant calls (VCF format)
- Core phenotype data: diagnosis, age, sex
- Cognitive assessments: IQ measures
- Behavioral assessments: Vineland Adaptive Behavior Scales
- Medical history: seizure history, GI symptoms
- Sample size: All available ASD probands with exome data

From SSC:
- Exome sequencing variant calls (VCF format)
- Core phenotype data: diagnosis, age, sex
- Cognitive assessments: verbal IQ, nonverbal IQ
- Behavioral assessments: Vineland, ADI-R, ADOS
- Medical history: available comorbidity data
- Sample size: All available trios/quads with exome data

We do not require:
- Raw sequencing reads (BAM/FASTQ)
- Audio/video recordings
- Identifiable information beyond coded IDs
```

---

## Analysis Plan

```
Data Processing:
1. Quality control: Filter variants by depth (>10x), genotype quality (>20), allele balance
2. Variant annotation: Annotate with VEP for gene symbols, consequence, CADD scores
3. Variant filtering: Retain rare (MAF < 0.1%) damaging variants (LoF + missense CADD>25)

Pathway Analysis:
4. Gene burden: Calculate per-gene burden scores weighted by variant consequence
5. Pathway aggregation: Aggregate gene burdens across 15 curated ASD-relevant pathways
6. Normalization: Z-score normalize pathway scores across samples

Subtype Discovery:
7. Clustering: Apply Gaussian Mixture Model with BIC-based model selection (k=2-8)
8. Validation gates:
   - Negative control 1: Label shuffle test (expect ARI < 0.15)
   - Negative control 2: Random gene sets (expect ARI < 0.15)
   - Stability test: Bootstrap resampling (expect ARI > 0.8)

Cross-Cohort Validation:
9. Train on SSC, predict on SPARK (and vice versa)
10. Measure cross-cohort ARI and cluster stability

Clinical Correlation:
11. ANOVA/chi-square tests for subtype × phenotype associations
12. Bonferroni correction for multiple testing
13. Effect size reporting (Cohen's d, odds ratios)

All analysis will be performed using the Autism Pathway Framework:
- GitHub: https://github.com/topmist-admin/autism-pathway-framework
- DOI: 10.5281/zenodo.18403844
```

---

## Timeline

```
Months 1-2: Data access, quality control, variant annotation
Months 3-4: SSC analysis - pathway scoring and subtype discovery
Months 5-6: SPARK analysis - pathway scoring and subtype discovery
Months 7-8: Cross-cohort validation and stability testing
Months 9-10: Clinical phenotype correlation analysis
Months 11-12: Manuscript preparation and framework release

Total duration: 12 months
```

---

## Data Security Plan

```
All data will be:
- Stored on encrypted local storage or SFARI-approved cloud environment
- Accessed only by the PI (Rohit Chauhan)
- Not shared with third parties
- Deleted upon project completion or as required by SFARI policies

No individual-level data will be published. Only aggregate statistics and subtype-level summaries will be reported in publications.

The analysis code is open-source, but trained model parameters will not enable re-identification of individuals.
```

---

## Expected Outcomes and Dissemination

```
Expected Outcomes:
1. Identification of reproducible molecular subtypes in ASD
2. Characterization of clinical features associated with each subtype
3. Validated open-source framework for pathway-based subtype analysis

Dissemination Plan:
1. Peer-reviewed publication (target: Molecular Autism, Nature Communications)
2. Preprint on medRxiv/bioRxiv upon manuscript submission
3. Updated framework release with validated parameters (GitHub + Zenodo DOI)
4. Presentation at SFARI annual meeting (if applicable)
5. Conference presentation at ASHG or IMFAR

All publications will:
- Acknowledge SFARI and SPARK/SSC participants
- Reference SFARI data access policies
- Include data availability statement per SFARI requirements
```

---

## Relevant Publications and Resources

```
Framework:
- GitHub: https://github.com/topmist-admin/autism-pathway-framework
- DOI: 10.5281/zenodo.18403844
- Preprint: https://doi.org/10.13140/RG.2.2.25221.41441

Relevant Background (by others):
- Satterstrom et al. (2020). Large-scale exome sequencing study implicates both developmental and functional changes in the neurobiology of autism. Cell.
- De Rubeis et al. (2014). Synaptic, transcriptional and chromatin genes disrupted in autism. Nature.
```

---

## Acknowledgment Statement (for publications)

```
We thank the SPARK and SSC participants and their families for their contributions to autism research. Data used in this study were obtained from the Simons Foundation Autism Research Initiative (SFARI) Base.
```

---

## Notes for Application

- Submit at: https://base.sfari.org
- Approval typically takes 2-4 weeks
- You may be asked clarifying questions
- Having a DOI and published code strengthens the application
