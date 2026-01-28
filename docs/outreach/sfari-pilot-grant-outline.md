# SFARI Pilot Award - Grant Outline

**Program:** SFARI Pilot Award
**URL:** https://www.sfari.org/funding-opportunities/pilot-awards/
**Amount:** Up to $300,000 over 2 years
**Deadline:** Rolling (check website)

---

## Project Title

**Validation of Pathway-Based Molecular Subtyping in Autism Spectrum Disorder Across Independent Cohorts**

---

## Specific Aims

### Aim 1: Validate Pathway-Based Subtype Discovery in SPARK/SSC Cohorts

**Hypothesis:** Pathway-level disruption scores will identify reproducible molecular subtypes across independent ASD cohorts.

**Approach:**
- Apply the Autism Pathway Framework to SPARK (N>50,000) and SSC (N~2,600) exome data
- Compare subtype assignments across cohorts using Adjusted Rand Index
- Test stability via bootstrap resampling and cross-validation

**Expected Outcome:** ARI > 0.7 between cohorts, demonstrating reproducible subtypes.

### Aim 2: Characterize Clinical and Biological Features of Molecular Subtypes

**Hypothesis:** Identified subtypes will show distinct clinical phenotypes and biological pathway enrichments.

**Approach:**
- Correlate subtypes with clinical features (IQ, adaptive behavior, language, seizures)
- Perform pathway enrichment analysis within each subtype
- Validate enrichments against published functional genomics data (BrainSpan, PsychENCODE)

**Expected Outcome:** Clinically meaningful subtype definitions with biological interpretability.

### Aim 3: Release Validated Framework as Community Resource

**Hypothesis:** An open, validated tool will accelerate ASD subtype research across the field.

**Approach:**
- Release validated model weights and cohort-specific parameters
- Publish detailed protocol for applying to new cohorts
- Engage with 3+ external groups for independent validation

**Expected Outcome:** Peer-reviewed publication + widely-used open-source tool.

---

## Research Strategy

### Significance

Autism Spectrum Disorder affects 1 in 36 children and is genetically heterogeneous, with hundreds of implicated genes. Despite large-scale sequencing efforts, translating genetic findings to clinical subtypes remains challenging. Gene-level analyses often fail to replicate across cohorts due to:

1. **Allelic heterogeneity** - Different variants in the same gene
2. **Locus heterogeneity** - Different genes, same pathway
3. **Small effect sizes** - Individual variants contribute modestly

**Our approach** aggregates rare variant burden at the **pathway level**, capturing convergent biology rather than individual gene effects. Preliminary results on synthetic data demonstrate:

- Successful recovery of planted subtypes (ARI > 0.9)
- Robust negative controls (label shuffle, random gene sets)
- Stable clustering under bootstrap resampling

### Innovation

| Traditional Approach | Our Approach |
|---------------------|--------------|
| Gene-level burden tests | Pathway-level disruption scores |
| Supervised classification | Unsupervised subtype discovery |
| Single-cohort analysis | Cross-cohort validation |
| Black-box methods | Built-in validation gates |

**Key innovations:**
1. **Pathway aggregation** reduces dimensionality while preserving biological signal
2. **Validation gates** prevent overfitting and spurious clusters
3. **Open-source framework** enables community adoption and extension

### Approach

#### Aim 1: Cross-Cohort Validation (Months 1-12)

**Data:**
- SPARK: >50,000 families, exome sequencing (via SFARI Base)
- SSC: 2,600 families, exome sequencing (via SFARI Base)

**Analysis Pipeline:**
1. Quality control and variant annotation
2. Gene burden calculation (LoF + damaging missense)
3. Pathway score aggregation (15 curated pathways)
4. GMM clustering with BIC-based model selection
5. Validation gates (negative controls + stability)

**Cross-Cohort Comparison:**
- Train on SSC, validate on SPARK (and vice versa)
- Measure ARI, silhouette score, cluster stability
- Success criterion: ARI > 0.7 between cohorts

#### Aim 2: Clinical Characterization (Months 6-18)

**Phenotype Data:**
- Vineland Adaptive Behavior Scales
- IQ measures (verbal, nonverbal)
- Language milestones
- Medical comorbidities (seizures, GI, sleep)
- ADI-R / ADOS scores

**Analysis:**
- ANOVA / chi-square for subtype × phenotype associations
- Bonferroni correction for multiple testing
- Effect size reporting (Cohen's d, odds ratios)

**Biological Validation:**
- Gene Ontology enrichment within subtypes
- Overlap with BrainSpan developmental expression
- Comparison to published ASD gene modules

#### Aim 3: Community Release (Months 12-24)

**Deliverables:**
1. Peer-reviewed publication (target: Nature Communications, Molecular Autism)
2. Updated framework with validated parameters
3. SFARI-specific documentation and tutorials
4. Engagement with 3+ external validation groups

---

## Timeline

| Months | Milestone |
|--------|-----------|
| 1-3 | SFARI data access, QC pipeline |
| 4-6 | SSC analysis, initial subtype discovery |
| 7-9 | SPARK analysis, cross-cohort validation |
| 10-12 | Clinical phenotype correlation |
| 13-15 | Biological pathway validation |
| 16-18 | External collaborator validation |
| 19-21 | Manuscript preparation |
| 22-24 | Publication, community release |

---

## Budget Justification

### Year 1: $150,000

| Category | Amount | Description |
|----------|--------|-------------|
| Personnel | $100,000 | PI salary (50% effort) |
| Computing | $30,000 | Cloud compute for large cohorts |
| Travel | $10,000 | SFARI meetings, ASHG conference |
| Other | $10,000 | Publication fees, supplies |

### Year 2: $150,000

| Category | Amount | Description |
|----------|--------|-------------|
| Personnel | $100,000 | PI salary (50% effort) |
| Computing | $20,000 | Continued analysis |
| Subcontract | $20,000 | External validation collaborator |
| Travel | $5,000 | Conference presentations |
| Other | $5,000 | Publication fees |

---

## Preliminary Data

### Framework Validation (Synthetic Data)

| Metric | Value | Threshold |
|--------|-------|-----------|
| Ground truth ARI | 0.92 | > 0.7 |
| Label shuffle ARI | 0.03 | < 0.15 |
| Random genes ARI | 0.08 | < 0.15 |
| Bootstrap stability | 0.89 | > 0.8 |

### Published Framework

- **GitHub:** https://github.com/topmist-admin/autism-pathway-framework
- **DOI:** 10.5281/zenodo.18403844
- **Preprint:** 10.13140/RG.2.2.25221.41441

---

## Alignment with SFARI Priorities

This project directly addresses SFARI's mission to improve understanding of autism by:

1. **Leveraging SFARI data** - Designed for SPARK/SSC cohorts
2. **Enabling reproducibility** - Open-source, validated framework
3. **Bridging genetics to phenotype** - Clinically meaningful subtypes
4. **Community resource** - Freely available for all researchers

---

## References

1. Satterstrom FK, et al. (2020). Large-scale exome sequencing study implicates both developmental and functional changes in the neurobiology of autism. *Cell*.

2. Geschwind DH, State MW. (2015). Gene hunting in autism spectrum disorder. *Lancet*.

3. Voineagu I, et al. (2011). Transcriptomic analysis of autistic brain reveals convergent molecular pathology. *Nature*.

4. De Rubeis S, et al. (2014). Synaptic, transcriptional and chromatin genes disrupted in autism. *Nature*.

---

## Contact

**Rohit Chauhan**
Email: info@topmist.com
GitHub: https://github.com/topmist-admin
