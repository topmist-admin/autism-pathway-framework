# Terminology and Glossary

> This document defines key terms used throughout the Autism Pathway Framework to ensure consistent usage across all documentation and code.

## Genetic Concepts

### Variant
A genetic difference from the reference genome at a specific position. In this framework, we focus on **rare variants** (allele frequency < 1%) that may contribute to ASD risk.

### Consequence
The predicted functional effect of a variant on a gene. Common consequences include:
- **Frameshift**: Insertion or deletion that shifts the reading frame
- **Stop-gained (nonsense)**: Creates a premature stop codon
- **Missense**: Changes one amino acid to another
- **Synonymous**: No change to amino acid sequence
- **Splice site**: Affects mRNA splicing

### Loss-of-Function (LoF)
Variants that are predicted to severely disrupt gene function. Includes:
- Frameshift variants
- Stop-gained variants
- Splice donor/acceptor variants
- Start-lost variants

LoF variants are typically given the highest weight in burden calculations.

### Allele Frequency (AF)
The proportion of chromosomes in a population carrying a specific allele. Key thresholds:
- **Ultra-rare**: AF < 0.0001 (< 0.01%)
- **Rare**: AF < 0.01 (< 1%)
- **Common**: AF ≥ 0.01 (≥ 1%)

### gnomAD
The Genome Aggregation Database - a reference database of genetic variation from ~140,000 individuals. Used for:
- Allele frequency filtering
- Gene constraint scores

## Scoring Concepts

### Gene Burden (Burden Score)
A numerical score representing the cumulative impact of variants in a gene for a single individual. Higher burden = more predicted disruption.

**Calculation**: Sum of weighted variant contributions
```
burden(gene, sample) = Σ weight(variant)
```

Where `weight(variant)` considers:
- Consequence type (LoF > missense > other)
- Pathogenicity scores (CADD, REVEL)
- Allele frequency (rarer = higher weight)

**Synonyms**: Gene-level score, disruption score

### CADD Score
Combined Annotation Dependent Depletion - a pathogenicity predictor that integrates multiple annotations. Scores are typically presented as **Phred-scaled**:
- CADD ≥ 10: Top 10% most deleterious
- CADD ≥ 20: Top 1% most deleterious
- CADD ≥ 30: Top 0.1% most deleterious

Used in this framework to weight missense variants.

### REVEL Score
Rare Exome Variant Ensemble Learner - an ensemble predictor for missense variant pathogenicity.
- Score range: 0-1
- REVEL ≥ 0.5: Likely pathogenic
- REVEL ≥ 0.75: High confidence pathogenic

### Pathway Score (Pathway Disruption Score)
A numerical score representing the aggregate genetic burden across all genes in a biological pathway for a single individual.

**Calculation**: Aggregate of gene burdens with size normalization
```
pathway_score = Σ burden(gene) / √(pathway_size)
```

**Synonyms**: Pathway disruption, pathway perturbation

### Z-Score
A standardized score indicating how many standard deviations a value is from the population mean.
```
z-score = (value - mean) / std
```

Used to normalize pathway scores across the cohort for comparability.

## Constraint Concepts

### Gene Constraint
A measure of how intolerant a gene is to mutations, inferred from the deficit of observed variants in gnomAD compared to expectation.

### pLI Score
Probability of being Loss-of-function Intolerant. Ranges 0-1.
- pLI ≥ 0.9: Highly constrained (LoF variants likely pathogenic)
- pLI < 0.1: Tolerant to LoF

### LOEUF Score
Loss-of-function Observed/Expected Upper bound Fraction. Lower = more constrained.
- LOEUF < 0.35: Highly constrained
- LOEUF > 1.0: Tolerant

### mis_z Score
Z-score for missense constraint. Higher = more constrained against missense variants.

## Pathway Concepts

### Pathway
A set of genes that function together in a biological process. Sources include:
- **GO (Gene Ontology)**: Hierarchical ontology of biological processes, cellular components, molecular functions
- **Reactome**: Curated pathway database with detailed molecular interactions
- **KEGG**: Pathway maps including metabolism, signaling, disease

### Pathway Size
The number of genes annotated to a pathway. We typically filter:
- Minimum size: 5 genes (too small = unstable)
- Maximum size: 500 genes (too large = uninformative)

### Network Propagation
A technique to spread gene-level signals through a gene-gene interaction network. Captures indirect effects through protein-protein interactions.

Uses Random Walk with Restart (RWR):
```
p(t+1) = (1-α) × W × p(t) + α × p(0)
```
Where α is the restart probability.

## Clustering Concepts

### Subtype
A genetically coherent subgroup of individuals discovered through unsupervised clustering of pathway profiles. Subtypes represent **hypotheses** about distinct etiological mechanisms.

### GMM (Gaussian Mixture Model)
A probabilistic clustering method that models data as a mixture of Gaussian distributions. Provides soft (probabilistic) cluster assignments.

### Stability
The consistency of clustering results under resampling (bootstrap). Measured as the proportion of sample pairs that are consistently co-clustered. Target: stability > 0.7.

### BIC (Bayesian Information Criterion)
A model selection criterion balancing fit and complexity. Lower BIC = better model. Used to select optimal number of clusters.

## Reasoning Concepts

### Neuro-symbolic
An approach combining neural networks (statistical learning) with symbolic reasoning (logical rules). Enables interpretable predictions grounded in biological knowledge.

### Biological Rules
Logical constraints derived from domain knowledge. Examples:
- "Synaptic genes are more relevant than housekeeping genes for ASD"
- "Prenatal brain expression suggests neurodevelopmental relevance"
- "High constraint (pLI > 0.9) implies functional importance"

### Causal Inference
Methods for reasoning about cause-and-effect relationships, not just correlations. Includes:
- **SCM**: Structural Causal Model - directed graph of causal relationships
- **do-calculus**: Formal rules for interventional queries
- **Counterfactuals**: "What if" reasoning about alternative scenarios

## Data Format Terms

### VCF
Variant Call Format - standard file format for genetic variants. Contains:
- Chromosome, position, reference/alternate alleles
- Quality and filter status
- Per-sample genotype information

### GMT
Gene Matrix Transposed - simple format for pathway definitions:
```
PATHWAY_NAME<tab>DESCRIPTION<tab>GENE1<tab>GENE2<tab>...
```

### h5ad
HDF5-based format for annotated data matrices, commonly used for single-cell RNA-seq data (via AnnData library).

## Abbreviations

| Abbreviation | Full Term |
|--------------|-----------|
| AF | Allele Frequency |
| ASD | Autism Spectrum Disorder |
| BIC | Bayesian Information Criterion |
| CADD | Combined Annotation Dependent Depletion |
| GNN | Graph Neural Network |
| GMM | Gaussian Mixture Model |
| GO | Gene Ontology |
| KEGG | Kyoto Encyclopedia of Genes and Genomes |
| LoF | Loss of Function |
| LOEUF | Loss-of-function Observed/Expected Upper bound Fraction |
| PPI | Protein-Protein Interaction |
| QC | Quality Control |
| REVEL | Rare Exome Variant Ensemble Learner |
| RWR | Random Walk with Restart |
| SCM | Structural Causal Model |
| SFARI | Simons Foundation Autism Research Initiative |
| SNV | Single Nucleotide Variant |
| VCF | Variant Call Format |
| VEP | Variant Effect Predictor |

## See Also

- [Framework Overview](framework_overview.md) - High-level architecture
- [Implementation Plan](implementation_plan.md) - Module specifications
- [Data Formats](data_formats.md) - Detailed format specifications
