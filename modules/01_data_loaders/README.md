# Module 01: Data Loaders

## Overview

This module provides standardized data loading utilities for the autism genetics analysis pipeline. It handles loading and parsing of:

- **VCF files**: Variant Call Format files containing genetic variants
- **Pathway databases**: Gene Ontology (GO), Reactome, KEGG pathways
- **Expression data**: BrainSpan developmental expression atlas
- **Single-cell data**: Allen Brain and other cortical single-cell atlases
- **Gene constraints**: gnomAD constraint scores (pLI, LOEUF) and SFARI gene annotations

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
pysam>=0.19.0  # For VCF parsing
anndata>=0.8.0  # For single-cell data (h5ad format)
```

## Interface Contract

### Core Data Structures

```python
@dataclass
class Variant:
    """Single genetic variant for one sample."""
    chrom: str              # Chromosome (e.g., "chr1", "1")
    pos: int                # 1-based position
    ref: str                # Reference allele
    alt: str                # Alternate allele
    sample_id: str          # Sample identifier
    genotype: str           # Genotype call (e.g., "0/1", "1/1")
    quality: float          # Variant quality score (QUAL)
    filter_status: str      # Filter status ("PASS" or filter name)
    info: Dict[str, Any]    # INFO field key-value pairs
    variant_id: Optional[str] = None  # Variant ID (rsID)

    # Computed properties
    @property
    def variant_type(self) -> str:
        """Determine variant type: SNV, insertion, deletion, MNV"""
        ...

    @property
    def is_snv(self) -> bool:
        """Check if single nucleotide variant"""
        ...

@dataclass
class VariantDataset:
    """Collection of variants with associated sample information."""
    variants: List[Variant]     # All variants in the dataset
    samples: List[str]          # Unique sample identifiers
    metadata: Dict[str, Any]    # Dataset metadata (source, date, etc.)

@dataclass
class PathwayDatabase:
    pathways: Dict[str, Set[str]]  # pathway_id -> gene_ids
    pathway_names: Dict[str, str]  # pathway_id -> name
    gene_to_pathways: Dict[str, Set[str]]  # gene_id -> pathway_ids
    source: str  # "GO", "Reactome", "KEGG"

@dataclass
class DevelopmentalExpression:
    genes: List[str]
    stages: List[str]
    regions: List[str]
    expression: np.ndarray  # (n_genes, n_stages, n_regions)

@dataclass
class SingleCellAtlas:
    genes: List[str]
    cell_types: List[str]
    expression: np.ndarray  # (n_genes, n_cell_types)
    cell_type_hierarchy: Dict[str, List[str]]

@dataclass
class GeneConstraints:
    gene_ids: List[str]
    pli_scores: Dict[str, float]
    loeuf_scores: Dict[str, float]
    mis_z_scores: Dict[str, float]

@dataclass
class SFARIGenes:
    gene_ids: List[str]
    scores: Dict[str, int]  # 1=high confidence, 2=strong, 3=suggestive
    syndromic: Dict[str, bool]
    evidence: Dict[str, List[str]]
```

### Loaders

```python
class VCFLoader:
    def load(self, vcf_path: str) -> VariantDataset
    def validate(self, dataset: VariantDataset) -> ValidationReport

class PathwayLoader:
    def load_go(self, obo_path: str, gaf_path: str) -> PathwayDatabase
    def load_reactome(self, gmt_path: str) -> PathwayDatabase
    def load_kegg(self, gmt_path: str) -> PathwayDatabase
    def merge(self, databases: List[PathwayDatabase]) -> PathwayDatabase

class ExpressionLoader:
    def load_brainspan(self, data_dir: str) -> DevelopmentalExpression
    def get_expression_by_stage(self, gene_id: str, stage: str) -> float
    def get_prenatal_expressed_genes(self, threshold: float = 1.0) -> List[str]

class SingleCellLoader:
    def load_allen_brain(self, h5ad_path: str) -> SingleCellAtlas
    def get_cell_type_markers(self, cell_type: str) -> List[str]
    def get_expression_by_cell_type(self, gene_id: str) -> Dict[str, float]

class ConstraintLoader:
    def load_gnomad_constraints(self, tsv_path: str) -> GeneConstraints
    def load_sfari_genes(self, csv_path: str) -> SFARIGenes
    def get_constrained_genes(self, pli_threshold: float = 0.9) -> List[str]
```

## Usage Examples

### Loading VCF Data

```python
from modules.01_data_loaders import VCFLoader

loader = VCFLoader()
dataset = loader.load("path/to/variants.vcf.gz")
report = loader.validate(dataset)

print(f"Loaded {len(dataset.variants)} variants from {len(dataset.samples)} samples")
```

### Loading Pathway Databases

```python
from modules.01_data_loaders import PathwayLoader

loader = PathwayLoader()
go_db = loader.load_go("go.obo", "goa_human.gaf")
reactome_db = loader.load_reactome("ReactomePathways.gmt")

# Merge databases
combined = loader.merge([go_db, reactome_db])
print(f"Total pathways: {len(combined.pathways)}")
```

### Loading BrainSpan Expression

```python
from modules.01_data_loaders import ExpressionLoader

loader = ExpressionLoader()
expr = loader.load_brainspan("brainspan_data/")

# Get prenatal expressed genes
prenatal_genes = loader.get_prenatal_expressed_genes(threshold=1.0)
print(f"Genes expressed prenatally: {len(prenatal_genes)}")
```

### Loading Gene Constraints

```python
from modules.01_data_loaders import ConstraintLoader

loader = ConstraintLoader()
constraints = loader.load_gnomad_constraints("gnomad.v2.1.1.lof_metrics.txt")
sfari = loader.load_sfari_genes("SFARI-Gene_genes.csv")

# Get highly constrained genes
constrained = loader.get_constrained_genes(pli_threshold=0.9)
print(f"Highly constrained genes: {len(constrained)}")
```

## Testing

```bash
python -m pytest modules/01_data_loaders/tests/ -v
```

## File Structure

```
modules/01_data_loaders/
├── README.md
├── __init__.py
├── vcf_loader.py
├── pathway_loader.py
├── expression_loader.py
├── single_cell_loader.py
├── constraint_loader.py
├── annotation_loader.py
└── tests/
    ├── __init__.py
    ├── test_vcf_loader.py
    ├── test_pathway_loader.py
    ├── test_expression_loader.py
    └── test_constraint_loader.py
```
