# Module 02: Variant Processing

## Overview

This module provides variant processing utilities including:

- **Quality Control (QC) Filters**: Filter variants and samples based on quality metrics
- **Functional Annotation**: Annotate variants with functional impact predictions
- **Gene Burden Calculation**: Aggregate variants to gene-level burden scores

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
```

Requires Module 01 (data loaders) for data structures.

## Interface Contract

### QC Filters

```python
@dataclass
class QCConfig:
    min_quality: float = 20.0
    min_depth: int = 10
    max_missing_rate: float = 0.1
    min_allele_freq: float = 0.0
    max_allele_freq: float = 0.01  # Rare variants only
    filter_pass_only: bool = True

class QCFilter:
    def filter_variants(self, dataset: VariantDataset, config: QCConfig) -> VariantDataset
    def filter_samples(self, dataset: VariantDataset, config: QCConfig) -> VariantDataset
    def get_qc_report(self) -> QCReport
```

### Functional Annotation

```python
class VariantAnnotator:
    def annotate(self, variants: List[Variant], gene_db: GeneAnnotationDB) -> List[AnnotatedVariant]
    def classify_impact(self, variant: AnnotatedVariant) -> str  # HIGH, MODERATE, LOW, MODIFIER

@dataclass
class AnnotatedVariant:
    variant: Variant
    gene_id: str
    consequence: str  # e.g., "missense", "nonsense", "frameshift"
    impact: str  # HIGH, MODERATE, LOW, MODIFIER
    cadd_score: Optional[float]
    revel_score: Optional[float]
```

### Gene Burden

```python
@dataclass
class WeightConfig:
    consequence_weights: Dict[str, float]  # e.g., {"nonsense": 1.0, "missense": 0.5}
    cadd_threshold: float = 20.0
    include_synonymous: bool = False

class GeneBurdenCalculator:
    def compute(self, dataset: VariantDataset, weights: WeightConfig) -> GeneBurdenMatrix

@dataclass
class GeneBurdenMatrix:
    samples: List[str]
    genes: List[str]
    scores: np.ndarray  # shape: (n_samples, n_genes)

    def to_sparse(self) -> scipy.sparse.csr_matrix
    def get_sample(self, sample_id: str) -> Dict[str, float]
    def get_gene(self, gene_id: str) -> np.ndarray
```

## Usage Examples

### Filtering Variants

```python
from modules.02_variant_processing import QCFilter, QCConfig

config = QCConfig(
    min_quality=30,
    max_allele_freq=0.001,  # Very rare variants only
)

filter = QCFilter()
filtered_dataset = filter.filter_variants(dataset, config)
report = filter.get_qc_report()
print(f"Retained {len(filtered_dataset)} of {len(dataset)} variants")
```

### Computing Gene Burdens

```python
from modules.02_variant_processing import GeneBurdenCalculator, WeightConfig

weights = WeightConfig(
    consequence_weights={
        "nonsense": 1.0,
        "frameshift": 1.0,
        "splice": 0.8,
        "missense": 0.5,
    },
    cadd_threshold=25.0,
)

calculator = GeneBurdenCalculator()
burden_matrix = calculator.compute(annotated_variants, weights)

# Get burden for a specific sample
sample_burden = burden_matrix.get_sample("Sample1")
```

## Testing

```bash
python -m pytest modules/02_variant_processing/tests/ -v
```

## File Structure

```
modules/02_variant_processing/
├── README.md
├── __init__.py
├── qc_filters.py
├── annotation.py
├── gene_burden.py
└── tests/
    ├── __init__.py
    ├── test_qc_filters.py
    ├── test_annotation.py
    └── test_gene_burden.py
```
