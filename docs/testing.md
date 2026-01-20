# Testing Guide

> This document describes how to test the Autism Pathway Framework modules, including unit tests, integration tests, and test fixtures.

## Test Structure

```
autism-pathway-framework/
├── modules/
│   ├── 01_data_loaders/
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── test_vcf_loader.py
│   │       ├── test_pathway_loader.py
│   │       └── test_constraint_loader.py
│   │
│   └── 02_variant_processing/
│       └── tests/
│           ├── __init__.py
│           ├── test_qc_filters.py
│           ├── test_annotation.py
│           └── test_gene_burden.py
│
└── tests/
    ├── __init__.py
    ├── fixtures/
    │   ├── README.md
    │   ├── sample_variants.vcf
    │   ├── sample_pathways.gmt
    │   └── sample_constraints.tsv
    │
    └── integration/
        ├── test_variant_to_burden.py
        └── test_full_pipeline.py
```

## Running Tests

### All Tests

```bash
# Run all tests
pytest

# With verbose output
pytest -v

# With coverage report
pytest --cov=modules --cov-report=html
```

### Module-Specific Tests

```bash
# Module 01: Data Loaders
pytest modules/01_data_loaders/tests/ -v

# Module 02: Variant Processing
pytest modules/02_variant_processing/tests/ -v

# Specific test file
pytest modules/01_data_loaders/tests/test_vcf_loader.py -v

# Specific test function
pytest modules/01_data_loaders/tests/test_vcf_loader.py::TestVCFLoader::test_load_basic -v
```

### Test Categories

```bash
# Run only fast tests
pytest -m "not slow"

# Run integration tests
pytest tests/integration/ -v

# Skip tests requiring external data
pytest -m "not requires_external_data"
```

## Test Fixtures

### Using pytest Fixtures

```python
import pytest
from pathlib import Path

@pytest.fixture
def sample_vcf_path(tmp_path):
    """Create a minimal VCF file for testing."""
    vcf_content = """##fileformat=VCFv4.2
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1
chr1\t100\t.\tA\tG\t50\tPASS\tDP=30\tGT\t0/1
chr1\t200\t.\tC\tT\t60\tPASS\tDP=25\tGT\t1/1
"""
    vcf_file = tmp_path / "test.vcf"
    vcf_file.write_text(vcf_content)
    return str(vcf_file)


@pytest.fixture
def sample_variants():
    """Create sample Variant objects for testing."""
    return [
        Variant(
            chrom="chr1",
            pos=100,
            ref="A",
            alt="G",
            quality=50.0,
            filter_status="PASS",
            sample_id="sample1",
            genotype="0/1",
            info={"DP": 30},
        ),
        Variant(
            chrom="chr1",
            pos=200,
            ref="C",
            alt="T",
            quality=60.0,
            filter_status="PASS",
            sample_id="sample1",
            genotype="1/1",
            info={"DP": 25},
        ),
    ]


@pytest.fixture
def sample_dataset(sample_variants):
    """Create sample VariantDataset for testing."""
    return VariantDataset(
        variants=sample_variants,
        samples=["sample1"],
        metadata={"source": "test"},
    )
```

### Shared Fixtures (conftest.py)

```python
# tests/conftest.py
import pytest
import numpy as np

@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test fixture directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_burden_matrix():
    """Create a sample GeneBurdenMatrix for testing."""
    samples = ["sample1", "sample2", "sample3"]
    genes = ["GENE_A", "GENE_B", "GENE_C"]
    scores = np.array([
        [1.0, 0.5, 0.0],
        [0.0, 1.5, 0.3],
        [0.8, 0.0, 1.2],
    ])
    return GeneBurdenMatrix(
        samples=samples,
        genes=genes,
        scores=scores,
    )
```

## Test Categories

### Unit Tests

Test individual functions and classes in isolation:

```python
class TestQCConfig:
    """Unit tests for QCConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = QCConfig()
        assert config.min_quality == 20.0
        assert config.filter_pass_only is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = QCConfig(min_quality=30.0, min_depth=20)
        assert config.min_quality == 30.0
        assert config.min_depth == 20


class TestVariantAnnotator:
    """Unit tests for VariantAnnotator."""

    def test_annotate_single(self, sample_variants):
        """Test annotating a single variant."""
        annotator = VariantAnnotator()
        annotated = annotator.annotate(sample_variants[0])

        assert isinstance(annotated, AnnotatedVariant)
        assert annotated.variant == sample_variants[0]

    def test_variant_key_generation(self):
        """Test consistent key generation."""
        annotator = VariantAnnotator()
        variant = Variant(
            chrom="chr1", pos=100, ref="A", alt="G",
            quality=50, filter_status="PASS",
            sample_id="s1", genotype="0/1"
        )
        key = annotator._variant_key(variant)
        assert key == "chr1:100:A>G"
```

### Edge Case Tests

Test boundary conditions and error handling:

```python
class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_variant_list(self):
        """Test with no variants."""
        calculator = GeneBurdenCalculator()
        matrix = calculator.compute([])
        assert matrix.n_samples == 0
        assert matrix.n_genes == 0

    def test_missing_gene_id(self, sample_variants):
        """Test variants without gene ID."""
        annotated = AnnotatedVariant(
            variant=sample_variants[0],
            gene_id=None,  # No gene
            consequence=VariantConsequence.INTERGENIC,
            impact=ImpactLevel.MODIFIER,
        )
        calculator = GeneBurdenCalculator()
        matrix = calculator.compute([annotated])
        assert matrix.n_genes == 0

    def test_zero_division_prevention(self):
        """Test normalization with zero values."""
        matrix = GeneBurdenMatrix(
            samples=["s1"],
            genes=["g1"],
            scores=np.array([[0.0]]),  # All zeros
        )
        normalized = matrix.normalize(method="zscore")
        assert not np.isnan(normalized.scores).any()
```

### Integration Tests

Test multiple components working together:

```python
# tests/integration/test_variant_to_burden.py

class TestVariantToBurdenPipeline:
    """Integration tests for variant -> burden pipeline."""

    def test_full_pipeline(self, sample_vcf_path):
        """Test complete variant to burden pipeline."""
        # Load VCF
        loader = VCFLoader()
        dataset = loader.load(sample_vcf_path)

        # QC filter
        qc = QCFilter()
        config = QCConfig(min_quality=20)
        filtered, report = qc.run_full_qc(dataset, config)

        # Annotate
        annotator = VariantAnnotator()
        annotated = annotator.annotate_batch(filtered.variants)

        # Compute burden
        calculator = GeneBurdenCalculator()
        matrix = calculator.compute(annotated)

        # Verify
        assert matrix.n_samples > 0
        assert matrix.n_genes >= 0

    def test_qc_reduces_variants(self, sample_dataset):
        """Test that QC reduces variant count."""
        qc = QCFilter()
        config = QCConfig(min_quality=100)  # High threshold
        filtered = qc.filter_variants(sample_dataset, config)

        assert len(filtered.variants) < len(sample_dataset.variants)
```

## Test Markers

Define custom markers in `pytest.ini`:

```ini
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks integration tests
    requires_external_data: marks tests requiring external files
```

Use markers in tests:

```python
@pytest.mark.slow
def test_large_vcf_loading():
    """Test loading large VCF file."""
    pass


@pytest.mark.integration
def test_full_pipeline():
    """Test complete analysis pipeline."""
    pass


@pytest.mark.requires_external_data
def test_with_real_gnomad():
    """Test with real gnomAD data."""
    pass
```

## Test Data Generation

### Generate Synthetic VCF

```python
def generate_test_vcf(n_variants=100, n_samples=10, output_path="test.vcf"):
    """Generate a synthetic VCF file for testing."""
    import random

    header = """##fileformat=VCFv4.2
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
"""

    samples = [f"SAMPLE_{i:03d}" for i in range(n_samples)]
    header += "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
    header += "\t".join(samples) + "\n"

    bases = "ACGT"
    chroms = [f"chr{i}" for i in range(1, 23)]

    lines = [header]
    for i in range(n_variants):
        chrom = random.choice(chroms)
        pos = random.randint(1000, 1000000)
        ref = random.choice(bases)
        alt = random.choice([b for b in bases if b != ref])
        qual = random.randint(20, 100)
        dp = random.randint(10, 100)
        af = round(random.uniform(0.0001, 0.01), 6)

        info = f"DP={dp};AF={af}"
        fmt = "GT:DP"

        gts = []
        for _ in range(n_samples):
            gt = random.choice(["0/0", "0/1", "1/1"])
            sample_dp = random.randint(5, 50)
            gts.append(f"{gt}:{sample_dp}")

        line = f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t{qual}\tPASS\t{info}\t{fmt}\t"
        line += "\t".join(gts)
        lines.append(line)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path
```

### Generate Synthetic Burden Matrix

```python
def generate_test_burden_matrix(n_samples=100, n_genes=1000, sparsity=0.95):
    """Generate a sparse burden matrix for testing."""
    import numpy as np

    # Most genes have zero burden (sparse)
    scores = np.zeros((n_samples, n_genes))

    # Add some non-zero values
    n_nonzero = int(n_samples * n_genes * (1 - sparsity))
    indices = np.random.choice(n_samples * n_genes, n_nonzero, replace=False)

    for idx in indices:
        i, j = divmod(idx, n_genes)
        scores[i, j] = np.random.exponential(0.5)  # Exponential distribution

    samples = [f"SAMPLE_{i:03d}" for i in range(n_samples)]
    genes = [f"GENE_{i:05d}" for i in range(n_genes)]

    return GeneBurdenMatrix(samples=samples, genes=genes, scores=scores)
```

## Code Coverage

### Configure Coverage

```ini
# pyproject.toml
[tool.coverage.run]
source = ["modules"]
omit = ["*/tests/*", "*/__init__.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
]
```

### Generate Reports

```bash
# Terminal report
pytest --cov=modules --cov-report=term-missing

# HTML report
pytest --cov=modules --cov-report=html
open htmlcov/index.html

# XML for CI
pytest --cov=modules --cov-report=xml
```

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run tests
        run: |
          pytest --cov=modules --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## See Also

- [Configuration Guide](configuration.md) - Test configurations
- [API Reference](api_reference.md) - Module documentation
- [Data Formats](data_formats.md) - Test file formats
