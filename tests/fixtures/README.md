# Test Fixtures

This directory contains sample data files for testing the Autism Pathway Framework.

## Files

### Input Data

| File | Format | Description |
|------|--------|-------------|
| `sample_variants.vcf` | VCF 4.2 | 6 variants across 3 samples on chr1, chr2, chr22 |
| `sample_pathways.gmt` | GMT | 8 pathway definitions (GO, KEGG, Reactome, custom) |
| `sample_constraints.tsv` | TSV | gnomAD constraint scores for 15 ASD-relevant genes |
| `sample_sfari_genes.csv` | CSV | SFARI gene annotations (18 genes, scores 1-3) |

### Output Data (Expected Results)

| File | Format | Description |
|------|--------|-------------|
| `sample_gene_burdens.json` | JSON | Gene burden scores for 3 samples |
| `sample_pathway_scores.json` | JSON | Pathway disruption scores for 3 samples |
| `sample_cluster_assignments.json` | JSON | Subtype assignments with 3 clusters |

## Sample IDs

- `SAMPLE_001`: Synaptic-chromatin profile (SHANK3, NRXN1, CHD8)
- `SAMPLE_002`: Glutamatergic profile (SYNGAP1, GRIN2B)
- `SAMPLE_003`: Mixed profile (ARID1B, SCN2A, SHANK3)

## Genes Included

### High-Confidence ASD Genes (SFARI Score 1)
- SHANK3, CHD8, ADNP, SYNGAP1, ARID1B, SCN2A, DYRK1A, ANK2

### Strong Candidate Genes (SFARI Score 2)
- NRXN1, GRIN2B, PTEN, TBR1, FOXP1, CNTNAP2, SETD5

### Suggestive Evidence (SFARI Score 3)
- KMT2C, POGZ, ASH1L

## Pathways Included

- `GO:0007268` - Synaptic transmission
- `GO:0007399` - Nervous system development
- `GO:0006351` - Chromatin remodeling
- `GO:0030182` - Neuron differentiation
- `REACT:12345` - Neurotransmitter release
- `KEGG:04724` - Glutamatergic synapse
- `KEGG:04727` - GABAergic synapse
- `CUSTOM:ASD_HIGH` - SFARI category 1 genes

## Usage

```python
import json
from pathlib import Path

fixtures = Path(__file__).parent / "fixtures"

# Load gene burdens
with open(fixtures / "sample_gene_burdens.json") as f:
    gene_burdens = json.load(f)

# Load pathway scores
with open(fixtures / "sample_pathway_scores.json") as f:
    pathway_scores = json.load(f)

# Load cluster assignments
with open(fixtures / "sample_cluster_assignments.json") as f:
    clusters = json.load(f)
```

## Pytest Fixtures

These files support the pytest fixtures defined in `conftest.py`:

```python
@pytest.fixture
def sample_gene_burdens():
    """Load sample gene burdens from fixture file."""
    with open(FIXTURES_DIR / "sample_gene_burdens.json") as f:
        return json.load(f)

@pytest.fixture
def sample_pathway_definitions():
    """Load sample pathway definitions from GMT file."""
    return load_gmt(FIXTURES_DIR / "sample_pathways.gmt")
```

## Notes

- All data is synthetic and for testing purposes only
- Variant positions are arbitrary and may not correspond to real variants
- Scores are designed to produce meaningful test scenarios
- The fixture data is intentionally small to enable fast test execution
