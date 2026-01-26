# Demo Data

This folder contains synthetic demonstration data for the Autism Pathway Framework.

## Purpose

- Enable reproducible end-to-end pipeline testing
- Provide a "golden path" for new users (≤60 minutes)
- Validate outputs without requiring sensitive genomic data

## Contents (To Be Added in Week 3)

| File | Description | Format |
|------|-------------|--------|
| `demo_variants.vcf` | Synthetic variant calls | VCF 4.2 |
| `demo_phenotypes.csv` | Sample metadata | CSV |
| `demo_pathways.gmt` | Pathway definitions | GMT |
| `expected_outputs/` | Golden outputs for validation | Various |

## Data Characteristics

- **Samples:** ~50 synthetic individuals
- **Variants:** ~1,000 synthetic variants
- **Pathways:** Subset of Reactome/GO terms
- **Design:** Includes planted signal for validation

## Privacy Notice

This is entirely synthetic data generated for demonstration purposes.
It does not contain any real patient information.

## Usage

```bash
# Run demo pipeline with this data
python -m autism_pathway_framework --config configs/demo.yaml
```

## Generation Script

The synthetic data will be generated using:
```bash
python scripts/generate_demo_data.py --seed 42 --output examples/demo_data/
```

---

**Status:** Placeholder - Data to be generated in Week 3
