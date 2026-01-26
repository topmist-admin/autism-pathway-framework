# Reproducibility Guide

This document describes the deterministic seed strategy and reproducibility guarantees for the Autism Pathway Framework.

## Seed Strategy

### Overview

All stochastic operations in the framework are controlled by a single master seed, ensuring fully reproducible results across runs.

### Configuration

Set the master seed in your config file:

```yaml
pipeline:
  seed: 42  # Any integer; null = random (non-reproducible)
```

**Demo runs**: Always use `seed: 42` for the demo pipeline.

### Seed Propagation

The master seed propagates to all randomized components:

| Component | Seeding Method | Notes |
|-----------|----------------|-------|
| Python random | `random.seed(seed)` | Standard library |
| NumPy | `np.random.seed(seed)` | Array operations |
| PyTorch CPU | `torch.manual_seed(seed)` | Tensor operations |
| PyTorch CUDA | `torch.cuda.manual_seed_all(seed)` | GPU operations |
| Scikit-learn | `random_state=seed` | Estimators |
| GMM Clustering | `GaussianMixture(random_state=seed)` | Module 08 |
| Bootstrap sampling | `np.random.RandomState(seed)` | Stability tests |
| Train/test splits | `train_test_split(random_state=seed)` | Validation |

### Implementation Pattern

All modules should follow this pattern:

```python
from autism_pathway_framework.utils.seed import set_global_seed, get_rng

def run_analysis(data, config):
    # Set seed at entry point
    seed = config.get('pipeline', {}).get('seed')
    if seed is not None:
        set_global_seed(seed)

    # Get module-specific RNG for reproducible sampling
    rng = get_rng(seed, module_name='clustering')

    # Use RNG for stochastic operations
    samples = rng.choice(data, size=100)
```

### Seed Utility Module

Location: `modules/utils/seed.py`

```python
"""Reproducibility utilities for deterministic execution."""

import random
import numpy as np
import torch
from typing import Optional

def set_global_seed(seed: int) -> None:
    """Set seed for all random number generators.

    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_rng(seed: Optional[int], module_name: str = "") -> np.random.RandomState:
    """Get a module-specific random state for isolated reproducibility.

    Args:
        seed: Base seed (None for random)
        module_name: Module identifier for offset calculation

    Returns:
        NumPy RandomState instance
    """
    if seed is None:
        return np.random.RandomState()

    # Create deterministic offset from module name
    offset = sum(ord(c) for c in module_name) % 1000
    return np.random.RandomState(seed + offset)
```

---

## Reproducibility Checklist

### For Demo Runs

- [ ] Use `configs/demo.yaml` (seed: 42)
- [ ] Use pinned dependencies from `requirements.lock`
- [ ] Use demo dataset from `examples/demo_data/`
- [ ] Run in Docker or verified environment

### For Production Runs

- [ ] Set explicit seed in config
- [ ] Record seed in output metadata
- [ ] Pin all dependencies
- [ ] Document software versions
- [ ] Archive input data hashes

---

## Validation

### Hash Verification

The framework generates SHA-256 hashes of key outputs for verification:

```yaml
# outputs/run_001/checksums.yaml
input_files:
  vcf: "sha256:abc123..."
  phenotypes: "sha256:def456..."
  pathways: "sha256:ghi789..."

output_files:
  subtype_assignments: "sha256:jkl012..."
  pathway_enrichment: "sha256:mno345..."
  embeddings: "sha256:pqr678..."
```

### Reproducibility Test

Run the built-in reproducibility test:

```bash
# Run twice with same seed
python -m autism_pathway_framework --config configs/demo.yaml
mv outputs/demo_run outputs/run_a

python -m autism_pathway_framework --config configs/demo.yaml
mv outputs/demo_run outputs/run_b

# Compare outputs
python -m autism_pathway_framework.utils.compare_runs outputs/run_a outputs/run_b
```

Expected output: `All outputs match. Reproducibility verified.`

---

## Known Non-Deterministic Operations

Some operations may produce slight variations even with fixed seeds:

| Operation | Cause | Mitigation |
|-----------|-------|------------|
| PyTorch GPU ops | CUDA non-determinism | Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Parallel execution | Thread ordering | Use `num_workers=0` in DataLoader |
| Float accumulation | Order of operations | Accept small epsilon differences |
| External API calls | Server-side randomness | Cache responses |

### Environment Variables for Full Determinism

```bash
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

---

## Demo Dataset Reproducibility

The demo dataset includes "planted" ground truth for validation:

### Planted Subtypes

The `planted_subtype` column in `demo_phenotypes.csv` contains the expected clustering outcome:

| Subtype | Samples | Key Pathways |
|---------|---------|--------------|
| synaptic | ~18 | SYNAPTIC_TRANSMISSION, GLUTAMATERGIC_SIGNALING |
| chromatin | ~17 | CHROMATIN_REMODELING, HISTONE_MODIFICATION |
| ion_channel | ~15 | ION_CHANNEL_REGULATION, SODIUM_CHANNEL_COMPLEX |

### Validation Test

```python
from sklearn.metrics import adjusted_rand_score

# After clustering
predicted = cluster_assignments['cluster_label']
expected = phenotypes['planted_subtype']

ari = adjusted_rand_score(expected, predicted)
assert ari > 0.7, f"Clustering failed to recover planted subtypes (ARI={ari})"
```

---

## Troubleshooting

### Different results on different machines

1. Check Python version: `python --version` (require 3.11+)
2. Check NumPy version: `pip show numpy` (must match requirements.lock)
3. Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
4. Verify seed is set: Check `run_metadata.yaml` for `random_seed`

### Results differ between CPU and GPU

GPU operations have inherent non-determinism. For exact reproducibility:

```yaml
pipeline:
  seed: 42
  device: "cpu"  # Force CPU for exact reproducibility
```

### Results differ across runs

Ensure no cached state from previous runs:

```bash
rm -rf outputs/demo_run
rm -rf .cache/
python -m autism_pathway_framework --config configs/demo.yaml
```
