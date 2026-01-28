# Troubleshooting Guide

> **RESEARCH USE ONLY** — This framework is for research purposes only. Not for clinical decision-making. See [DISCLAIMER.md](../DISCLAIMER.md).

This guide covers common issues when running the Autism Pathway Framework and their solutions.

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Environment Setup](#environment-setup)
3. [Pipeline Errors](#pipeline-errors)
4. [Reproducibility Issues](#reproducibility-issues)
5. [Performance Issues](#performance-issues)
6. [Validation Gate Failures](#validation-gate-failures)
7. [Platform-Specific Issues](#platform-specific-issues)
8. [Getting Help](#getting-help)

---

## Installation Issues

### `pip install` fails with dependency conflicts

**Symptom:**
```
ERROR: Cannot install autism-pathway-framework because these package versions have conflicting dependencies.
```

**Solution:**
1. Create a fresh virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or: venv\Scripts\activate  # Windows
   ```

2. Install with pinned dependencies:
   ```bash
   pip install -r requirements.lock
   pip install -e .
   ```

### PyTorch installation fails

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement torch
```

**Solution:**
Install PyTorch separately before other dependencies:
```bash
# CPU only (recommended for reproducibility)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# With CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Missing system dependencies on Linux

**Symptom:**
```
error: command 'gcc' failed with exit status 1
```

**Solution:**
Install build dependencies:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential python3-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```

---

## Environment Setup

### `make setup` fails

**Symptom:**
```
bash: scripts/setup_local.sh: No such file or directory
```

**Solution:**
Ensure you're in the repository root directory:
```bash
cd autism-pathway-framework
ls scripts/setup_local.sh  # Should exist
make setup
```

### Python version mismatch

**Symptom:**
```
This package requires Python >=3.10
```

**Solution:**
1. Check your Python version:
   ```bash
   python --version
   ```

2. Install Python 3.10+ using pyenv:
   ```bash
   pyenv install 3.11.0
   pyenv local 3.11.0
   ```

### Module import errors after installation

**Symptom:**
```python
ModuleNotFoundError: No module named 'autism_pathway_framework'
```

**Solution:**
Install in editable mode:
```bash
pip install -e .
```

Or add the repository to your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/autism-pathway-framework"
```

---

## Pipeline Errors

### FileNotFoundError for demo data

**Symptom:**
```
FileNotFoundError: VCF file not found: examples/demo_data/demo_variants.vcf
```

**Solution:**
1. Ensure demo data exists:
   ```bash
   ls examples/demo_data/
   ```

2. If missing, the demo data may need to be generated:
   ```bash
   python scripts/generate_demo_data.py
   ```

3. Or download from releases:
   ```bash
   make download-data
   ```

### Configuration file not found

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'configs/demo.yaml'
```

**Solution:**
Run from the repository root:
```bash
cd /path/to/autism-pathway-framework
python -m autism_pathway_framework --config configs/demo.yaml
```

### Memory error during clustering

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solution:**
1. Reduce the dataset size or use a machine with more RAM
2. Close other applications to free memory
3. For large cohorts, use the production config with batching:
   ```bash
   python -m autism_pathway_framework --config configs/production.yaml
   ```

### Validation module import error

**Symptom:**
```
ImportError: cannot import name 'ValidationGates' from 'autism_pathway_framework.validation'
```

**Solution:**
Reinstall the package:
```bash
pip uninstall autism-pathway-framework
pip install -e .
```

---

## Reproducibility Issues

### Different results on different machines

**Symptom:**
Outputs differ between runs or machines despite using the same seed.

**Diagnosis:**
1. Check Python version:
   ```bash
   python --version
   ```

2. Check NumPy version:
   ```bash
   python -c "import numpy; print(numpy.__version__)"
   ```

3. Check PyTorch version:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

4. Verify seed is set:
   ```bash
   grep "random_seed" outputs/demo_run/run_metadata.yaml
   ```

**Solution:**
1. Use pinned dependencies:
   ```bash
   pip install -r requirements.lock
   ```

2. Set environment variables for full determinism:
   ```bash
   export PYTHONHASHSEED=42
   export CUBLAS_WORKSPACE_CONFIG=:4096:8
   export OMP_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   ```

3. Force CPU execution:
   ```yaml
   # In config file
   pipeline:
     seed: 42
     device: "cpu"
   ```

### Results differ between CPU and GPU

**Symptom:**
GPU runs produce slightly different results than CPU runs.

**Explanation:**
GPU operations have inherent non-determinism due to parallel execution order.

**Solution:**
For exact reproducibility, use CPU:
```yaml
pipeline:
  seed: 42
  device: "cpu"
```

Or enable PyTorch deterministic mode (slower):
```python
import torch
torch.use_deterministic_algorithms(True)
```

### Hash verification fails

**Symptom:**
```
ERROR: Hash mismatch for pathway_scores.csv
```

**Solution:**
1. Ensure you're using the exact same:
   - Python version
   - Package versions (use `requirements.lock`)
   - Input files (verify hashes)

2. Clear any cached data:
   ```bash
   rm -rf outputs/demo_run
   rm -rf .cache/
   ```

3. Run with verbose logging:
   ```bash
   python -m autism_pathway_framework --config configs/demo.yaml --verbose
   ```

### Reproducibility verification fails in CI

**Symptom:**
CI reproducibility check fails but local runs succeed.

**Solution:**
1. Check the CI logs for the specific failure
2. Compare environment:
   ```bash
   # Local
   pip freeze > local_packages.txt

   # Compare with CI output in artifacts
   ```

3. Ensure `PYTHONHASHSEED` is set in CI

---

## Performance Issues

### Pipeline runs very slowly

**Symptom:**
Demo takes more than 60 minutes.

**Solution:**
1. Check for disk I/O bottlenecks:
   ```bash
   iostat -x 1 5
   ```

2. Reduce validation iterations:
   ```yaml
   validation:
     label_shuffle:
       n_permutations: 50  # Reduce from 100
     stability:
       n_bootstrap: 25     # Reduce from 100
   ```

3. Use fewer clusters:
   ```yaml
   clustering:
     n_clusters_range: [2, 5]  # Narrower range
   ```

### High memory usage

**Symptom:**
System becomes unresponsive or swaps heavily.

**Solution:**
1. Monitor memory:
   ```bash
   watch -n 1 free -h
   ```

2. Use memory-efficient settings:
   ```yaml
   pipeline:
     batch_size: 10  # Process fewer samples at once
   ```

3. Run on a machine with more RAM (16 GB recommended)

---

## Validation Gate Failures

### Negative Control 1 fails (Label Shuffle)

**Symptom:**
```
Negative Control 1: FAIL (ARI = 0.25)
```

**Meaning:**
Clustering is finding structure in random data, suggesting overfitting.

**Investigation:**
1. Check if the number of clusters is too high
2. Review pathway definitions for redundancy
3. Consider increasing the shuffle permutations

### Negative Control 2 fails (Random Gene Sets)

**Symptom:**
```
Negative Control 2: FAIL (mean_random_ARI = 0.18)
```

**Meaning:**
Random gene sets produce similar cluster structure, suggesting signal may not be pathway-specific.

**Investigation:**
1. Check pathway overlap with demo genes
2. Verify pathway database integrity
3. This can fail on small datasets - not always a concern

### Stability Test fails (Bootstrap)

**Symptom:**
```
Stability Test: FAIL (ARI = 0.65)
```

**Meaning:**
Clusters are not robust to resampling.

**Investigation:**
1. Dataset may be too small for stable clustering
2. Try reducing the number of clusters
3. Check for outlier samples driving instability

---

## Platform-Specific Issues

### macOS

**Issue:** `libiomp5.dylib` conflict with Accelerate
```
OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
```

**Solution:**
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

**Issue:** SSL certificate errors
```
ssl.SSLCertVerificationError: certificate verify failed
```

**Solution:**
```bash
/Applications/Python\ 3.11/Install\ Certificates.command
```

### Linux (Ubuntu)

**Issue:** Missing BLAS libraries
```
numpy.linalg.LinAlgError: LAPACK library not found
```

**Solution:**
```bash
sudo apt-get install libopenblas-dev liblapack-dev
```

### Windows (WSL2)

**Issue:** Path issues with Windows/Linux
```
FileNotFoundError: /mnt/c/Users/.../demo_variants.vcf
```

**Solution:**
Run entirely within WSL2:
```bash
cd ~/autism-pathway-framework  # Linux path
python -m autism_pathway_framework --config configs/demo.yaml
```

**Issue:** Display issues for matplotlib
```
cannot connect to X server
```

**Solution:**
Use non-interactive backend:
```python
import matplotlib
matplotlib.use('Agg')
```

---

## Getting Help

### Before asking for help

1. **Check this guide** for your specific error
2. **Search existing issues** on GitHub
3. **Collect diagnostic information:**
   ```bash
   python --version
   pip freeze | grep -E "(numpy|pandas|torch|sklearn)"
   cat outputs/demo_run/pipeline.log | tail -50
   ```

### Reporting issues

When opening a GitHub issue, include:

1. **Environment:**
   - OS and version
   - Python version
   - Key package versions

2. **Steps to reproduce:**
   - Exact commands run
   - Configuration file used

3. **Error message:**
   - Full traceback
   - Relevant log output

4. **Expected vs actual behavior**

### Community resources

- **GitHub Issues:** [Report bugs and request features](https://github.com/your-org/autism-pathway-framework/issues)
- **Discussions:** [Ask questions and share ideas](https://github.com/your-org/autism-pathway-framework/discussions)

---

## Quick Reference

### Common Commands

```bash
# Verify environment
make verify

# Run demo
python -m autism_pathway_framework --config configs/demo.yaml

# Verify reproducibility
python -m autism_pathway_framework.utils.verify_reproducibility \
    --output-dir outputs/demo_run \
    --golden tests/golden/expected_outputs.yaml

# Run tests
make test

# Clean up
rm -rf outputs/ .cache/ __pycache__/
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `PYTHONHASHSEED` | Hash randomization seed | Random |
| `OMP_NUM_THREADS` | OpenMP threads | Auto |
| `MKL_NUM_THREADS` | MKL threads | Auto |
| `CUDA_VISIBLE_DEVICES` | GPU selection | All |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |
| 3 | Validation failure |

---

*Last updated: January 2026 (Week 7)*
