# Golden Outputs Reference

This directory contains reference values for cross-machine reproducibility validation.

## Purpose

Golden outputs serve as the "expected" results that CI and local verification
compare against. They ensure that:

1. **Reproducibility**: Same inputs + same seed = same outputs across machines
2. **Regression detection**: Unintended changes to pipeline logic are caught
3. **CI validation**: Automated checks verify every push

## Files

| File | Description |
|------|-------------|
| `expected_outputs.yaml` | Reference hashes, metrics, and validation thresholds |

## How It Works

1. **CI runs the demo pipeline** with `seed=42` and `PYTHONHASHSEED=42`
2. **Verification script compares outputs** against `expected_outputs.yaml`
3. **Checks include:**
   - Input file hashes (unchanged demo data)
   - Output dimensions (n_samples, n_variants, etc.)
   - Cluster counts
   - Validation gate results within expected ranges

## Updating Golden Outputs

**Only update when pipeline changes are intentional!**

```bash
make update-golden
```

This will:
1. Run the demo pipeline
2. Extract metrics from outputs
3. Update `expected_outputs.yaml`

Then commit the changes:
```bash
git add tests/golden/expected_outputs.yaml
git commit -m "chore: update golden outputs for pipeline changes"
```

## Tolerances

Not all values must match exactly:

| Metric | Tolerance | Reason |
|--------|-----------|--------|
| Pathway scores | 1e-10 | Should be exactly reproducible |
| ARI values | 0.05 | Validation metrics may vary slightly |
| Cluster counts | Exact | Must be identical |

## Troubleshooting

If reproducibility verification fails:

1. **Check Python/package versions** - Must match CI environment
2. **Verify input files unchanged** - Hash check should pass
3. **Set environment variables**:
   ```bash
   export PYTHONHASHSEED=42
   export OMP_NUM_THREADS=1
   ```
4. **Clear cached data**:
   ```bash
   rm -rf outputs/ .cache/
   ```

See [docs/troubleshooting.md](../../docs/troubleshooting.md) for more details.
