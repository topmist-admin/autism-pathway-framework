#!/usr/bin/env python3
"""
Update Golden Outputs Reference

This script updates the golden outputs reference file with values from
a fresh demo pipeline run. Only use this when pipeline changes are intentional.

Usage:
    python scripts/update_golden_outputs.py

The script will:
1. Run the demo pipeline with seed=42
2. Extract metrics from the outputs
3. Update tests/golden/expected_outputs.yaml

WARNING: This should only be done when pipeline logic changes are intentional.
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import yaml


def compute_file_hash(path: Path, length: int = 16) -> str:
    """Compute SHA-256 hash prefix of a file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:length]


def run_demo_pipeline():
    """Run the demo pipeline."""
    print("Running demo pipeline...")

    # Set environment for reproducibility
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"

    result = subprocess.run(
        [sys.executable, "-m", "autism_pathway_framework", "--config", "configs/demo.yaml"],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Pipeline failed:\n{result.stderr}")
        sys.exit(1)

    print("Pipeline completed successfully")


def extract_metrics() -> dict:
    """Extract metrics from pipeline outputs."""
    output_dir = Path("outputs/demo_run")

    # Load report.json
    with open(output_dir / "report.json", "r") as f:
        report = json.load(f)

    summary = report.get("summary", {})
    clusters = report.get("clusters", {})

    return {
        "n_samples": summary.get("n_samples"),
        "n_variants": summary.get("n_variants"),
        "n_genes": summary.get("n_genes"),
        "n_pathways": summary.get("n_pathways"),
        "n_clusters": summary.get("n_clusters"),
        "cluster_distribution": clusters,
    }


def update_golden_outputs():
    """Update the golden outputs reference file."""
    golden_path = Path("tests/golden/expected_outputs.yaml")

    # Load existing golden outputs
    with open(golden_path, "r") as f:
        golden = yaml.safe_load(f)

    # Compute input file hashes
    input_files = {
        "vcf": {
            "path": "examples/demo_data/demo_variants.vcf",
            "sha256_prefix": compute_file_hash(Path("examples/demo_data/demo_variants.vcf")),
        },
        "phenotypes": {
            "path": "examples/demo_data/demo_phenotypes.csv",
            "sha256_prefix": compute_file_hash(Path("examples/demo_data/demo_phenotypes.csv")),
        },
        "pathways": {
            "path": "examples/demo_data/demo_pathways.gmt",
            "sha256_prefix": compute_file_hash(Path("examples/demo_data/demo_pathways.gmt")),
        },
    }

    # Extract metrics from current run
    metrics = extract_metrics()

    # Update golden outputs
    golden["generated_date"] = str(date.today())
    golden["input_files"] = input_files
    golden["expected_metrics"].update(metrics)

    # Write updated golden outputs
    with open(golden_path, "w") as f:
        yaml.dump(golden, f, default_flow_style=False, sort_keys=False)

    print(f"Updated {golden_path}")
    print(f"  n_samples: {metrics['n_samples']}")
    print(f"  n_variants: {metrics['n_variants']}")
    print(f"  n_genes: {metrics['n_genes']}")
    print(f"  n_pathways: {metrics['n_pathways']}")
    print(f"  n_clusters: {metrics['n_clusters']}")
    print(f"  clusters: {metrics['cluster_distribution']}")


def main():
    """Main entry point."""
    # Change to repository root
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)

    print("=" * 60)
    print("Update Golden Outputs Reference")
    print("=" * 60)
    print()
    print("WARNING: This will update the reference values used for")
    print("reproducibility validation. Only proceed if pipeline")
    print("changes are intentional.")
    print()

    # Run pipeline
    run_demo_pipeline()

    # Update golden outputs
    update_golden_outputs()

    print()
    print("Done! Don't forget to commit the updated golden outputs:")
    print("  git add tests/golden/expected_outputs.yaml")
    print("  git commit -m 'chore: update golden outputs for v0.1.x'")


if __name__ == "__main__":
    main()
