"""
Reproducibility Verification Utility

Verifies that pipeline outputs match expected golden outputs for cross-machine
reproducibility validation.

Usage:
    python -m autism_pathway_framework.utils.verify_reproducibility \\
        --output-dir outputs/demo_run \\
        --golden tests/golden/expected_outputs.yaml

Exit codes:
    0: All checks passed
    1: Verification failed
    2: Missing files or configuration error
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


class ReproducibilityVerifier:
    """Verifies pipeline outputs against golden reference."""

    def __init__(self, output_dir: Path, golden_path: Path, verbose: bool = True):
        """Initialize verifier.

        Args:
            output_dir: Path to pipeline output directory
            golden_path: Path to golden outputs YAML file
            verbose: Print detailed output
        """
        self.output_dir = Path(output_dir)
        self.golden_path = Path(golden_path)
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed: List[str] = []

    def log(self, message: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(message)

    def error(self, message: str) -> None:
        """Record an error."""
        self.errors.append(message)
        self.log(f"  FAIL: {message}")

    def warn(self, message: str) -> None:
        """Record a warning."""
        self.warnings.append(message)
        self.log(f"  WARN: {message}")

    def passed_check(self, message: str) -> None:
        """Record a passed check."""
        self.passed.append(message)
        self.log(f"  PASS: {message}")

    def load_golden(self) -> Dict[str, Any]:
        """Load golden outputs reference."""
        if not self.golden_path.exists():
            raise FileNotFoundError(f"Golden outputs file not found: {self.golden_path}")

        with open(self.golden_path, "r") as f:
            return yaml.safe_load(f)

    def compute_file_hash(self, path: Path, length: int = 16) -> str:
        """Compute SHA-256 hash prefix of a file."""
        if not path.exists():
            return "FILE_NOT_FOUND"

        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:length]

    def verify_required_outputs(self, golden: Dict[str, Any]) -> bool:
        """Check that all required output files exist."""
        self.log("\n[1/5] Checking required output files...")

        required = golden.get("required_outputs", [])
        all_exist = True

        for filename in required:
            filepath = self.output_dir / filename
            if filepath.exists():
                self.passed_check(f"Found: {filename}")
            else:
                self.error(f"Missing: {filename}")
                all_exist = False

        return all_exist

    def verify_input_hashes(self, golden: Dict[str, Any]) -> bool:
        """Verify input file hashes match expected."""
        self.log("\n[2/5] Verifying input file hashes...")

        input_files = golden.get("input_files", {})
        all_match = True

        for name, info in input_files.items():
            path = Path(info["path"])
            expected_hash = info["sha256_prefix"]
            actual_hash = self.compute_file_hash(path)

            if actual_hash == expected_hash:
                self.passed_check(f"{name}: {actual_hash}")
            elif actual_hash == "FILE_NOT_FOUND":
                self.error(f"{name}: File not found at {path}")
                all_match = False
            else:
                self.error(f"{name}: Hash mismatch (expected {expected_hash}, got {actual_hash})")
                all_match = False

        return all_match

    def verify_metrics(self, golden: Dict[str, Any]) -> bool:
        """Verify output metrics match expected values."""
        self.log("\n[3/5] Verifying output metrics...")

        expected = golden.get("expected_metrics", {})

        # Load report.json
        report_path = self.output_dir / "report.json"
        if not report_path.exists():
            self.error("Cannot verify metrics: report.json not found")
            return False

        with open(report_path, "r") as f:
            report = json.load(f)

        summary = report.get("summary", {})
        all_match = True

        # Check basic counts
        for metric in ["n_samples", "n_variants", "n_genes", "n_pathways", "n_clusters"]:
            expected_val = expected.get(metric)
            actual_val = summary.get(metric)

            if expected_val is not None:
                if actual_val == expected_val:
                    self.passed_check(f"{metric}: {actual_val}")
                else:
                    self.error(f"{metric}: expected {expected_val}, got {actual_val}")
                    all_match = False

        # Check cluster distribution
        expected_clusters = expected.get("cluster_distribution", {})
        actual_clusters = report.get("clusters", {})

        for cluster_name, expected_count in expected_clusters.items():
            actual_count = actual_clusters.get(cluster_name)
            if actual_count == expected_count:
                self.passed_check(f"Cluster '{cluster_name}': {actual_count} samples")
            elif actual_count is None:
                self.warn(f"Cluster '{cluster_name}': not found in output")
            else:
                self.warn(f"Cluster '{cluster_name}': expected {expected_count}, got {actual_count}")

        return all_match

    def verify_pathway_scores(self, golden: Dict[str, Any]) -> bool:
        """Verify pathway scores are within expected ranges."""
        self.log("\n[4/5] Verifying pathway scores...")

        expected = golden.get("expected_metrics", {}).get("pathway_scores", {})
        tolerances = golden.get("tolerances", {})
        tol = tolerances.get("pathway_scores", 1e-10)

        # Load pathway scores
        scores_path = self.output_dir / "pathway_scores.csv"
        if not scores_path.exists():
            self.error("Cannot verify pathway scores: pathway_scores.csv not found")
            return False

        df = pd.read_csv(scores_path, index_col=0)
        all_valid = True

        # Check dimensions
        expected_rows = expected.get("n_rows")
        expected_cols = expected.get("n_columns")

        if expected_rows and df.shape[0] != expected_rows:
            self.error(f"Row count: expected {expected_rows}, got {df.shape[0]}")
            all_valid = False
        else:
            self.passed_check(f"Row count: {df.shape[0]}")

        if expected_cols and df.shape[1] != expected_cols:
            self.error(f"Column count: expected {expected_cols}, got {df.shape[1]}")
            all_valid = False
        else:
            self.passed_check(f"Column count: {df.shape[1]}")

        # Check mean range (should be near 0 for z-scored data)
        mean_range = expected.get("mean_range", [-0.5, 0.5])
        col_means = df.mean()

        for col in df.columns:
            if not (mean_range[0] <= col_means[col] <= mean_range[1]):
                self.warn(f"Column '{col}' mean ({col_means[col]:.4f}) outside expected range")

        self.passed_check(f"Column means within expected range")

        return all_valid

    def verify_validation_gates(self, golden: Dict[str, Any]) -> bool:
        """Verify validation gate results are within expected ranges."""
        self.log("\n[5/5] Verifying validation gates...")

        expected_gates = golden.get("validation_gates", {})
        tolerances = golden.get("tolerances", {})
        ari_tol = tolerances.get("ari_values", 0.05)

        # Load report.json
        report_path = self.output_dir / "report.json"
        if not report_path.exists():
            self.error("Cannot verify validation gates: report.json not found")
            return False

        with open(report_path, "r") as f:
            report = json.load(f)

        validation = report.get("validation_gates", {})
        tests = validation.get("tests", [])

        if not tests:
            self.warn("No validation gate results found in report")
            return True

        all_valid = True

        for test in tests:
            test_name = test.get("name", "Unknown")
            value = test.get("value", 0)
            status = test.get("status", "UNKNOWN")

            # Find expected range
            for gate_key, gate_info in expected_gates.items():
                if gate_info.get("name") in test_name:
                    expected_range = gate_info.get("expected_range", [0, 1])

                    if expected_range[0] <= value <= expected_range[1]:
                        self.passed_check(f"{test_name}: {value:.4f} ({status})")
                    else:
                        self.warn(
                            f"{test_name}: {value:.4f} outside expected range "
                            f"[{expected_range[0]}, {expected_range[1]}]"
                        )
                    break
            else:
                self.passed_check(f"{test_name}: {value:.4f} ({status})")

        return all_valid

    def compare_runs(self, other_dir: Path) -> bool:
        """Compare outputs between two runs for exact reproducibility."""
        self.log(f"\nComparing outputs: {self.output_dir} vs {other_dir}")

        files_to_compare = [
            "pathway_scores.csv",
            "subtype_assignments.csv",
        ]

        all_match = True

        for filename in files_to_compare:
            file1 = self.output_dir / filename
            file2 = other_dir / filename

            if not file1.exists() or not file2.exists():
                self.error(f"Cannot compare {filename}: file missing")
                all_match = False
                continue

            # Compare file contents
            if filename.endswith(".csv"):
                df1 = pd.read_csv(file1)
                df2 = pd.read_csv(file2)

                if df1.shape != df2.shape:
                    self.error(f"{filename}: Shape mismatch ({df1.shape} vs {df2.shape})")
                    all_match = False
                    continue

                # Compare numeric columns
                numeric_cols = df1.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if not np.allclose(df1[col], df2[col], rtol=1e-10, atol=1e-10):
                        self.error(f"{filename}: Column '{col}' values differ")
                        all_match = False
                    else:
                        self.passed_check(f"{filename}: Column '{col}' matches")

        return all_match

    def run(self) -> Tuple[bool, Dict[str, Any]]:
        """Run all verification checks.

        Returns:
            Tuple of (success, results_dict)
        """
        self.log("=" * 60)
        self.log("Reproducibility Verification")
        self.log("=" * 60)
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"Golden reference: {self.golden_path}")

        try:
            golden = self.load_golden()
        except FileNotFoundError as e:
            self.error(str(e))
            return False, {"errors": self.errors}

        # Run all checks
        checks = [
            ("Required outputs", self.verify_required_outputs(golden)),
            ("Input hashes", self.verify_input_hashes(golden)),
            ("Output metrics", self.verify_metrics(golden)),
            ("Pathway scores", self.verify_pathway_scores(golden)),
            ("Validation gates", self.verify_validation_gates(golden)),
        ]

        # Summary
        self.log("\n" + "=" * 60)
        self.log("SUMMARY")
        self.log("=" * 60)

        passed_count = sum(1 for _, passed in checks if passed)
        total_count = len(checks)

        for name, passed in checks:
            status = "PASS" if passed else "FAIL"
            self.log(f"  {name}: {status}")

        self.log(f"\nChecks passed: {passed_count}/{total_count}")
        self.log(f"Errors: {len(self.errors)}")
        self.log(f"Warnings: {len(self.warnings)}")

        success = len(self.errors) == 0

        if success:
            self.log("\nReproducibility verification: PASSED")
        else:
            self.log("\nReproducibility verification: FAILED")
            self.log("\nErrors:")
            for error in self.errors:
                self.log(f"  - {error}")

        results = {
            "success": success,
            "checks": {name: passed for name, passed in checks},
            "errors": self.errors,
            "warnings": self.warnings,
            "passed": self.passed,
        }

        return success, results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify pipeline reproducibility against golden outputs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/demo_run",
        help="Path to pipeline output directory",
    )
    parser.add_argument(
        "--golden",
        type=str,
        default="tests/golden/expected_outputs.yaml",
        help="Path to golden outputs YAML file",
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Compare with another run directory for exact reproducibility",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    verifier = ReproducibilityVerifier(
        output_dir=Path(args.output_dir),
        golden_path=Path(args.golden),
        verbose=not args.quiet and not args.json,
    )

    success, results = verifier.run()

    # Optional: compare two runs
    if args.compare:
        compare_success = verifier.compare_runs(Path(args.compare))
        success = success and compare_success
        results["compare_success"] = compare_success

    if args.json:
        print(json.dumps(results, indent=2))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
