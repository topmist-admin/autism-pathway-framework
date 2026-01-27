#!/bin/bash
# =============================================================================
# Autism Pathway Framework - Demo Runner
# =============================================================================
# Usage: ./run_demo.sh
#
# Runs the complete demo pipeline with the synthetic dataset.
# Output will be generated in: outputs/demo_run/

set -e  # Exit on error

echo "=============================================="
echo "Autism Pathway Framework - Demo Pipeline"
echo "=============================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Check if virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment active"
    echo "Consider running: source autismenv/bin/activate"
    echo ""
fi

# Check required files exist
if [ ! -f "configs/demo.yaml" ]; then
    echo "Error: configs/demo.yaml not found"
    exit 1
fi

if [ ! -f "examples/demo_data/demo_variants.vcf" ]; then
    echo "Error: Demo data not found. Run Week 3 setup first."
    exit 1
fi

# Clean previous output
if [ -d "outputs/demo_run" ]; then
    echo "Removing previous demo output..."
    rm -rf outputs/demo_run
fi

# Run the pipeline
echo "Starting pipeline..."
echo ""

python -m autism_pathway_framework --config configs/demo.yaml

echo ""
echo "=============================================="
echo "Demo Complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  - outputs/demo_run/pathway_scores.csv"
echo "  - outputs/demo_run/subtype_assignments.csv"
echo "  - outputs/demo_run/figures/summary.png"
echo "  - outputs/demo_run/report.json"
echo "  - outputs/demo_run/report.md"
echo ""
