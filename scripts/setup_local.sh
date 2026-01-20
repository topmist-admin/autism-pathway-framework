#!/bin/bash
# =============================================================================
# Local Development Setup Script
# =============================================================================
# Usage: bash scripts/setup_local.sh
#
# This script sets up the local development environment:
# 1. Creates directory structure
# 2. Copies environment template
# 3. Installs dependencies
# 4. Runs initial tests

set -e  # Exit on error

echo "=========================================="
echo "Autism Pathway Framework - Local Setup"
echo "=========================================="

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Create directory structure
# -----------------------------------------------------------------------------
echo "[1/6] Creating directory structure..."

mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/models
mkdir -p data/embeddings
mkdir -p logs
mkdir -p outputs

echo "  Created: data/raw, data/processed, data/models, data/embeddings"
echo "  Created: logs, outputs"

# -----------------------------------------------------------------------------
# Step 2: Set up environment file
# -----------------------------------------------------------------------------
echo ""
echo "[2/6] Setting up environment configuration..."

if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from .env.example"
    echo "  IMPORTANT: Edit .env with your GCP project ID and settings"
else
    echo "  .env already exists, skipping"
fi

# -----------------------------------------------------------------------------
# Step 3: Check Python virtual environment
# -----------------------------------------------------------------------------
echo ""
echo "[3/6] Checking Python environment..."

if [ -z "$VIRTUAL_ENV" ]; then
    echo "  WARNING: No virtual environment active"
    echo "  Please activate your virtual environment first:"
    echo "    source autismenv/bin/activate"
    echo ""
    read -p "  Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "  Virtual environment active: $VIRTUAL_ENV"
fi

# -----------------------------------------------------------------------------
# Step 4: Install dependencies
# -----------------------------------------------------------------------------
echo ""
echo "[4/6] Installing Python dependencies..."

pip install --upgrade pip
pip install -e ".[dev,bio,viz]"

echo "  Dependencies installed"

# -----------------------------------------------------------------------------
# Step 5: Set up pre-commit hooks (optional)
# -----------------------------------------------------------------------------
echo ""
echo "[5/6] Setting up development tools..."

# Install pre-commit if available
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "  Pre-commit hooks installed"
else
    echo "  Pre-commit not found, skipping hooks setup"
    echo "  Install with: pip install pre-commit"
fi

# -----------------------------------------------------------------------------
# Step 6: Run tests to verify setup
# -----------------------------------------------------------------------------
echo ""
echo "[6/6] Running tests to verify setup..."

python -m pytest tests/fixtures/ -v --tb=short 2>/dev/null || echo "  (Some tests may fail until data is downloaded)"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Edit .env with your configuration:"
echo "   vim .env"
echo ""
echo "2. Download required databases:"
echo "   bash scripts/download_databases.sh"
echo ""
echo "3. Authenticate with Google Cloud:"
echo "   gcloud auth login"
echo "   gcloud auth application-default login"
echo "   gcloud auth application-default set-quota-project YOUR_PROJECT_ID"
echo ""
echo "4. Run tests:"
echo "   pytest modules/01_data_loaders/tests/ -v"
echo ""
echo "5. Start developing!"
echo ""
