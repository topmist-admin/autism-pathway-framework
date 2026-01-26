# =============================================================================
# Autism Pathway Framework - Makefile
# =============================================================================
# Usage:
#   make help          - Show available commands
#   make setup         - Set up local development environment
#   make test          - Run all tests
#   make lint          - Run linters
#   make deploy-gcp    - Deploy to Google Cloud

.PHONY: help setup install test lint format clean deploy-gcp download-data docker-build docker-test verify

# Default target
help:
	@echo "Autism Pathway Framework - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "Development:"
	@echo "  make setup          Set up local development environment"
	@echo "  make install        Install dependencies only"
	@echo "  make test           Run all tests"
	@echo "  make test-fast      Run tests (skip slow tests)"
	@echo "  make lint           Run linters (ruff, mypy)"
	@echo "  make format         Format code with black"
	@echo "  make clean          Remove build artifacts"
	@echo ""
	@echo "Data:"
	@echo "  make download-data  Download required databases (minimal)"
	@echo "  make download-all   Download all databases (includes large files)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-test    Run tests in Docker"
	@echo "  make docker-demo    Run demo pipeline in Docker"
	@echo "  make docker-shell   Open shell in Docker container"
	@echo ""
	@echo "Verification:"
	@echo "  make verify         Verify environment setup"
	@echo "  make verify-lock    Check if requirements.lock is up to date"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-gcp     Set up GCP resources"
	@echo "  make sync-to-gcs    Sync local data to GCS bucket"
	@echo "  make sync-from-gcs  Sync GCS data to local"
	@echo ""

# =============================================================================
# Development Commands
# =============================================================================

setup:
	@echo "Setting up local development environment..."
	bash scripts/setup_local.sh

install:
	pip install --upgrade pip
	pip install -e ".[dev,bio,viz]"

install-all:
	pip install --upgrade pip
	pip install -e ".[all]"

test:
	python -m pytest tests/ modules/ -v --tb=short

test-fast:
	python -m pytest tests/ modules/ -v --tb=short -m "not slow"

test-cov:
	python -m pytest tests/ modules/ -v --cov=modules --cov-report=html

lint:
	ruff check modules/ tests/
	mypy modules/ --ignore-missing-imports

format:
	black modules/ tests/
	ruff check modules/ tests/ --fix

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# =============================================================================
# Data Commands
# =============================================================================

download-data:
	bash scripts/download_databases.sh --minimal

download-all:
	bash scripts/download_databases.sh --all

# =============================================================================
# GCP Deployment Commands
# =============================================================================

deploy-gcp:
	bash deploy/gcp_setup.sh

sync-to-gcs:
	@if [ -z "$(GCS_BUCKET)" ]; then \
		echo "Error: GCS_BUCKET not set. Run: export GCS_BUCKET=your-bucket-name"; \
		exit 1; \
	fi
	gcloud storage rsync data/ gs://$(GCS_BUCKET)/data/ --recursive

sync-from-gcs:
	@if [ -z "$(GCS_BUCKET)" ]; then \
		echo "Error: GCS_BUCKET not set. Run: export GCS_BUCKET=your-bucket-name"; \
		exit 1; \
	fi
	gcloud storage rsync gs://$(GCS_BUCKET)/data/ data/ --recursive

# =============================================================================
# Module-specific Commands
# =============================================================================

test-module-01:
	python -m pytest modules/01_data_loaders/tests/ -v

test-module-02:
	python -m pytest modules/02_variant_processing/tests/ -v

# =============================================================================
# Documentation
# =============================================================================

docs-serve:
	@echo "Documentation is in docs/ directory"
	python -m http.server 8000 --directory docs/

# =============================================================================
# Docker Commands (v0.1)
# =============================================================================

docker-build:
	@echo "Building Docker image..."
	docker build -t autism-pathway-framework:v0.1 .

docker-test:
	@echo "Running tests in Docker..."
	docker run autism-pathway-framework:v0.1 pytest tests/ -v --tb=short

docker-demo:
	@echo "Running demo in Docker..."
	docker run -v $(PWD)/outputs:/app/outputs autism-pathway-framework:v0.1 \
		python -m autism_pathway_framework --config configs/demo.yaml

docker-shell:
	docker run -it autism-pathway-framework:v0.1 bash

# =============================================================================
# Verification Commands (v0.1)
# =============================================================================

verify:
	@echo "Verifying environment setup..."
	@echo ""
	@echo "1. Python version:"
	@python --version
	@echo ""
	@echo "2. Key packages:"
	@python -c "import numpy; print(f'  numpy: {numpy.__version__}')"
	@python -c "import pandas; print(f'  pandas: {pandas.__version__}')"
	@python -c "import torch; print(f'  torch: {torch.__version__}')"
	@python -c "import networkx; print(f'  networkx: {networkx.__version__}')"
	@echo ""
	@echo "3. Module imports:"
	@python -c "from modules import __init__; print('  modules: OK')" 2>/dev/null || echo "  modules: SKIP (editable install required)"
	@python -c "import sys; sys.path.insert(0, '.'); from modules.01_data_loaders import vcf_loader; print('  01_data_loaders: OK')" 2>/dev/null || echo "  01_data_loaders: SKIP"
	@python -c "import sys; sys.path.insert(0, '.'); from modules.03_knowledge_graph import schema; print('  03_knowledge_graph: OK')" 2>/dev/null || echo "  03_knowledge_graph: SKIP"
	@echo ""
	@echo "Verification complete!"

verify-lock:
	@echo "Verifying requirements.lock matches installed packages..."
	@pip freeze | grep -v "^-e " | diff - requirements.lock && echo "requirements.lock is up to date" || echo "WARNING: requirements.lock may be out of date"
