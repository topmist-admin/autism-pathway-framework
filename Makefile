# =============================================================================
# Autism Pathway Framework - Makefile
# =============================================================================
# Usage:
#   make help          - Show available commands
#   make setup         - Set up local development environment
#   make test          - Run all tests
#   make lint          - Run linters
#   make deploy-gcp    - Deploy to Google Cloud

.PHONY: help setup install test lint format clean deploy-gcp download-data

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
