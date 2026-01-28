# =============================================================================
# Autism Pathway Framework - Dockerfile
# =============================================================================
# Build: docker build -t autism-pathway-framework:v0.1 .
# Run:   docker run -it autism-pathway-framework:v0.1
# Demo:  docker run -it autism-pathway-framework:v0.1 python -m autism_pathway_framework --config configs/demo.yaml

FROM python:3.11-slim-bookworm

# =============================================================================
# Metadata
# =============================================================================
LABEL maintainer="Autism Pathway Framework Team"
LABEL version="0.1.0"
LABEL description="Research framework for pathway-based analysis of genetic heterogeneity in ASD"

# =============================================================================
# Environment Variables
# =============================================================================
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# =============================================================================
# System Dependencies
# =============================================================================
# Note: libhts-dev depends on libcurl4-gnutls-dev, so we don't install libcurl4-openssl-dev
# (they conflict with each other)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libhdf5-dev \
    libhts-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libssl-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python Dependencies
# =============================================================================

# Copy requirements first for better caching
COPY requirements.txt requirements.lock ./

# Install from locked requirements if available, otherwise from requirements.txt
RUN if [ -f requirements.lock ]; then \
        pip install -r requirements.lock; \
    else \
        pip install -r requirements.txt; \
    fi

# =============================================================================
# Application Code
# =============================================================================

# Copy project files
COPY pyproject.toml setup.py* ./
COPY modules/ ./modules/
COPY pipelines/ ./pipelines/
COPY configs/ ./configs/
COPY examples/ ./examples/
COPY docs/ ./docs/

# Install the package in editable mode
RUN pip install -e .

# =============================================================================
# Create Runtime Directories
# =============================================================================
RUN mkdir -p /app/data/raw \
    /app/data/processed \
    /app/data/models \
    /app/data/embeddings \
    /app/outputs \
    /app/logs

# =============================================================================
# Health Check
# =============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import modules; print('OK')" || exit 1

# =============================================================================
# Default Command
# =============================================================================
CMD ["python", "-c", "print('Autism Pathway Framework v0.1 - Use: python -m autism_pathway_framework --help')"]

# =============================================================================
# Usage Examples (in comments)
# =============================================================================
# Build the image:
#   docker build -t autism-pathway-framework:v0.1 .
#
# Run demo:
#   docker run -v $(pwd)/outputs:/app/outputs autism-pathway-framework:v0.1 \
#     python -m autism_pathway_framework --config configs/demo.yaml
#
# Interactive shell:
#   docker run -it autism-pathway-framework:v0.1 bash
#
# Run tests:
#   docker run autism-pathway-framework:v0.1 pytest tests/ -v
