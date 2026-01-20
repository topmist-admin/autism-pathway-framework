#!/bin/bash
# =============================================================================
# Download External Databases Script
# =============================================================================
# Usage: bash scripts/download_databases.sh [--all|--minimal]
#
# Downloads required biological databases for the framework.
# Use --minimal for development (smaller files only)
# Use --all for full production setup

set -e

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default data directory
DATA_DIR="${DATA_DIR:-./data}"
RAW_DIR="${DATA_DIR}/raw"

# Parse arguments
DOWNLOAD_MODE="${1:---minimal}"

echo "=========================================="
echo "Downloading Biological Databases"
echo "Mode: $DOWNLOAD_MODE"
echo "Target: $RAW_DIR"
echo "=========================================="

mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

# -----------------------------------------------------------------------------
# Gene Ontology (Required)
# -----------------------------------------------------------------------------
echo ""
echo "[1/7] Gene Ontology..."

if [ ! -f "go-basic.obo" ]; then
    echo "  Downloading go-basic.obo..."
    curl -L -o go-basic.obo "http://purl.obolibrary.org/obo/go/go-basic.obo"
    echo "  Downloaded: go-basic.obo ($(du -h go-basic.obo | cut -f1))"
else
    echo "  Already exists: go-basic.obo"
fi

# -----------------------------------------------------------------------------
# Gene Ontology Annotations (Required)
# -----------------------------------------------------------------------------
echo ""
echo "[2/7] GO Annotations (Human)..."

if [ ! -f "goa_human.gaf.gz" ]; then
    echo "  Downloading GO annotations..."
    curl -L -o goa_human.gaf.gz "http://geneontology.org/gene-associations/goa_human.gaf.gz"
    gunzip -k goa_human.gaf.gz 2>/dev/null || true
    echo "  Downloaded: goa_human.gaf"
else
    echo "  Already exists: goa_human.gaf.gz"
fi

# -----------------------------------------------------------------------------
# SFARI Genes (Required)
# -----------------------------------------------------------------------------
echo ""
echo "[3/7] SFARI Gene Database..."

echo "  NOTE: SFARI genes require manual download from https://gene.sfari.org/"
echo "  Please download and place in: $RAW_DIR/SFARI-Gene_genes.csv"

if [ -f "SFARI-Gene_genes.csv" ]; then
    echo "  Found: SFARI-Gene_genes.csv"
else
    echo "  NOT FOUND: Please download manually"
fi

# -----------------------------------------------------------------------------
# gnomAD Constraint Scores (Required)
# -----------------------------------------------------------------------------
echo ""
echo "[4/7] gnomAD Constraint Scores..."

if [ ! -f "gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz" ]; then
    echo "  Downloading gnomAD constraints..."
    curl -L -o gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz \
        "https://storage.googleapis.com/gcp-public-data--gnomad/release/2.1.1/constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz"
    gunzip -k gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz 2>/dev/null || \
        zcat gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz > gnomad.v2.1.1.lof_metrics.by_gene.txt
    echo "  Downloaded: gnomad constraints"
else
    echo "  Already exists: gnomad constraints"
fi

# -----------------------------------------------------------------------------
# Reactome Pathways (Required)
# -----------------------------------------------------------------------------
echo ""
echo "[5/7] Reactome Pathways..."

if [ ! -f "ReactomePathways.gmt" ]; then
    echo "  Downloading Reactome GMT..."
    curl -L -o ReactomePathways.gmt \
        "https://reactome.org/download/current/ReactomePathways.gmt.zip"
    unzip -o ReactomePathways.gmt.zip 2>/dev/null || \
        mv ReactomePathways.gmt.zip ReactomePathways.gmt
    echo "  Downloaded: ReactomePathways.gmt"
else
    echo "  Already exists: ReactomePathways.gmt"
fi

# -----------------------------------------------------------------------------
# STRING PPI Network (Full mode only - large file)
# -----------------------------------------------------------------------------
echo ""
echo "[6/7] STRING PPI Network..."

if [ "$DOWNLOAD_MODE" == "--all" ]; then
    if [ ! -f "9606.protein.links.v12.0.txt.gz" ]; then
        echo "  Downloading STRING PPI (this may take a while, ~300MB)..."
        curl -L -o 9606.protein.links.v12.0.txt.gz \
            "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
        gunzip -k 9606.protein.links.v12.0.txt.gz
        echo "  Downloaded: STRING PPI network"
    else
        echo "  Already exists: STRING PPI"
    fi
else
    echo "  Skipping STRING PPI (use --all to download, ~300MB)"
fi

# -----------------------------------------------------------------------------
# BrainSpan Expression (Full mode only - requires registration)
# -----------------------------------------------------------------------------
echo ""
echo "[7/7] BrainSpan Expression Data..."

echo "  NOTE: BrainSpan requires registration at https://www.brainspan.org/"
echo "  Download 'RNA-Seq Gencode v10 summarized to genes' and place in:"
echo "  $RAW_DIR/brainspan/"

mkdir -p brainspan

if [ -d "brainspan" ] && [ "$(ls -A brainspan 2>/dev/null)" ]; then
    echo "  Found data in brainspan/"
else
    echo "  NOT FOUND: Please download manually after registration"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Download Summary"
echo "=========================================="
echo ""
echo "Downloaded files:"
ls -lh "$RAW_DIR" 2>/dev/null | grep -v "^total" | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo "Manual downloads required:"
echo "  - SFARI genes: https://gene.sfari.org/"
echo "  - BrainSpan: https://www.brainspan.org/"
echo ""

if [ "$DOWNLOAD_MODE" == "--minimal" ]; then
    echo "Run with --all for complete database download (includes STRING PPI)"
fi
echo ""
