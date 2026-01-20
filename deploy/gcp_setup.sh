#!/bin/bash
# =============================================================================
# Google Cloud Platform Setup Script
# =============================================================================
# Usage: bash deploy/gcp_setup.sh
#
# This script sets up GCP resources for production deployment:
# 1. Creates Cloud Storage bucket
# 2. Creates Vertex AI Workbench instance
# 3. Uploads data to GCS
# 4. Configures IAM permissions

set -e

# -----------------------------------------------------------------------------
# Configuration (edit these or set via environment)
# -----------------------------------------------------------------------------
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"
BUCKET_NAME="${GCS_BUCKET:-${PROJECT_ID}-autism-pathway}"

echo "=========================================="
echo "GCP Production Setup"
echo "=========================================="
echo "Project:  $PROJECT_ID"
echo "Region:   $REGION"
echo "Zone:     $ZONE"
echo "Bucket:   $BUCKET_NAME"
echo "=========================================="
echo ""

read -p "Continue with these settings? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# -----------------------------------------------------------------------------
# Set active project
# -----------------------------------------------------------------------------
echo ""
echo "[1/6] Setting active project..."
gcloud config set project "$PROJECT_ID"
gcloud config set compute/region "$REGION"
gcloud config set compute/zone "$ZONE"

# -----------------------------------------------------------------------------
# Enable required APIs
# -----------------------------------------------------------------------------
echo ""
echo "[2/6] Enabling required APIs..."

gcloud services enable \
    compute.googleapis.com \
    storage.googleapis.com \
    aiplatform.googleapis.com \
    notebooks.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com

echo "  APIs enabled"

# -----------------------------------------------------------------------------
# Create Cloud Storage bucket
# -----------------------------------------------------------------------------
echo ""
echo "[3/6] Creating Cloud Storage bucket..."

if gcloud storage buckets describe "gs://$BUCKET_NAME" &>/dev/null; then
    echo "  Bucket already exists: gs://$BUCKET_NAME"
else
    gcloud storage buckets create "gs://$BUCKET_NAME" \
        --location="$REGION" \
        --uniform-bucket-level-access \
        --public-access-prevention
    echo "  Created bucket: gs://$BUCKET_NAME"
fi

# Create folder structure
echo "  Creating folder structure..."
echo "" | gcloud storage cp - "gs://$BUCKET_NAME/data/raw/.keep"
echo "" | gcloud storage cp - "gs://$BUCKET_NAME/data/processed/.keep"
echo "" | gcloud storage cp - "gs://$BUCKET_NAME/models/.keep"
echo "" | gcloud storage cp - "gs://$BUCKET_NAME/outputs/.keep"

# -----------------------------------------------------------------------------
# Create Artifact Registry for Docker images (optional)
# -----------------------------------------------------------------------------
echo ""
echo "[4/6] Creating Artifact Registry repository..."

if gcloud artifacts repositories describe autism-pathway --location="$REGION" &>/dev/null; then
    echo "  Repository already exists"
else
    gcloud artifacts repositories create autism-pathway \
        --repository-format=docker \
        --location="$REGION" \
        --description="Docker images for Autism Pathway Framework"
    echo "  Created repository: autism-pathway"
fi

# -----------------------------------------------------------------------------
# Create Vertex AI Workbench Instance
# -----------------------------------------------------------------------------
echo ""
echo "[5/6] Creating Vertex AI Workbench instance..."

INSTANCE_NAME="autism-pathway-workbench"

# Check if instance exists
if gcloud notebooks instances describe "$INSTANCE_NAME" --location="$ZONE" &>/dev/null; then
    echo "  Instance already exists: $INSTANCE_NAME"
else
    echo "  Creating Workbench instance (this may take a few minutes)..."

    gcloud notebooks instances create "$INSTANCE_NAME" \
        --location="$ZONE" \
        --machine-type="n1-standard-8" \
        --boot-disk-size="100GB" \
        --data-disk-size="100GB" \
        --no-public-ip \
        --metadata="proxy-mode=service_account" \
        --install-gpu-driver \
        --accelerator-type="NVIDIA_TESLA_T4" \
        --accelerator-core-count=1 \
        2>/dev/null || \
    gcloud notebooks instances create "$INSTANCE_NAME" \
        --location="$ZONE" \
        --machine-type="n1-standard-8" \
        --boot-disk-size="100GB" \
        --data-disk-size="100GB"

    echo "  Created instance: $INSTANCE_NAME"
fi

# -----------------------------------------------------------------------------
# Upload local data to GCS
# -----------------------------------------------------------------------------
echo ""
echo "[6/6] Syncing local data to GCS..."

if [ -d "data/raw" ]; then
    echo "  Uploading data/raw to gs://$BUCKET_NAME/data/raw/"
    gcloud storage rsync data/raw "gs://$BUCKET_NAME/data/raw/" --recursive
else
    echo "  No local data/raw directory found, skipping upload"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "GCP Setup Complete!"
echo "=========================================="
echo ""
echo "Resources created:"
echo "  - Storage bucket: gs://$BUCKET_NAME"
echo "  - Artifact Registry: $REGION-docker.pkg.dev/$PROJECT_ID/autism-pathway"
echo "  - Workbench instance: $INSTANCE_NAME"
echo ""
echo "Next steps:"
echo ""
echo "1. Access Workbench:"
echo "   gcloud notebooks instances describe $INSTANCE_NAME --location=$ZONE"
echo "   Or visit: https://console.cloud.google.com/vertex-ai/workbench"
echo ""
echo "2. Clone repo in Workbench:"
echo "   git clone https://github.com/topmist-admin/autism-pathway-framework.git"
echo ""
echo "3. Install dependencies:"
echo "   pip install -e '.[all]'"
echo ""
echo "4. Download data from GCS:"
echo "   gcloud storage cp -r gs://$BUCKET_NAME/data/raw ./data/"
echo ""
