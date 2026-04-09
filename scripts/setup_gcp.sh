#!/bin/bash
# =============================================================
# GCP Setup Script
#
# Enables required APIs, sets up authentication, and deploys
# infrastructure using Terraform.
#
# Usage:
#   ./scripts/setup_gcp.sh <PROJECT_ID>
# =============================================================

set -e

PROJECT_ID=${1:?"Usage: $0 <PROJECT_ID>"}

echo "============================================"
echo "  GCP Setup for Cloud Cost Optimization"
echo "  Project: $PROJECT_ID"
echo "============================================"

# Step 1: Set project
echo ""
echo "[1/5] Setting GCP project..."
gcloud config set project "$PROJECT_ID"

# Step 2: Enable required APIs
echo ""
echo "[2/5] Enabling required GCP APIs..."
gcloud services enable compute.googleapis.com \
    monitoring.googleapis.com \
    storage.googleapis.com \
    cloudresourcemanager.googleapis.com \
    --project="$PROJECT_ID"

echo "  APIs enabled."

# Step 3: Create service account for ML pipeline
echo ""
echo "[3/5] Creating service account..."
SA_NAME="cloud-cost-opt-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" &>/dev/null; then
    echo "  Service account already exists: $SA_EMAIL"
else
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="Cloud Cost Optimization Service Account" \
        --project="$PROJECT_ID"
    echo "  Created: $SA_EMAIL"
fi

# Grant roles
for ROLE in roles/monitoring.viewer roles/compute.admin roles/storage.admin; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SA_EMAIL" \
        --role="$ROLE" --quiet
done
echo "  Roles assigned."

# Step 4: Download key
echo ""
echo "[4/5] Creating service account key..."
KEY_FILE="gcp-key.json"
if [ ! -f "$KEY_FILE" ]; then
    gcloud iam service-accounts keys create "$KEY_FILE" \
        --iam-account="$SA_EMAIL" --project="$PROJECT_ID"
    echo "  Key saved to $KEY_FILE"
    echo "  Set GOOGLE_APPLICATION_CREDENTIALS=$KEY_FILE"
else
    echo "  Key file already exists: $KEY_FILE"
fi

# Step 5: Deploy with Terraform
echo ""
echo "[5/5] Deploying infrastructure with Terraform..."
cd terraform

if [ ! -f "terraform.tfvars" ]; then
    echo "project_id = \"$PROJECT_ID\"" > terraform.tfvars
    echo "  Created terraform.tfvars"
fi

terraform init
terraform plan -out=tfplan
echo ""
echo "Review the plan above. To apply:"
echo "  cd terraform && terraform apply tfplan"
echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. cd terraform && terraform apply tfplan"
echo "  2. export GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json"
echo "  3. export GCP_PROJECT_ID=$PROJECT_ID"
echo "  4. ./scripts/generate_load.sh $PROJECT_ID us-central1-a workload-mig 30"
echo "  5. python main.py --mode gcp"
echo "============================================"
