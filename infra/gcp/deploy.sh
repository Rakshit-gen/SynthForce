#!/bin/bash
# =============================================================================
# GCP Cloud Run Deployment Script
# =============================================================================
# This script deploys the Synthetic Workforce Simulator to GCP Cloud Run
#
# Prerequisites:
#   - Google Cloud SDK (gcloud) installed and configured
#   - Docker installed and running
#   - Appropriate GCP permissions
#
# Usage:
#   ./deploy.sh [project-id] [region]
#
# Example:
#   ./deploy.sh my-gcp-project us-central1
# =============================================================================

set -e

# Configuration
PROJECT_ID=${1:-$(gcloud config get-value project)}
REGION=${2:-us-central1}
SERVICE_NAME="workforce-simulator"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check if authenticated
    if ! gcloud auth print-access-token &> /dev/null; then
        log_error "Not authenticated with GCP. Please run 'gcloud auth login'."
        exit 1
    fi
    
    log_info "All prerequisites met."
}

# Enable required APIs
enable_apis() {
    log_info "Enabling required GCP APIs..."
    
    gcloud services enable \
        run.googleapis.com \
        containerregistry.googleapis.com \
        secretmanager.googleapis.com \
        cloudbuild.googleapis.com \
        --project=${PROJECT_ID}
    
    log_info "APIs enabled."
}

# Create secrets if they don't exist
create_secrets() {
    log_info "Setting up secrets..."
    
    # Check if .env file exists
    if [ -f "../../.env" ]; then
        source ../../.env
    else
        log_warn ".env file not found. Please provide secrets manually."
        read -p "Enter Groq API Key: " GROQ_API_KEY
        read -p "Enter Database URL: " DATABASE_URL
        read -p "Enter Redis URL: " REDIS_URL
    fi
    
    # Create secrets if they don't exist
    if ! gcloud secrets describe groq-api-key --project=${PROJECT_ID} &> /dev/null; then
        echo -n "${GROQ_API_KEY}" | gcloud secrets create groq-api-key \
            --data-file=- \
            --project=${PROJECT_ID}
    else
        echo -n "${GROQ_API_KEY}" | gcloud secrets versions add groq-api-key \
            --data-file=- \
            --project=${PROJECT_ID}
    fi
    
    if ! gcloud secrets describe database-url --project=${PROJECT_ID} &> /dev/null; then
        echo -n "${DATABASE_URL}" | gcloud secrets create database-url \
            --data-file=- \
            --project=${PROJECT_ID}
    else
        echo -n "${DATABASE_URL}" | gcloud secrets versions add database-url \
            --data-file=- \
            --project=${PROJECT_ID}
    fi
    
    if ! gcloud secrets describe redis-url --project=${PROJECT_ID} &> /dev/null; then
        echo -n "${REDIS_URL}" | gcloud secrets create redis-url \
            --data-file=- \
            --project=${PROJECT_ID}
    else
        echo -n "${REDIS_URL}" | gcloud secrets versions add redis-url \
            --data-file=- \
            --project=${PROJECT_ID}
    fi
    
    log_info "Secrets configured."
}

# Create service account
create_service_account() {
    log_info "Setting up service account..."
    
    SA_NAME="${SERVICE_NAME}-sa"
    SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
    
    # Create service account if it doesn't exist
    if ! gcloud iam service-accounts describe ${SA_EMAIL} --project=${PROJECT_ID} &> /dev/null; then
        gcloud iam service-accounts create ${SA_NAME} \
            --display-name="Workforce Simulator Service Account" \
            --project=${PROJECT_ID}
    fi
    
    # Grant secret accessor role
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/secretmanager.secretAccessor" \
        --quiet
    
    log_info "Service account configured: ${SA_EMAIL}"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building Docker image..."
    
    # Configure Docker to use gcloud as credential helper
    gcloud auth configure-docker --quiet
    
    # Build image
    docker build -t ${IMAGE_NAME}:latest --target production ../../
    
    # Push image
    log_info "Pushing image to Container Registry..."
    docker push ${IMAGE_NAME}:latest
    
    log_info "Image pushed: ${IMAGE_NAME}:latest"
}

# Deploy to Cloud Run
deploy_cloud_run() {
    log_info "Deploying to Cloud Run..."
    
    SA_EMAIL="${SERVICE_NAME}-sa@${PROJECT_ID}.iam.gserviceaccount.com"
    
    gcloud run deploy ${SERVICE_NAME} \
        --image=${IMAGE_NAME}:latest \
        --region=${REGION} \
        --platform=managed \
        --allow-unauthenticated \
        --memory=2Gi \
        --cpu=2 \
        --min-instances=1 \
        --max-instances=10 \
        --timeout=300 \
        --concurrency=80 \
        --port=8000 \
        --service-account=${SA_EMAIL} \
        --set-env-vars="ENVIRONMENT=production,LOG_LEVEL=INFO,WORKERS=4" \
        --set-secrets="GROQ_API_KEY=groq-api-key:latest,DATABASE_URL=database-url:latest,REDIS_URL=redis-url:latest" \
        --project=${PROJECT_ID}
    
    log_info "Deployment complete."
}

# Get service URL
get_service_url() {
    log_info "Service URL:"
    
    gcloud run services describe ${SERVICE_NAME} \
        --region=${REGION} \
        --platform=managed \
        --project=${PROJECT_ID} \
        --format="value(status.url)"
}

# Main deployment flow
main() {
    log_info "Starting deployment to GCP Cloud Run..."
    log_info "Project: ${PROJECT_ID}"
    log_info "Region: ${REGION}"
    
    check_prerequisites
    enable_apis
    create_secrets
    create_service_account
    build_and_push_image
    deploy_cloud_run
    
    echo ""
    log_info "=============================================="
    log_info "Deployment Complete!"
    log_info "=============================================="
    get_service_url
}

# Run main function
main
