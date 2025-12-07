#!/bin/bash
# =============================================================================
# AWS ECS Deployment Script
# =============================================================================
# This script deploys the Synthetic Workforce Simulator to AWS ECS
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - Docker installed and running
#   - jq installed for JSON parsing
#
# Usage:
#   ./deploy.sh [environment] [region]
#
# Example:
#   ./deploy.sh production us-east-1
# =============================================================================

set -e

# Configuration
ENVIRONMENT=${1:-production}
REGION=${2:-us-east-1}
APP_NAME="workforce-simulator"
STACK_NAME="${ENVIRONMENT}-${APP_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
    
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        log_error "jq is not installed. Please install it first."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Please run 'aws configure'."
        exit 1
    fi
    
    log_info "All prerequisites met."
}

# Get AWS account ID
get_account_id() {
    aws sts get-caller-identity --query Account --output text
}

# Create ECR repository if it doesn't exist
create_ecr_repository() {
    log_info "Creating ECR repository..."
    
    aws ecr describe-repositories --repository-names ${APP_NAME} --region ${REGION} &> /dev/null || \
        aws ecr create-repository \
            --repository-name ${APP_NAME} \
            --region ${REGION} \
            --image-scanning-configuration scanOnPush=true
    
    log_info "ECR repository ready."
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building Docker image..."
    
    ACCOUNT_ID=$(get_account_id)
    ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${APP_NAME}"
    IMAGE_TAG="${ECR_URI}:${ENVIRONMENT}-$(date +%Y%m%d%H%M%S)"
    LATEST_TAG="${ECR_URI}:${ENVIRONMENT}-latest"
    
    # Login to ECR
    aws ecr get-login-password --region ${REGION} | \
        docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com
    
    # Build image
    docker build -t ${IMAGE_TAG} -t ${LATEST_TAG} --target production ../../
    
    # Push image
    log_info "Pushing image to ECR..."
    docker push ${IMAGE_TAG}
    docker push ${LATEST_TAG}
    
    log_info "Image pushed: ${IMAGE_TAG}"
    echo ${LATEST_TAG}
}

# Deploy CloudFormation stack
deploy_stack() {
    local IMAGE_URI=$1
    
    log_info "Deploying CloudFormation stack..."
    
    # Check if .env file exists for secrets
    if [ ! -f "../../.env" ]; then
        log_warn ".env file not found. Please provide parameters manually."
        read -p "Enter Groq API Key: " GROQ_API_KEY
        read -p "Enter Database URL: " DATABASE_URL
        read -p "Enter Redis URL: " REDIS_URL
    else
        source ../../.env
    fi
    
    # Deploy or update stack
    aws cloudformation deploy \
        --template-file cloudformation.yaml \
        --stack-name ${STACK_NAME} \
        --region ${REGION} \
        --parameter-overrides \
            EnvironmentName=${ENVIRONMENT} \
            ContainerImage=${IMAGE_URI} \
            GroqApiKey=${GROQ_API_KEY} \
            DatabaseUrl=${DATABASE_URL} \
            RedisUrl=${REDIS_URL} \
        --capabilities CAPABILITY_IAM \
        --no-fail-on-empty-changeset
    
    log_info "Stack deployment complete."
}

# Get stack outputs
get_stack_outputs() {
    log_info "Stack outputs:"
    
    aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME} \
        --region ${REGION} \
        --query "Stacks[0].Outputs" \
        --output table
}

# Main deployment flow
main() {
    log_info "Starting deployment to ${ENVIRONMENT} in ${REGION}..."
    
    check_prerequisites
    create_ecr_repository
    
    IMAGE_URI=$(build_and_push_image)
    
    deploy_stack ${IMAGE_URI}
    
    get_stack_outputs
    
    log_info "Deployment complete!"
    log_info "Your application will be available at the LoadBalancerURL shown above."
}

# Run main function
main
