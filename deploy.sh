#!/bin/bash

# Configuration defaults (overridden by .env)
PROJECT_ID="your-project-id"
SERVICE_NAME="breakout-game"
REGION="asia-east1"

# Load environment variables if .env file exists
if [ -f .env ]; then
  source .env
  # Override with env vars mapping if they exist
  [ ! -z "$GCP_PROJECT_ID" ] && PROJECT_ID="$GCP_PROJECT_ID"
  [ ! -z "$GCP_SERVICE_NAME" ] && SERVICE_NAME="$GCP_SERVICE_NAME"
  [ ! -z "$GCP_REGION" ] && REGION="$GCP_REGION"
fi

echo "========================================================"
echo "🚀 Deploying to Google Cloud Run"
echo "Project: $PROJECT_ID"
echo "Service: $SERVICE_NAME ($REGION)"
echo "========================================================"

# Make sure we're on the right project
gcloud config set project $PROJECT_ID

# Deploy from source using Cloud Build & Cloud Run
gcloud run deploy $SERVICE_NAME \
  --source . \
  --region $REGION \
  --allow-unauthenticated \
  --port 80

echo "✅ Deployment initialization finished!"
