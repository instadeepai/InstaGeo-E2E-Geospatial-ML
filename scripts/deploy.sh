#!/bin/bash

# Usage: ./deploy.sh [--registry-sync-only] [--skip-registry-sync] [--cloudflare]
# --registry-sync-only: Only sync model registry
# --skip-registry-sync: Skip model registry sync
# --cloudflare: Enable Cloudflare Tunnel creation

# Exit on error
set -e

echo "🚀 Starting InstaGeo deployment..."
# Parse arguments
REGISTRY_SYNC_ONLY=false
SKIP_REGISTRY_SYNC=false
SKIP_CLOUDFLARE=true
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --registry-sync-only) REGISTRY_SYNC_ONLY=true ;;
        --skip-registry-sync) SKIP_REGISTRY_SYNC=true ;;
        --cloudflare) SKIP_CLOUDFLARE=false ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Load environment variables from config.env
if [ ! -f "instageo/new_apps/config.env" ]; then
    echo "❌ config.env not found at instageo/new_apps/config.env"
    exit 1
fi

echo "🔑 Loading environment from config.env"
set -a
source instageo/new_apps/config.env
set +a

# Validate required environment variables
# If cloudflare is enabled, then CLOUDFLARE_TUNNEL_NAME, CLOUDFLARE_TUNNEL_CREDS and DOMAIN_NAME must be set
# If registry sync is enabled, then HOST_MODELS_PATH and MODELS_REGISTRY_GCS_URI must be set
if [ "$SKIP_CLOUDFLARE" = false ]; then
    if [ -z "$CLOUDFLARE_TUNNEL_NAME" ] || [ -z "$CLOUDFLARE_TUNNEL_CREDS" ] || [ -z "$DOMAIN_NAME" ]; then
        echo "Error: CLOUDFLARE_TUNNEL_NAME, CLOUDFLARE_TUNNEL_CREDS and DOMAIN_NAME must be set in config.env"
        exit 1
    fi
fi
if [ "$SKIP_REGISTRY_SYNC" = false ]; then
    if [ -z "$HOST_MODELS_PATH" ] || [ -z "$MODELS_REGISTRY_GCS_URI" ]; then
        echo "Error: HOST_MODELS_PATH and MODELS_REGISTRY_GCS_URI must be set in config.env"
        exit 1
    fi
fi

# Sync model registry if not skipped
if [ "$SKIP_REGISTRY_SYNC" = false ]; then
    echo "🔄 Syncing model registry..."
    mkdir -p "$HOST_MODELS_PATH"
    cd instageo/model/registry
    ./model_registry_sync.sh "$MODELS_REGISTRY_GCS_URI" "$HOST_MODELS_PATH"

    if [ "$REGISTRY_SYNC_ONLY" = true ]; then
        echo "✅ Model registry sync completed successfully!"
        exit 0
    fi
    cd ../../..
fi

# Start application stack
echo "🚀 Starting application stack..."
if [ "$SKIP_CLOUDFLARE" = true ]; then
    export DOMAIN_NAME="localhost"
fi
./scripts/start_app_stack.sh
echo "✅ Deployment completed successfully!"
