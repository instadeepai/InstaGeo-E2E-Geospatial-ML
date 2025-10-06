#!/bin/bash

# InstaGeo Full-Stack Application Startup Script

echo "🚀 Starting InstaGeo Full-Stack Application..."

COMPOSE_FILE="instageo/new_apps/docker-compose.${STAGE}.yml"

echo "🔧 Running in ${STAGE} mode"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "❌ $COMPOSE_FILE not found. Valid stages are: dev, prod"
    exit 1
fi

# Load environment variables from config.env
if [ ! -f "instageo/new_apps/config.env" ]; then
    echo "❌ config.env not found at instageo/new_apps/config.env"
    exit 1
fi

echo "🔑 Loading environment from config.env"
set -a
source instageo/new_apps/config.env
set +a

# Enable Docker Compose Bake for better build performance
export COMPOSE_BAKE=true

# Create htpasswd file for RQ dashboard in production
if [ "$STAGE" = "prod" ]; then
    echo "🔐 Creating RQ dashboard authentication..."
    docker run --rm httpd:alpine htpasswd -nb $RQ_DASHBOARD_USERNAME $RQ_DASHBOARD_PASSWORD > instageo/new_apps/.htpasswd
fi

# Build and start services
echo "🔨 Building and starting services with $COMPOSE_FILE..."

# Set profile based on Cloudflare flag and prepare credentials file for cloudflare tunnel
if [ "$SKIP_CLOUDFLARE" = "true" ]; then
    echo "⏭️ Skipping Cloudflare tunnel service..."
    COMPOSE_PROFILES=""
else
    echo "🌐 Starting all services including Cloudflare tunnel..."
    COMPOSE_PROFILES="cloudflare"
    echo "🔐 Creating Cloudflare tunnel credentials file..."
    echo $CLOUDFLARE_TUNNEL_CREDS > instageo/new_apps/cloudflared/credentials.json
    echo "🔐 Cloudflare tunnel credentials file created successfully!"
    echo "🔐 Creating Cloudflare tunnel config file from template: instageo/new_apps/cloudflared/config_template.yml"
    sed -e "s/\$CLOUDFLARE_TUNNEL_NAME/$CLOUDFLARE_TUNNEL_NAME/g" -e "s/\$DOMAIN_NAME/$DOMAIN_NAME/g" instageo/new_apps/cloudflared/config_template.yml > instageo/new_apps/cloudflared/config.yml
    echo "🔐 Cloudflare tunnel config file created successfully!"
fi

# Build and start services with appropriate profile
docker compose -f $COMPOSE_FILE --profile $COMPOSE_PROFILES build # --no-cache
docker compose -f $COMPOSE_FILE --profile $COMPOSE_PROFILES up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 15

# Check service status
echo "📊 Service Status:"
docker compose -f $COMPOSE_FILE --profile $COMPOSE_PROFILES ps

echo ""
echo "✅ InstaGeo Full-Stack Application is running!"
echo ""
echo "📋 Useful commands:"
echo "  View logs: docker compose -f $COMPOSE_FILE --profile $COMPOSE_PROFILES logs -f"
echo "  Stop services: docker compose -f $COMPOSE_FILE --profile $COMPOSE_PROFILES down"
echo "  Restart services: docker compose -f $COMPOSE_FILE --profile $COMPOSE_PROFILES restart"
echo "  Scale data workers: docker compose -f $COMPOSE_FILE up -d --scale instageo-backend-data-processing-worker=4"
echo "  Scale model workers: docker compose -f $COMPOSE_FILE up -d --scale instageo-backend-model-prediction-worker=4"
echo ""
