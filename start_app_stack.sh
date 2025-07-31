#!/bin/bash

# InstaGeo Full-Stack Application Startup Script

echo "🚀 Starting InstaGeo Full-Stack Application..."


#TODO: Add a check for the environment and use the appropriate compose file
COMPOSE_FILE="instageo/new_apps/docker-compose.dev.yml"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if config.env exists in backend
if [ ! -f "instageo/new_apps/backend/config.env" ]; then
    echo "⚠️  instageo/new_apps/backend/config.env not found. Creating from example..."
    cp instageo/new_apps/backend/config.env.example instageo/new_apps/backend/config.env
    echo "📝 Please edit instageo/new_apps/backend/config.env with your specific settings before running again."
    exit 1
fi

# Load environment variables from backend config
set -a
source instageo/new_apps/backend/config.env
set +a

# Enable Docker Compose Bake for better build performance
export COMPOSE_BAKE=true

# Build and start services
echo "🔨 Building and starting services with $COMPOSE_FILE..."
docker-compose -f $COMPOSE_FILE build --no-cache
docker-compose -f $COMPOSE_FILE up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 15

# Check service status
echo "📊 Service Status:"
docker-compose -f $COMPOSE_FILE ps

echo ""
echo "✅ InstaGeo Full-Stack Application is running!"
echo ""
echo "🌐 Frontend: http://localhost"
echo "🔧 Backend API: http://localhost/api"
echo "📊 RQ Dashboard: http://localhost:9181"
echo "📚 API Documentation: ???"
echo ""
echo "📋 Useful commands:"
echo "  View logs: docker-compose -f $COMPOSE_FILE logs -f"
echo "  Stop services: docker-compose -f $COMPOSE_FILE down"
echo "  Restart services: docker-compose -f $COMPOSE_FILE restart"
echo "  Scale data workers: docker-compose -f $COMPOSE_FILE up -d --scale instageo-backend-data-processing-worker=4"
echo "  Scale model workers: docker-compose -f $COMPOSE_FILE up -d --scale instageo-backend-model-prediction-worker=4"
echo ""
