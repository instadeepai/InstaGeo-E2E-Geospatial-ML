# InstaGeo Backend

This is the backend service for the InstaGeo application, providing API endpoints for extracting and processing geospatial data and running model predictions.

## Architecture

**Task-based system** with two stages:
1. **Data Processing**: Extract satellite data from bounding boxes
2. **Model Prediction**: Run models

Each task has a unique ID and progresses through stages automatically.

## Quick Start

### Using Docker (Recommended)

```bash
cd instageo/new_apps
docker-compose -f docker-compose.dev.yml up -d
```

### Manual Setup

```bash
cd backend
pip install -r requirements.txt
python -m app.main
```

## Services

- **API Server**: Port 8000 - REST endpoints
- **Redis**: Port 6379 - Job queues & tasks storage
- **Data Processing Workers**: 2 replicas
- **Model Prediction Workers**: 2 replicas
- **RQ Dashboard**: Port 9181 - Monitor queues

## API Endpoints

### Create Task
```bash
POST /api/run-model
{
  "bounding_boxes": [
    {
      "coordinates": [[116.0, 39.0, 116.5, 39.5]],
      "date": "2024-01-01"
    }
  ],
  "parameters": {"test": true}
}
```

### Check Task Status
```bash
GET /api/task/{task_id}
```

### Get All Tasks
```bash
GET /api/tasks
```
Returns a list of all tasks with their status, creation date, bounding box count, model type, and stage details.

### Health Check
```bash
GET /api/health
```

### Queue Status
```bash
GET /api/queues/status
```

## Task Status Values

- `data_processing`: Task is processing satellite data
- `model_prediction`: Task is running model predictions
- `completed`: Task finished successfully
- `failed`: Task encountered an error

## Tentative environment Variables to configure

```env
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
DATA_PROCESSING_WORKER_REPLICAS=2
MODEL_PREDICTION_WORKER_REPLICAS=2
```
