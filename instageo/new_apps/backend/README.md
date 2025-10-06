# InstaGeo Backend

This is the backend service for the InstaGeo application, providing API endpoints for extracting and processing geospatial data and running model predictions.

## Architecture

**Task-based system** with two stages:
1. **Data Processing**: Extract satellite data from bounding boxes
2. **Model Prediction**: Run models

Each task has a unique ID and progresses through stages automatically. Tasks are managed using Redis for persistence and RQ (Redis Queue) for job processing.

### Key Components

- **DataProcessor**: Proxy class that integrates with the `instageo.data.raster_chip_creator` pipeline
- **Task Management**: Redis-backed task tracking with stage-based progression
- **Job Queues**: Separate RQ queues for data processing and model prediction
- **Error Handling**: Comprehensive error tracking and recovery


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
  "bboxes": [
    [116.0, 39.0, 116.5, 39.5],
    [-74.1, 40.6, -73.9, 40.8],
    [2.2, 48.8, 2.4, 49.0]
  ],
  "model_type": "aod_estimation",
  "date": "2024-06-01",
  "chip_size": 256,
  "cloud_coverage": 15,
  "num_steps": 5,
  "data_source": "S2",
  "temporal_step": 10,
  "temporal_tolerance": 3
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
EARTHDATA_USERNAME=xxxxxxx
EARTHDATA_PASSWORD=xxxxxxx
DATA_FOLDER=/app/instageo-data
DATA_PROCESSING_WORKER_REPLICAS=1
MODEL_PREDICTION_WORKER_REPLICAS=1
```
