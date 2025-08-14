# InstaGeo Full-Stack Application

A complete geospatial analysis platform with React frontend and FastAPI backend, featuring a two-stage task system (data extraction and model prediction) with Redis-backed job queues.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React         │    │   Nginx         │    │   FastAPI       │    │   Redis         │
│   Frontend      │◄──►│   Reverse Proxy │◄──►│   Backend       │◄──►│   Queue         │
│   (Port 3000)   │    │   (Port 80)     │    │   (Port 8000)   │    │   (Port 6379)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                                              │
                                                                              ▼
                                                                     ┌─────────────────┐
                                                                     │   RQ Workers    │
                                                                     │   (Data + Model)│
                                                                     └─────────────────┘
```

## Features

### Frontend
- Interactive Leaflet map with drawing tools
- Bounding box creation and editing
- Real-time area validation
- Model parameter configuration
- Task submission and progress tracking

### Backend
- Task-based architecture (each task has two jobs: data processing and model prediction)
- RESTful API endpoints
- Job queue management with Redis
- Worker scaling capabilities
- RQ Dashboard for monitoring

## Quick Start

### Prerequisites
- Docker and Docker Compose

### Start the Application
```bash
cd instageo/new_apps
./start.sh
```

The application will be available at:
- **Frontend**: http://localhost
- **Backend API**: http://localhost/api
- **RQ Dashboard**: http://localhost:9181

## Services

| Service | Port | Purpose | Development |
|---------|------|---------|-------------|
| Frontend | 3000 | React App | Hot reload |
| Backend API | 8000 | FastAPI | Hot reload |
| Redis | 6379 | Queue Broker | Persisted |
| RQ Dashboard | 9181 | Queue Monitor | Available |
| Data Workers | - | Data Processing | 2 replicas |
| Model Workers | - | Model Prediction | 2 replicas |
| **Nginx** | 80 | Reverse Proxy (routes /api to backend, / to frontend) | Available |

## API Endpoints

### Task Management
- `POST /api/run-model` - Submit a new task (data processing + model prediction)
- `GET /api/task/{task_id}` - Get status and results of a task
- `GET /api/queues/status` - Monitor queue status

### Health & Monitoring
- `GET /api/health` - Health check
- `GET /` - API root

## Task System

- When you submit a task via `/api/run-model`, the backend creates a new task with a unique ID and enqueues a data processing job.
- When data processing completes, a model prediction job is automatically enqueued and linked to the same task.
- Task status and results are stored in Redis and can be queried at any time.

**Note:**
- Redis does not accept `None` values. All fields in task metadata are stored as strings. Empty fields are stored as `""` (empty string) and converted back to `null`/`None` in API responses.


## Configuration

### Environment Variables

Create `backend/config.env` from `backend/config.env.example`:

```env
# Redis Configuration
REDIS_HOST=instageo-redis
REDIS_PORT=6379
REDIS_DB=0

# EarthData credentials
EARTHDATA_USERNAME=xxxxxxx
EARTHDATA_PASSWORD=xxxxxxx

# Storage folder
DATA_FOLDER=/app/instageo-data

# Worker Configuration
DATA_PROCESSING_WORKER_REPLICAS=2
MODEL_PREDICTION_WORKER_REPLICAS=2
VISUALIZATION_PREPARATION_WORKER_REPLICAS=1


# Model Configuration
MODEL_CHECKPOINT_PATH=/path/to/model/checkpoint.ckpt
DATA_STORAGE_PATH=/path/to/data/storage

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### Frontend Configuration

Create `frontend/.env` from `frontend/env.example`:

```env
REACT_APP_INSTAGEO_BACKEND_API_BASE_URL=http://localhost
```

## Development
### Docker Development
Navigate at the root directory of the project and run
```bash
# Start all services
./start_app_stack.sh
```

## Monitoring

### RQ Dashboard
Access at http://localhost:9181 to:
- Monitor job queues
- View worker status
- Check job history
- Restart failed jobs

### Useful Commands (from the root directory of the project)

```bash
# View all service logs
docker-compose -f instageo/new_apps/docker-compose.dev.yml logs -f

# View specific service logs
docker-compose -f instageo/new_apps/docker-compose.dev.yml logs -f instageo-backend-api
docker-compose -f instageo/new_apps/docker-compose.dev.yml logs -f instageo-frontend

# Stop services
docker-compose -f instageo/new_apps/docker-compose.dev.yml down

# Restart all services
docker-compose -f instageo/new_apps/docker-compose.dev.yml restart

# Restart specific service
docker-compose -f instageo/new_apps/docker-compose.dev.yml restart instageo-backend-api

# Scale data processing workers
docker-compose -f instageo/new_apps/docker-compose.dev.yml up -d --scale instageo-backend-data-processing-worker=4

# Scale model prediction workers
docker-compose -f instageo/new_apps/docker-compose.dev.yml up -d --scale instageo-backend-model-prediction-worker=4

# Rebuild and restart services
docker-compose -f instageo/new_apps/docker-compose.dev.yml up -d --build

# Check service status
docker-compose -f instageo/new_apps/docker-compose.dev.yml ps
```


## Troubleshooting

### Common Issues

1. **Port conflicts**
   - Check if ports 3000, 8000, 6379, 9181 are available
   - Stop conflicting services

2. **Build failures**
   - Clear Docker cache: `docker system prune -a`
   - Rebuild: `docker-compose build --no-cache`

3. **Worker not processing jobs**
   - Check Redis connection
   - Verify queue names match
   - Check worker logs

4. **Redis DataError: NoneType**
   - This is now fixed: all fields are stored as strings, never as `None`.
   - If you see this error, ensure you are not passing `None` to Redis in your own code.


## Project Structure

```
InstaGeo/                           # Root project directory
├── start_app_stack.sh              # Main startup script for app
├── instageo/                       # Core InstaGeo package
│   ├── data/                       # Data processing modules
│   │   └── ...
│   ├── model/                      # Model implementations
│   │   └── ...
│   └── new_apps/                   # Full-stack application
│       ├── backend/                # FastAPI backend
│       │   ├── app/               # Application code
│       │   │   ├── main.py        # FastAPI application
│       │   │   ├── tasks.py       # Task management
│       │   │   ├── jobs.py        # Job queue handling
│       │   │   ├── data_processor.py  # Data pipeline integration
│       │   │   └── __init__.py    # Package initialization
│       │   ├── tests/             # Backend tests
│       │   │   └── test_api.py    # API endpoint tests
│       │   ├── Dockerfile         # Backend container
│       │   ├── requirements.txt   # Python dependencies
│       │   ├── config.env.example # Environment template
│       │   └── README.md          # Backend documentation
│       ├── frontend/              # React frontend
│       │   ├── src/              # Source code
│       │   │   ├── components/   # React components
│       │   │   │   ├── BoundingBoxInfo.js
│       │   │   │   ├── ControlPanel.js
│       │   │   │   ├── MapComponent.js
│       │   │   │   ├── TaskResultPopup.js
│       │   │   │   └── TasksMonitor.js  # Task monitoring
│       │   │   ├── config.js     # Application configuration
│       │   │   ├── constants.js  # Shared constants
│       │   │   ├── index.js      # Application entry point
│       │   │   └── App.js        # Main application component
│       │   ├── public/           # Static files
│       │   ├── Dockerfile.dev    # Development container
│       │   ├── package.json      # Node dependencies
│       │   └── .env.example      # Environment template
│       ├── nginx.conf            # Reverse proxy configuration
│       ├── docker-compose.dev.yml # Development services
│       ├── restart-nginx-on-deps.sh # Nginx restart utility
│       └── README.md             # This file
├── experiments/                    # Research experiments
│   └── ...
├── tests/                          # Project tests
│   └── ...
└── ...                             # Other project files
```
