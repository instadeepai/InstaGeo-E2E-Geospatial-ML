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

### Starting the Application

1. **Configure Environment Variables**

   Copy the example environment file and update it with your settings:
   ```bash
   cp instageo/new_apps/config.env.example instageo/new_apps/config.env
   # Edit instageo/new_apps/config.env to set your configuration (e.g., credentials, ports, etc.)
   ```

2. **Launch the Full Stack**

   **Option 1: Using the deployment script (recommended)**
   ```bash
   # From project root directory
   export STAGE=dev
   ./scripts/deploy.sh --skip-registry-sync  # Quick local development

   # For production with Cloudflare tunnel
   export STAGE=prod
   ./scripts/deploy.sh --cloudflare
   ```

   The script will:
   - Load your environment variables
   - Build and start all Docker services
   - Set up Cloudflare tunnel (if `--cloudflare` flag is used)
   - Print useful commands for monitoring and scaling

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

Create `config.env` from `config.env.example`:

### Cloudflare Tunnel Setup

For production deployments, you can expose your application to the internet using Cloudflare Tunnels. This provides secure, encrypted connections without exposing ports or managing SSL certificates.

#### Prerequisites

Before running the deployment script, you must configure Cloudflare Tunnels:

1. **Install Cloudflare Tunnel CLI**:
   ```bash
   # macOS
   brew install cloudflare/cloudflare/cloudflared

   # Linux
   wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   sudo dpkg -i cloudflared.deb
   ```

2. **Authenticate with Cloudflare**:
   After creating your Cloudflare account, run this command:
   ```bash
   cloudflared login
   ```
   This will open your browser to authenticate with Cloudflare.

3. **Create a Tunnel**:
   ```bash
   cloudflared tunnel create instageo-tunnel
   ```
   This creates a tunnel and generates a credentials file at `~/.cloudflared/<tunnel_id>.json`.

4. **Configure DNS (DOMAIN_NAME should be defined in config.env)**:
   ```bash
   cloudflared tunnel route dns $CLOUDFLARE_TUNNEL_NAME $DOMAIN_NAME
   ```

5. **Get Tunnel Credentials**:
   ```bash
   # Find the tunnel ID
   cloudflared tunnel list

   # Copy the contents of the credentials file
   cat ~/.cloudflared/<tunnel_id>.json
   ```
   Copy the entire JSON content as value for the variable in your `config.env` file. You should have:
   ```bash
   CLOUDFLARE_TUNNEL_CREDS='{"AccountTag":"your-account-tag","TunnelSecret":"your-tunnel-secret","TunnelID":"your-tunnel-id"}'
   ```


#### Important Notes

- **Tunnel Credentials**: The JSON credentials file contains sensitive information (AccountTag, TunnelSecret, TunnelID). Keep it secure.
- **Credentials Format**: The `CLOUDFLARE_TUNNEL_CREDS` environment varioaablshould contain the entire JSON content from the `<tunnel_id>.json` file.
- **Domain Configuration**: Ensure your domain is properly configured in Cloudflare DNS.
- **Monitoring**: Monitor tunnel status in the Cloudflare dashboard.

## Deployment
Navigate at the root directory of the project and run
```bash
# Deploy with Cloudflare tunnel
./scripts/deploy.sh

# Deploy without Cloudflare tunnel
./scripts/deploy.sh --skip-cloudflare
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

## Detailed Features

### Frontend Features
- **Interactive Mapping**: Leaflet-based map interface with drawing tools for bounding box creation
- **Real-time Validation**: Area calculation and constraint validation
- **Task Management**: Comprehensive task monitoring with filtering and pagination
- **Dark/Light Theme**: Toggle between themes with consistent styling
- **Model Selection**: Dynamic model configuration with parameter adjustment
- **Progress Tracking**: Real-time status updates with detailed stage information
- **PDF Generation**: Export task results with charts and map visualizations

### Backend Architecture
- **Task-Based System**: Two-stage processing (data extraction + model prediction)
- **RESTful API**: FastAPI-based endpoints for task management
- **Job Queue Management**: Redis-backed queues with RQ (Redis Queue)
- **Worker Scaling**: Separate workers for data processing and model prediction
- **Model Registry**: Integrated model management and metadata system
- **TiTiler Integration**: Dynamic tile serving for geospatial data visualization

### Key Features
- **Containerized Deployment**: Docker Compose for easy setup and scaling
- **Production Ready**: Nginx reverse proxy, Cloudflare tunnel support
- **Monitoring Dashboard**: RQ Dashboard for queue and worker monitoring
- **Hot Reload**: Development mode with live code updates
- **Error Handling**: Comprehensive error tracking and recovery

## Using the Web Interface

1. **Draw Bounding Box**: Use the map interface to draw rectangular areas of interest
2. **Select Model**: Choose from available geospatial models (AOD estimation, flood detection, etc.)
3. **Configure Parameters**: Set date, cloud coverage, and processing parameters
4. **Submit Task**: Start data processing and model prediction
5. **Monitor Progress**: Track task status in real-time
6. **View Results**: Visualize predictions on the map and download PDF reports

## API Usage

### Task Management Endpoints

#### Create Task
```bash
POST /api/run-model
{
  "bboxes": [
    [116.0, 39.0, 116.5, 39.5],
    [-74.1, 40.6, -73.9, 40.8]
  ],
  "model_key": "aod_estimation",
  "model_size": "tiny",
  "date": "2024-06-01",
  "cloud_coverage": 15,
  "temporal_tolerance": 3
}
```

#### Check Task Status
```bash
GET /api/task/{task_id}
```

#### Get All Tasks
```bash
GET /api/tasks
```

### Health & Monitoring
```bash
GET /api/health        # Health check
GET /api/queues/status # Queue status
```

## Task System Details

- When you submit a task via `/api/run-model`, the backend creates a new task with a unique ID and enqueues a data processing job
- When data processing completes, a model prediction job is automatically enqueued and linked to the same task
- Task status and results are stored in Redis and can be queried at any time

**Task Status Values:**
- `data_processing`: Task is processing satellite data
- `model_prediction`: Task is running model predictions
- `completed`: Task finished successfully
- `failed`: Task encountered an error

## Performance and Scaling

### Worker Configuration
Configure worker replicas in `config.env`:
```bash
DATA_PROCESSING_WORKER_REPLICAS=1
MODEL_PREDICTION_WORKER_REPLICAS=1
VISUALIZATION_PREPARATION_WORKER_REPLICAS=1
```

### Scaling Commands
```bash
# Scale data processing workers
docker compose up -d --scale instageo-backend-data-processing-worker=4

# Scale model prediction workers
docker compose up -d --scale instageo-backend-model-prediction-worker=4
```
