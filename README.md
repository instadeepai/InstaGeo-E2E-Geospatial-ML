<picture>
  <source srcset="assets/logo-dark.png" media="(prefers-color-scheme: dark)">
  <img src="assets/logo.png" alt="Logo">
</picture>

## Overview

InstaGeo is an end-to-end geospatial machine learning framework that automates data preprocessing, model training, inference and deployment, enabling seamless extraction of actionable insights from satellite imagery such [Harmonized Landsat and Sentinel-2 (HLS)](https://hls.gsfc.nasa.gov/) and [Sentinel-2](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-data/sentinel-2) and [Sentinel-1](https://sentinels.copernicus.eu/copernicus/sentinel-1).

It leverages the [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) geospatial foundational model and consists of three core components: Data, Model, and Apps, each tailored to support various aspects of geospatial data retrieval, manipulation, preprocessing, model training, and inference serving.

### Components

1. [**Data**](./instageo/data/README.md): Focuses on retrieving, manipulating, and processing satellite data for classification and segmentation tasks such as disaster mapping, crop classification, and breeding ground prediction. Supports HLS, Sentinel-2, and Sentinel-1 data sources with advanced data pipeline capabilities.

2. [**Model**](./instageo/model/README.md): Centers around data loading, training, and evaluating models, particularly leveraging the Prithvi model for various modeling tasks. Features include chip inference for optimized processing, model registry system, and Ray-based model serving capabilities.

3. [**Apps**](./instageo/new_apps/README.md): A geospatial analysis platform featuring interactive mapping, task-based processing, and real-time monitoring capabilities.
<div align="center">

![InstaGeo App](assets/instageo_app.gif)
</div>

## Paper and Key Results

📄 Paper: [InstaGeo: Compute-Efficient Geospatial Machine Learning from Data to Deployment](https://arxiv.org/abs/2510.05617)

| Task                                      | Model                 | Dataset                                                                 | GFM              | mIoU (std) | Acc    | mF1 (std)   | ROC-AUC (std) |
| ----------------------------------------- | --------------------- | ----------------------------------------------------------------------- | ---------------- | ---------- | ------ | ----------- | ------------- |
| Flood Mapping                             | Baseline              | [Original](https://github.com/cloudtostreet/Sen1Floods11)               | Prithvi-V1-100M  | 88.3 (0.3) | --     | 97.3 (0.1)  | --            |
| Flood Mapping                             | InstaGeo-Baseline     | [Original](https://github.com/cloudtostreet/Sen1Floods11)               | Prithvi-V1-100M  | 88.53      | 97.24  | 93.71       | 99.16         |
| Flood Mapping                             | InstaGeo-Replica (HLS)| [Replica (HLS)](https://console.cloud.google.com/storage/browser/instageo/data/sen1floods-hls-replica) | Prithvi-V1-100M  | 85.40      | 96.39  | 91.78       | 97.15         |
| Flood Mapping                             | InstaGeo-Replica (S2) | [Replica (S2)](https://console.cloud.google.com/storage/browser/instageo/data/sen1floods-s2-replica)   | Prithvi-V1-100M  | 87.80      | 97.07  | 93.26       | 97.61         |
| Multi-Temporal Crop Segmentation (US)     | Baseline              | [Original](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification)      | Prithvi-V1-100M  | 42.7       | 60.7   | --          | --            |
| Multi-Temporal Crop Segmentation (US)     | InstaGeo-Baseline     | [Original](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification)      | Prithvi-V1-100M  | 48.07      | 65.77  | 64.34       | 95.79         |
| Multi-Temporal Crop Segmentation (US)     | InstaGeo-Replica      | [Replica](https://console.cloud.google.com/storage/browser/instageo/data/multitemporal-crop-classification-replica)       | Prithvi-V1-100M  | 47.87      | 66.10  | 64.19       | 95.82         |
| Multi-Temporal Crop Segmentation (US)     | InstaGeo-Expanded (2022, 14k) | [InstaGeo-US-CDL-2022-14k](https://console.cloud.google.com/storage/browser/instageo/data/multitemporal-crop-segmentation-US-CDL-2022) | Prithvi-V2-300M  | 60.65      | 83.02  | 73.46       | 97.99         |
| Multi-Temporal Crop Segmentation (US)     | InstaGeo-2024 (18k)   | [InstaGeo-US-CDL-2024-18k](https://console.cloud.google.com/storage/browser/instageo/data/multitemporal-crop-segmentation-US-CDL-2024) | Prithvi-V2-300M  | 54.86      | 83.30  | 67.19       | 97.96         |
| Locust Breeding Ground Prediction         | Baseline              | [Original](https://console.cloud.google.com/storage/browser/instageo/data/locust_breeding)               | Prithvi-V1-100M  | --         | 83.03  | 81.53       | --            |
| Locust Breeding Ground Prediction         | InstaGeo-Baseline     | [Original](https://console.cloud.google.com/storage/browser/instageo/data/locust_breeding)               | Prithvi-V1-100M  | 71.51      | 83.39  | 83.39       | 86.74         |
| Locust Breeding Ground Prediction         | InstaGeo-Replica      | [Replica](https://console.cloud.google.com/storage/browser/instageo/data/locust-replica)                 | Prithvi-V1-100M  | 73.30      | 84.60  | 84.60       | 88.66         |

For task-specific details and pretrained models, see the [Model component documentation](./instageo/model/README.md).

## Installation

InstaGeo uses modern Python dependency management with [uv](https://docs.astral.sh/uv/) for fast, reliable package installation.

### Prerequisites
- Python 3.11+ (required)
- Docker and Docker Compose (for full-stack application)

### Install uv Package Manager
If you don't have uv installed, install it using one of these methods:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Install InstaGeo

#### Option 1: Using uv (Recommended)
```bash
# Clone and navigate to the project
cd InstaGeo

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # (Linux/macOS)
.venv\Scripts\activate     # (Windows)

# Install dependencies (--locked ensures reproducible builds)
# For CPU-only PyTorch (recommended for most users)
uv sync --locked --extra all --extra dev --extra cpu

# For GPU-enabled PyTorch (Linux only, requires CUDA)
uv sync --locked --extra all --extra dev --extra gpu
```

#### Option 2: Using pip
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # (Linux/macOS)
.venv\Scripts\activate     # (Windows)

# Install directly from GitHub repository
# For CPU-only PyTorch (recommended for most users)
pip install "git+https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML.git#egg=InstaGeo[all,cpu]"

# For GPU-enabled PyTorch (Linux only, requires CUDA)
pip install "git+https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML.git#egg=InstaGeo[all,gpu]"

# For development (includes dev tools)
pip install "git+https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML.git#egg=InstaGeo[all,dev,cpu]"

# Install from specific branch (e.g., develop)
pip install "git+https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML.git@develop#egg=InstaGeo[all,cpu]"

# Install from specific tag/release
pip install "git+https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML.git@v0.1.0#egg=InstaGeo[all,cpu]"

```

#### Option 3: Using pip (Local Development)
```bash
# Clone and navigate to the project (for local development)
git clone https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML.git
cd InstaGeo-E2E-Geospatial-ML

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # (Linux/macOS)
.venv\Scripts\activate     # (Windows)

# Install in editable mode for development
# For CPU-only PyTorch (recommended for most users)
pip install -e ".[all,dev,cpu]"

# For GPU-enabled PyTorch (Linux only, requires CUDA)
pip install -e ".[all,dev,gpu]"
```

### Dependency Groups
InstaGeo organizes dependencies into focused groups:
- **data**: Geospatial data processing and satellite imagery handling
- **model**: Machine learning training and inference capabilities
- **apps**: Web application and API serving components
- **dev**: Development tools (linting, testing, pre-commit hooks)
- **all**: Includes data, model, and apps groups

**Note**: The `--locked` flag ensures you install the exact versions specified in `uv.lock`, providing reproducible builds across different environments.

### Install Specific Components

#### Using uv
```bash
# Data processing only
uv sync --locked --extra data --extra cpu

# Model training only
uv sync --locked --extra model --extra cpu

# Web application only
uv sync --locked --extra apps --extra cpu

# Development tools
uv sync --locked --extra dev --extra cpu
```

#### Using pip (from GitHub)
```bash
# Data processing only
pip install "git+https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML.git#egg=InstaGeo[data,cpu]"

# Model training only
pip install "git+https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML.git#egg=InstaGeo[model,cpu]"

# Web application only
pip install "git+https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML.git#egg=InstaGeo[apps,cpu]"

# Development tools
pip install "git+https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML.git#egg=InstaGeo[dev,cpu]"
```

## Updating Dependencies

### Updating the Lock File

InstaGeo uses `uv.lock` to ensure reproducible builds. When you need to update dependencies:

#### Update All Dependencies
```bash
# Update all dependencies to their latest compatible versions
uv lock --upgrade

# Sync your environment with the updated lock file
uv sync --locked --extra all --extra dev --extra cpu
```

#### Update Specific Packages
```bash
# Update specific packages
uv lock --upgrade-package numpy --upgrade-package pandas

# Update a package to a specific version
uv add "numpy>=2.0.0"
uv lock
```

#### Add New Dependencies
```bash
# Add to runtime dependencies
uv add "new-package>=1.0.0"

# Add to optional dependencies
uv add --optional apps "web-framework>=3.0.0"
```

#### Lock File Best Practices
- **Commit `uv.lock`**: Always commit the lock file to ensure reproducible builds
- **Regular Updates**: Update dependencies regularly for security and bug fixes
- **Test After Updates**: Run tests after updating to catch compatibility issues

#### Troubleshooting Lock Issues
```bash
# If you encounter lock file conflicts
uv lock --refresh

# Reset lock file completely (use with caution)
rm uv.lock
uv lock

# Verify lock file integrity
uv sync --locked --dry-run
```

## Running Tests
After installation, you may want to verify that InstaGeo has been correctly installed and is functioning as expected. To do this, run the included test suite with the following commands:

```bash
pytest --verbose .
```

## Quick Start - Full-Stack Application

To quickly get started with the modern InstaGeo web application:

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for development)

### Launch the Application

#### Prerequisites
1. **Docker and Docker Compose** installed and running
2. **Configuration file**: Copy and configure `instageo/new_apps/config.env.example` to `instageo/new_apps/config.env`

#### Basic Launch
```bash
# Set environment stage (dev or prod)
export STAGE=dev

# Basic deployment (skips model registry sync, Cloudflare tunnel disabled by default)
./scripts/deploy.sh --skip-registry-sync
```

#### Deployment Options and Flags

The `deploy.sh` script supports several flags for different deployment scenarios:

```bash
# Default deployment (skips Cloudflare tunnel, includes model registry sync)
./scripts/deploy.sh

# Skip model registry synchronization (local development)
./scripts/deploy.sh --skip-registry-sync

# Enable Cloudflare tunnel (production deployment)
./scripts/deploy.sh --cloudflare

# Production with models and Cloudflare tunnel
./scripts/deploy.sh --cloudflare

# Only sync model registry (don't start application)
./scripts/deploy.sh --registry-sync-only
```

**Note**: Cloudflare tunnel is **disabled by default**. Use `--cloudflare` flag to enable it for production deployments.

#### Deployment Flag Details

| Flag | Description | Use Case |
|------|-------------|----------|
| `--skip-registry-sync` | Skip downloading models from Google Cloud Storage | When models are already available locally or not needed |
| `--cloudflare` | Enable Cloudflare tunnel setup | Production deployments with public access |
| `--registry-sync-only` | Only sync models, don't start application | Update models without restarting services |

**Default Behavior:**
- **Cloudflare tunnel**: **Disabled by default** (use `--cloudflare` to enable)
- **Model registry sync**: **Enabled by default** (use `--skip-registry-sync` to disable)
- **Local development**: Simple setup with `./scripts/deploy.sh --skip-registry-sync`

#### Environment Configuration

Before deployment, configure `instageo/new_apps/config.env`:

**Required for all deployments:**
```bash
STAGE=dev                              # or 'prod'
REDIS_HOST=instageo-redis
REDIS_PORT=6379
DATA_FOLDER=/app/instageo-data
EARTHDATA_USERNAME=your-username       # NASA EarthData credentials
EARTHDATA_PASSWORD=your-password
```

**Required for model registry sync:**
```bash
HOST_MODELS_PATH=/path/to/models       # Local path for model storage
MODELS_REGISTRY_GCS_URI="gs://path/to/models/registry.yaml"
```

**Required for Cloudflare tunnel (use `--cloudflare` flag):**
```bash
DOMAIN_NAME=your-domain.com
CLOUDFLARE_TUNNEL_NAME=instageo-tunnel
CLOUDFLARE_TUNNEL_CREDS='{"AccountTag":"...","TunnelSecret":"...","TunnelID":"..."}'
```
*Note: Only needed when using `./scripts/deploy.sh --cloudflare`*

**Optional worker scaling:**
```bash
DATA_PROCESSING_WORKER_REPLICAS=1      # Scale data processing workers
MODEL_PREDICTION_WORKER_REPLICAS=1     # Scale model prediction workers
VISUALIZATION_PREPARATION_WORKER_REPLICAS=1
```

#### Access Points

**Development Mode (STAGE=dev):**
- **Frontend**: http://localhost (Interactive web interface)
- **Backend API**: http://localhost/api (REST API endpoints)
- **RQ Dashboard**: http://localhost:9181 (Job queue monitoring)

**Production Mode (STAGE=prod):**
- **Frontend**: https://your-domain.com (via Cloudflare tunnel)
- **Backend API**: https://your-domain.com/api
- **RQ Dashboard**: https://your-domain.com:9181 (password protected)

#### Common Deployment Scenarios

**Quick Local Development (no models needed):**
```bash
export STAGE=dev
./scripts/deploy.sh --skip-registry-sync
```

**Local Development with Models:**
```bash
export STAGE=dev
./scripts/deploy.sh
```

**Production with Cloudflare:**
```bash
export STAGE=prod
./scripts/deploy.sh --cloudflare
```

**Update Models Only:**
```bash
./scripts/deploy.sh --registry-sync-only
```

#### Deployment Troubleshooting

**Common Issues:**

1. **Docker not running:**
   ```bash
   # Check Docker status
   docker info
   # Start Docker if needed
   ```

2. **Configuration file missing:**
   ```bash
   # Copy and edit configuration
   cp instageo/new_apps/config.env.example instageo/new_apps/config.env
   # Edit the file with your settings
   ```

3. **Port conflicts:**
   ```bash
   # Check what's using ports 80, 3000, 8000, 6379, 9181
   lsof -i :80
   # Stop conflicting services or change ports in config.env
   ```

4. **Model registry sync fails:**
   ```bash
   # Check Google Cloud authentication
   gcloud auth list
   # Or skip registry sync for local development
   ./scripts/deploy.sh --skip-registry-sync
   ```

5. **Cloudflare tunnel issues:**
   ```bash
   # Cloudflare is disabled by default
   # To enable: ./scripts/deploy.sh --cloudflare
   # Check tunnel credentials in config.env if issues persist
   ```

**Useful Management Commands:**
```bash
# View service logs
docker compose -f instageo/new_apps/docker-compose.dev.yml logs -f

# Stop all services
docker compose -f instageo/new_apps/docker-compose.dev.yml down

# Restart specific service
docker compose -f instageo/new_apps/docker-compose.dev.yml restart instageo-backend-api

# Scale workers
docker compose -f instageo/new_apps/docker-compose.dev.yml up -d --scale instageo-backend-data-processing-worker=4
```

### Using the Web Interface
1. **Draw Bounding Box**: Use the map interface to draw rectangular areas of interest
2. **Select Model**: Choose from available geospatial models (AOD estimation, flood detection, etc.)
3. **Configure Parameters**: Set date, cloud coverage, and processing parameters
4. **Submit Task**: Start data processing and model prediction
5. **Monitor Progress**: Track task status in real-time
6. **View Results**: Visualize predictions on the map and download PDF reports

For detailed setup and configuration, see the [New Apps documentation](./instageo/new_apps/README.md).
## Usage

### Data Component

InstaGeo's data component provides powerful tools for satellite imagery processing:

- **Multi-Source Support**: Download and process HLS, Sentinel-2, and Sentinel-1 imagery
- **Automated Processing**: Create ML-ready chips with segmentation maps
- **Quality Control**: Built-in cloud masking and data validation
- **Scalable Architecture**: Dask integration for distributed processing

**Key Tools:**
- `chip_creator.py`: Create training chips from observation records
- `raster_chip_creator.py`: Generate chips from existing raster files

For detailed usage instructions, examples, and configuration options, see the [Data Component Documentation](./instageo/data/README.md).

### Model Component

Advanced machine learning capabilities built on the Prithvi foundational model:

- **Custom Training**: Fine-tune models for classification and regression tasks
- **Multiple Inference Modes**: Chip inference, sliding window, and Ray-based serving
- **Model Registry**: Centralized model management with GCS integration
- **Comprehensive Metrics**: Advanced evaluation and monitoring capabilities

**Key Features:**
- Support for temporal and non-temporal inputs
- Model distillation and custom loss functions
- Hydra-based configuration management
- Pre-trained models for various geospatial tasks

For training examples, inference modes, and model registry setup, see the [Model Component Documentation](./instageo/model/README.md).

### Apps Component (Legacy)

Basic model operationalization with interactive mapping and PDF report generation. See the [Apps Documentation](./instageo/apps/README.md) for details.

### New Apps Component (Modern Full-Stack Platform)

A complete geospatial analysis platform with React frontend and FastAPI backend:

- **Interactive Web Interface**: Draw bounding boxes, select models, monitor tasks
- **Production Ready**: Docker deployment with Nginx, Redis, and worker scaling
- **Real-time Processing**: Two-stage task system with progress tracking
- **API Integration**: RESTful endpoints for programmatic access

**Quick Start:**
```bash
export STAGE=dev
./scripts/deploy.sh --skip-registry-sync
```

For detailed setup, API documentation, and deployment options, see the [New Apps Documentation](./instageo/new_apps/README.md).

## Examples and Tutorials

### End-to-End Demo
See the [InstaGeo Demo Notebook](notebooks/InstaGeo_Demo.ipynb) for a complete end-to-end example using a locust breeding ground prediction task (Note: The App section in this notebook still uses the legacy `apps` component).

### Component-Specific Examples
- **Data Processing**: See [Data Component Documentation](./instageo/data/README.md) for chip creation examples and check the demo notebooks for data preparation scenarios
   - **Chip Creator Demo**: [notebooks/chip_creator_demo.ipynb](notebooks/chip_creator_demo.ipynb)
   - **Raster Chip Creator Demo**: [notebooks/raster_chip_creator_demo.ipynb](notebooks/raster_chip_creator_demo.ipynb)
   - **Data Cleaner Demo**: [notebooks/data_cleaner_demo.ipynb](notebooks/data_cleaner_demo.ipynb)
   - **Data Splitter Demo**: [notebooks/data_splitter_demo.ipynb](notebooks/data_splitter_demo.ipynb)
- **Model Training**: See [Model Component Documentation](./instageo/model/README.md) for training examples with Sen1Floods11, crop classification, and locust prediction
- **Web Application**: See [New Apps Documentation](./instageo/new_apps/README.md) for API usage and deployment examples

## Deployment

After preparing data and training models, the model can be deployed using InstaGeo.
See [Quick Start – Full-Stack Application](./instageo/new_apps/README.md#quick-start) for setup and deployment instructions.

### Deployment Features
- **Containerized Architecture**: Docker Compose for consistent environments
- **Cloudflare Tunnel Integration**: Secure public access without port forwarding
- **Nginx Reverse Proxy**: Production-ready load balancing and routing
- **Worker Scaling**: Configurable data processing and model prediction workers
- **Monitoring**: Built-in RQ Dashboard for queue and worker monitoring
- **Environment Management**: Separate configurations for development and production

### Infrastructure Components
- **Frontend**: React application with hot reload in development
- **Backend**: FastAPI with multiple worker processes
- **Database**: Redis for task storage and job queues
- **Workers**: Scalable RQ workers for data processing and model prediction
- **Proxy**: Nginx for routing and static file serving
- **Monitoring**: RQ Dashboard for operational visibility

## Contributing

We welcome contributions to InstaGeo. Please follow the [contribution guidelines](./CONTRIBUTING.md) for submitting pull requests and reporting issues to help us improve the package.

<!-- ## License -->

## Citation

If you use InstaGeo in your research, please cite:

```bibtex
@article{yusuf2025instageo,
  title={InstaGeo: Compute-Efficient Geospatial Machine
Learning from Data to Deployment},
  author={Yusuf, Ibrahim and {et al.}},
  journal={arXiv preprint arXiv:2510.05617},
  year={2025}
}
```
