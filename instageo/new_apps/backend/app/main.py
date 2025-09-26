"""FastAPI application for InstaGeo backend."""

import logging
import os
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

import redis  # type: ignore
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .jobs import (
    data_processing_queue,
    get_queues_status,
    model_prediction_queue,
    visualization_preparation_queue,
)
from .tasks import Task
from .tiler_service import InstaGeoTilerService

# Configure logging
logger = logging.getLogger(__name__)

app = FastAPI(title="InstaGeo API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis connection for health checks
redis_conn: Any = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
)

# Initialize TiTiler service
DATA_FOLDER = os.getenv("DATA_FOLDER", "/app/instageo-data")
tiler_service = InstaGeoTilerService(base_directory=DATA_FOLDER)


# Add middleware to intercept TiTiler requests and map task_id to file paths
# We do this because we don't want to expose the file paths to the frontend
@app.middleware("http")
async def intercept_titiler_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Intercept TiTiler requests and map task_id to actual file paths."""
    # Check if this is a TiTiler request with task_id
    if request.url.path.startswith("/api/titiler/") and "url=" in str(request.query_params):
        url_param = request.query_params.get("url")
        if (
            not url_param.startswith("file://")
            and not url_param.startswith("http")
            and not url_param.startswith("https")
        ):
            try:
                # Check if this is a chips or predictions request
                cog_file = None
                if url_param.endswith("_chips"):
                    task_id = url_param[:-6]  # Remove "_chips" suffix
                    cog_file = tiler_service._get_chips_cog_file(task_id)
                elif url_param.endswith("_predictions"):
                    task_id = url_param[:-12]  # Remove "_predictions" suffix
                    cog_file = tiler_service._get_predictions_cog_file(task_id)

                # Modify the request query parameters with file:// protocol
                query_params = dict(request.query_params)
                query_params["url"] = f"file://{cog_file}"

                # Create new query string
                new_query = "&".join([f"{k}={v}" for k, v in query_params.items()])

                # Modify the request scope to update query string
                request.scope["query_string"] = new_query.encode()
                logger.info(f"Mapped task from {url_param} to file URL file://{cog_file}")

            except Exception as e:
                logger.error(f"Failed to map task from {url_param} to file path: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to map task_id to visualization files",
                )

    response = await call_next(request)

    # Remove tiles key from TileJSON responses to prevent file path exposure
    if (
        request.url.path.endswith("/tilejson.json")
        and response.headers.get("content-type", "").startswith("application/json")
        and response.status_code == 200
    ):
        try:
            import json

            # Read the response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk

            # Parse JSON and remove tiles key
            data = json.loads(body.decode())
            if "tiles" in data:
                del data["tiles"]
                logger.info(
                    "Removed tiles key from TileJSON response to prevent file path exposure"
                )

            # Create new response without tiles key and correct Content-Length
            modified_content = json.dumps(data)
            return Response(
                content=modified_content,
                media_type="application/json",
                status_code=response.status_code,
                headers={
                    key: value
                    for key, value in response.headers.items()
                    if key.lower() != "content-length"  # Remove old Content-Length
                },
            )
        except Exception as e:
            logger.error(f"Failed to filter TileJSON response: {e}")
            # Return original response if filtering fails
            pass

    return response


# Include custom task-based routes
app.include_router(tiler_service.get_router())

# Include actual TiTiler routes for serving tiles with /api/cog prefix
app.include_router(tiler_service.get_tiler_router(), prefix="/api/titiler")


class TaskCreationRequest(BaseModel):
    """Request model for task creation through run-model endpoint."""

    bboxes: List[List[float]]
    model_key: str
    model_size: str
    date: str
    cloud_coverage: int
    temporal_tolerance: int


class TaskCreationResponse(BaseModel):
    """Response model for task creation."""

    status: str
    task_id: Optional[str] = None
    message: Optional[str] = None


class TaskStatusResponse(BaseModel):
    """Response model for task status endpoints."""

    task_id: str
    status: str
    created_at: str
    data_processing_job_id: Optional[str] = None
    model_prediction_job_id: Optional[str] = None
    visualization_preparation_job_id: Optional[str] = None
    bboxes: Optional[List[List[float]]] = None
    parameters: Optional[Dict[str, Any]] = None
    stages: Dict[str, Any]


class ModelInfo(BaseModel):
    """Model info."""

    model_key: str
    model_type: str
    model_short_name: str
    model_name: str
    model_description: Optional[str] = None
    model_size: str
    num_params: float
    chip_size: int
    num_steps: int
    data_source: str
    temporal_step: int
    classes_mapping: Dict[int, str]


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {"message": "Welcome to InstaGeo API"}


@app.post("/api/run-model", response_model=TaskCreationResponse)
async def create_task(task_request: TaskCreationRequest) -> TaskCreationResponse:
    """Submit a task for data processing and model prediction."""
    try:
        logger.info(f"Received task creation request: {task_request}")
        # Generate task ID
        task_id = str(uuid.uuid4())

        # To avoid dealing with tempered/malicious model parameters from the frontend
        #  to run the data extraction for instance, let's get model parameters
        # (chip_size, num_steps, etc) directly from our registry
        DUMMY_REGISTRY = {
            "aod-estimator": {
                "chip_size": 224,
                "num_steps": 1,
                "data_source": "HLS",
                "temporal_step": 0,
                "model_type": "reg",
                "model_short_name": "aod-estim",
                "model_name": "Aerosol Optical Depth Estimation",
                "model_description": "Estimates aerosol optical depth values"
                "from satellite imagery for air quality monitoring",
                "model_size": "tiny",
                "num_params": 4114000,
                "classes_mapping": {},
            }
        }
        model_info = DUMMY_REGISTRY[task_request.model_key]

        parameters = {
            # Parameters from the request
            "model_key": task_request.model_key,
            "model_size": task_request.model_size,
            "date": task_request.date,
            "cloud_coverage": task_request.cloud_coverage,
            "temporal_tolerance": task_request.temporal_tolerance,
            # True model parameters from registry
            "chip_size": model_info["chip_size"],
            "num_steps": model_info["num_steps"],
            "data_source": model_info["data_source"],
            "temporal_step": model_info["temporal_step"],
            "model_type": model_info["model_type"],
            "model_short_name": model_info["model_short_name"],
            "model_name": model_info["model_name"],
            "classes_mapping": model_info["classes_mapping"],
        }
        # Create task instance (this automatically starts data processing)
        task = Task(
            task_id=task_id,
            bboxes=task_request.bboxes,
            parameters=parameters,
        )
        logger.info(f"Created task with ID: {task_id}")

        return TaskCreationResponse(
            status=task.status,
            task_id=task_id,
            message="Task submitted successfully. Data processing will start automatically.",
        )

    except Exception as e:
        logger.error(f"Error in task creation process: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status_endpoint(task_id: str) -> TaskStatusResponse:
    """Get status of a task."""
    try:
        task = Task.get(task_id)
        return TaskStatusResponse(
            task_id=task.task_id,
            status=task.status,
            created_at=task.created_at,
            data_processing_job_id=(
                task.data_processing_job.job_id if task.data_processing_job else None
            ),
            model_prediction_job_id=(
                task.model_prediction_job.job_id if task.model_prediction_job else None
            ),
            visualization_preparation_job_id=(
                task.visualization_preparation_job.job_id
                if task.visualization_preparation_job
                else None
            ),
            bboxes=task.bboxes,
            parameters=task.parameters,
            stages=task.stages,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks")
async def get_all_tasks() -> List[Dict[str, Any]]:
    """Get all tasks with their basic information."""
    try:
        # Get all task IDs from the sorted set, newest first
        task_ids = redis_conn.zrevrange("tasks_by_created", 0, -1)

        tasks = []
        for task_id_bytes in task_ids:
            task_id = task_id_bytes.decode("utf-8")
            try:
                task = Task.get(task_id)
                tasks.append(
                    {
                        "task_id": task.task_id,
                        "status": task.status,
                        "created_at": task.created_at,
                        "bboxes_count": len(task.bboxes),
                        "model_type": (
                            (
                                task.parameters.get("model_type")
                                or task.parameters.get("model_key", "unknown")
                            )
                            if task.parameters
                            else "unknown"
                        ),
                        "stages": task.stages,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to load task {task_id}: {e}")
                continue

        return tasks

    except Exception as e:
        logger.error(f"Error getting all tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/queues/status")
async def get_queues_status_endpoint() -> Dict[str, Any]:
    """Get status of all queues."""
    try:
        return get_queues_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check endpoint."""
    health_status: Dict[str, Any] = {
        "status": "unknown",
        "timestamp": None,
        "version": "1.0.0",
        "components": {
            "api": {"status": "healthy", "message": "API is running"},
            "redis": {"status": "unknown", "message": "Checking connection..."},
            "queues": {"status": "unknown", "message": "Checking queues..."},
            "workers": {"status": "unknown", "message": "Checking workers..."},
        },
    }

    try:
        from datetime import datetime

        health_status["timestamp"] = datetime.now().isoformat()

        # Check Redis connection
        try:
            redis_conn.ping()
            health_status["components"]["redis"]["status"] = "healthy"
            health_status["components"]["redis"]["message"] = "Redis connection successful"
        except Exception as e:
            health_status["components"]["redis"]["status"] = "unhealthy"
            health_status["components"]["redis"]["message"] = f"Redis connection failed: {str(e)}"

        # Check queue status
        try:
            queue_status = get_queues_status()
            health_status["components"]["queues"]["status"] = "healthy"
            health_status["components"]["queues"]["message"] = "Queues are accessible"
            health_status["components"]["queues"]["details"] = queue_status
        except Exception as e:
            health_status["components"]["queues"]["status"] = "unhealthy"
            health_status["components"]["queues"]["message"] = f"Queue check failed: {str(e)}"

        # Check worker status (basic check)
        try:
            # Check if workers can access the queues
            dp_queue_length = len(data_processing_queue)
            mp_queue_length = len(model_prediction_queue)
            vp_queue_length = len(visualization_preparation_queue)

            health_status["components"]["workers"]["status"] = "healthy"
            health_status["components"]["workers"]["message"] = "Workers can access queues"
            health_status["components"]["workers"]["details"] = {
                "data_processing_queue_length": dp_queue_length,
                "model_prediction_queue_length": mp_queue_length,
                "visualization_preparation_queue_length": vp_queue_length,
            }
        except Exception as e:
            health_status["components"]["workers"]["status"] = "unhealthy"
            health_status["components"]["workers"]["message"] = f"Worker check failed: {str(e)}"

        # Determine overall status based on component statuses
        component_statuses = [comp["status"] for comp in health_status["components"].values()]

        if any(status == "unhealthy" for status in component_statuses):
            health_status["status"] = "unhealthy"
        elif all(status == "healthy" for status in component_statuses):
            health_status["status"] = "healthy"

    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = f"Health check failed: {str(e)}"

    return health_status


@app.get("/api/models", response_model=List[ModelInfo])
async def get_models() -> List[ModelInfo]:
    """Dummy models endpoint for frontend integration (cached client-side)."""
    # Minimal set covering AOD reg with three sizes
    dummy: List[ModelInfo] = [
        ModelInfo(
            model_key="aod-estimator",
            model_type="reg",
            model_short_name="aod-estim",
            model_name="Aerosol Optical Depth Estimation",
            model_description="Estimates aerosol optical depth values"
            "from satellite imagery for air quality monitoring",
            model_size=size,
            num_params=num_params,
            chip_size=224,
            num_steps=1,
            data_source="HLS",
            temporal_step=0,
            classes_mapping={},
        )
        for size, num_params in (
            ("tiny", 4114000),
            ("medium", 16500000),
            ("large", 66000000),
        )
    ]
    return dummy


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
