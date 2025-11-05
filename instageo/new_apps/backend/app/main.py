"""FastAPI application for InstaGeo backend."""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session

from instageo.model.configs.config_dataclasses import ModelInfo
from instageo.model.registry.model_registry import ModelRegistry
from instageo.new_apps.backend.app.auth import (
    get_current_user,
    is_task_owner,
    verify_access_token,
)
from instageo.new_apps.backend.app.crud import create_task_in_db
from instageo.new_apps.backend.app.db import get_db, init_db
from instageo.new_apps.backend.app.jobs import (
    data_processing_queue,
    get_queues_status,
    model_prediction_queue,
    visualization_preparation_queue,
)
from instageo.new_apps.backend.app.models import User
from instageo.new_apps.backend.app.redis_client import redis_client
from instageo.new_apps.backend.app.tasks import Task
from instageo.new_apps.backend.app.tiler_service import InstaGeoTilerService

# Configure logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Lifespan for the InstaGeo backend."""
    init_db()
    yield


app = FastAPI(title="InstaGeo API", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PUBLIC_ENDPOINTS = ["/", "/api/health"]


@app.middleware("http")
async def verify_token_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Verify JWT token for API routes."""
    # skip auth for public endpoints
    if request.url.path in PUBLIC_ENDPOINTS:
        return await call_next(request)

    # skip auth for TiTiler routes since they don't use JWT auth
    if request.url.path.startswith("/api/titiler/"):
        return await call_next(request)

    # skip auth for OPTIONS requests (CORS preflight)
    if request.method == "OPTIONS":
        logger.info(f"Skipping auth for OPTIONS request (CORS preflight): {request.url.path}")
        return await call_next(request)

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return Response(
            content='{"detail":"Not authenticated - Login required"}',
            status_code=401,
            media_type="application/json",
        )

    try:
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=auth_header.split(" ")[1]
        )
        verify_access_token(credentials)
        return await call_next(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.debug(f"Token verification failed {e}")
        return Response(
            content='{"detail":"Not authenticated - Login required"}',
            status_code=401,
            media_type="application/json",
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


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {"message": "Welcome to InstaGeo API"}


@app.post("/api/run-model", response_model=TaskCreationResponse)
async def create_task(
    task_request: TaskCreationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TaskCreationResponse:
    """Submit a task for data processing and model prediction."""
    try:
        logger.info(f"Received task creation request: {task_request}")
        logger.info(f"Model key from request: {task_request.model_key}")
        logger.info(f"Model size from request: {task_request.model_size}")
        # Generate task ID
        task_id = str(uuid.uuid4())

        # Use the model parameters from the frontend request
        model_key = task_request.model_key
        model_size = task_request.model_size

        # Validate that model_key and model_size are provided
        if not model_key or not model_size:
            raise HTTPException(
                status_code=400,
                detail="Both model_key and model_size must be provided",
            )

        model_registry = ModelRegistry()
        model_info = model_registry.get_model_metadata_for_size(model_key, model_size)
        logger.info(f"Model info: {model_info}")

        if model_info is None:
            # Get available models for better error message
            available_models = model_registry.get_available_models()
            available_keys = list({m.model_key for m in available_models})
            available_sizes = list({m.model_size for m in available_models})

            raise HTTPException(
                status_code=400,
                detail=f"Model {model_key}/{model_size} not found in registry. "
                f"Available model keys: {available_keys}, "
                f"Available sizes: {available_sizes}",
            )

        parameters = {
            # Parameters from the frontend request
            "model_key": model_key,
            "model_size": model_size,
            "date": task_request.date,
            "cloud_coverage": task_request.cloud_coverage,
            "temporal_tolerance": task_request.temporal_tolerance or model_info.temporal_step,
            # True model parameters from registry
            "chip_size": model_info.chip_size,
            "num_steps": model_info.num_steps,
            "data_source": model_info.data_source,
            "temporal_step": model_info.temporal_step,
            "model_type": model_info.model_type,
            "model_short_name": model_info.model_short_name,
            "model_name": model_info.model_name,
            "classes_mapping": model_info.classes_mapping,
        }

        # Create task instance (this automatically starts data processing)
        task = Task(
            task_id=task_id,
            bboxes=task_request.bboxes,
            parameters=parameters,
        )
        logger.info(f"Created task with ID: {task_id}")

        # Create database record for the task
        create_task_in_db(
            db, current_user.sub, task_id, task_request.bboxes, parameters, task.status
        )

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
@is_task_owner
async def get_task_status_endpoint(
    task_id: str,
    claims: Dict[str, Any] = Depends(verify_access_token),
    db: Session = Depends(get_db),
) -> TaskStatusResponse:
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
async def get_all_tasks(
    claims: Dict[str, Any] = Depends(verify_access_token), db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get all tasks with their basic information."""
    try:
        from instageo.new_apps.backend.app.crud import (
            get_completed_failed_tasks,
            get_in_progress_tasks,
        )

        # Get completed and failed tasks from database
        user_sub: str | Any = claims.get("sub")
        completed_tasks = get_completed_failed_tasks(db, user_sub)

        # Get in-progress tasks (Redis with database fallback)
        in_progress_tasks = get_in_progress_tasks(db, user_sub)

        # Combine and return all tasks
        all_tasks = completed_tasks + in_progress_tasks
        return all_tasks

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


@app.get("/api/models/{model_name}")
async def get_model_details(model_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    try:
        model_registry = ModelRegistry()
        available_models = model_registry.get_available_models()

        # Check if model exists in available models
        model_exists = any(model.model_key == model_name for model in available_models)
        if not model_exists:
            available_model_keys = [m.model_key for m in available_models]
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: {available_model_keys}",
            )

        model_data = model_registry._all_models_metadata["models"][model_name]

        return {
            "status": "success",
            "model": {
                "name": model_name,
                "description": model_data.get("description", ""),
                "task_type": model_data.get("task_type", "unknown"),
                "data_source": model_data.get("data_source", "unknown"),
                "sizes": model_data.get("sizes", {}),
                "metadata": {
                    k: v
                    for k, v in model_data.items()
                    if k not in ["description", "task_type", "data_source", "sizes"]
                },
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model details for {model_name}: {str(e)}")
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
            redis_client.ping()
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
    """Get all available models from the model registry."""
    try:
        model_registry = ModelRegistry()
        models = model_registry.get_available_models()
        logger.info(f"list of available models: {models}")
        return models
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
