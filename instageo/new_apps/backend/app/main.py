"""FastAPI application for InstaGeo backend."""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import redis  # type: ignore
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .jobs import data_processing_queue, get_queues_status, model_prediction_queue
from .tasks import Task

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


class BoundingBox(BaseModel):
    """Bounding box model."""

    coordinates: List[
        List[float]
    ]  # [[lon_min_1, lat_min_1, lon_max_1, lat_max_1], ...]
    date: str


class ProcessDataRequest(BaseModel):
    """Request model for process-data endpoint."""

    bounding_boxes: List[BoundingBox]
    parameters: Optional[Dict[str, Any]] = None


class ProcessDataResponse(BaseModel):
    """Response model for process-data endpoints."""

    status: str
    task_id: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


class TaskStatusResponse(BaseModel):
    """Response model for task status endpoints."""

    task_id: str
    status: str
    created_at: str
    data_processing_job_id: Optional[str] = None
    model_prediction_job_id: Optional[str] = None
    bounding_boxes: Optional[List[Dict[str, Any]]] = None
    parameters: Optional[Dict[str, Any]] = None
    stages: Dict[str, Any]


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {"message": "Welcome to InstaGeo API"}


@app.post("/api/run-model", response_model=ProcessDataResponse)
async def process_data(request: ProcessDataRequest) -> ProcessDataResponse:
    """Submit a task for data processing and model prediction."""
    try:
        logger.info(
            f"Received request with {len(request.bounding_boxes)} bounding boxes"
        )

        # Convert Pydantic models to dictionaries
        bounding_boxes = [bbox.model_dump() for bbox in request.bounding_boxes]
        logger.info(f"Converted bounding boxes: {bounding_boxes}")

        # Generate task ID
        task_id = str(uuid.uuid4())

        # Create task instance (this automatically starts data processing)
        task = Task(
            task_id=task_id,
            bounding_boxes=bounding_boxes,
            parameters=request.parameters,
        )

        logger.info(f"Created task with ID: {task_id}")

        return ProcessDataResponse(
            status=task.status,
            task_id=task_id,
            message="Task submitted successfully. Data processing will start automatically.",
        )

    except Exception as e:
        logger.error(f"Error in process_data endpoint: {str(e)}")
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
            data_processing_job_id=task.data_processing_job.job_id
            if task.data_processing_job
            else None,
            model_prediction_job_id=task.model_prediction_job.job_id
            if task.model_prediction_job
            else None,
            bounding_boxes=task.bounding_boxes,
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
                        "bounding_boxes_count": len(task.bounding_boxes),
                        "model_type": task.parameters.get("model_type", "unknown")
                        if task.parameters
                        else "unknown",
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
            health_status["components"]["redis"][
                "message"
            ] = "Redis connection successful"
        except Exception as e:
            health_status["components"]["redis"]["status"] = "unhealthy"
            health_status["components"]["redis"][
                "message"
            ] = f"Redis connection failed: {str(e)}"

        # Check queue status
        try:
            queue_status = get_queues_status()
            health_status["components"]["queues"]["status"] = "healthy"
            health_status["components"]["queues"]["message"] = "Queues are accessible"
            health_status["components"]["queues"]["details"] = queue_status
        except Exception as e:
            health_status["components"]["queues"]["status"] = "unhealthy"
            health_status["components"]["queues"][
                "message"
            ] = f"Queue check failed: {str(e)}"

        # Check worker status (basic check)
        try:
            # Check if workers can access the queues
            dp_queue_length = len(data_processing_queue)
            mp_queue_length = len(model_prediction_queue)

            health_status["components"]["workers"]["status"] = "healthy"
            health_status["components"]["workers"][
                "message"
            ] = "Workers can access queues"
            health_status["components"]["workers"]["details"] = {
                "data_processing_queue_length": dp_queue_length,
                "model_prediction_queue_length": mp_queue_length,
            }
        except Exception as e:
            health_status["components"]["workers"]["status"] = "unhealthy"
            health_status["components"]["workers"][
                "message"
            ] = f"Worker check failed: {str(e)}"

        # Determine overall status based on component statuses
        component_statuses = [
            comp["status"] for comp in health_status["components"].values()
        ]

        if any(status == "unhealthy" for status in component_statuses):
            health_status["status"] = "unhealthy"
        elif all(status == "healthy" for status in component_statuses):
            health_status["status"] = "healthy"

    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = f"Health check failed: {str(e)}"

    return health_status


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
