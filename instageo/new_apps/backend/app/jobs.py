"""Job class for InstaGeo backend."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rq import Queue
from rq.job import Job as RQJob

from instageo.new_apps.backend.app.redis_client import redis_client

PACKAGE_TASKS_PATH = "instageo.new_apps.backend.app.tasks"
DATA_PROCESSING_QUEUE_NAME = "data-processing"
MODEL_PREDICTION_QUEUE_NAME = "model-prediction"
VISUALIZATION_PREPARATION_QUEUE_NAME = "visualization-preparation"

# Initialize RQ queues for the two-stage pipeline
data_processing_queue = redis_client.get_queue(DATA_PROCESSING_QUEUE_NAME)
model_prediction_queue = redis_client.get_queue(MODEL_PREDICTION_QUEUE_NAME)
visualization_preparation_queue = redis_client.get_queue(VISUALIZATION_PREPARATION_QUEUE_NAME)


class JobType:
    """Job type constants."""

    DATA_PROCESSING = "data_processing"
    MODEL_PREDICTION = "model_prediction"
    VISUALIZATION_PREPARATION = "visualization_preparation"


class JobStatus:
    """Job status constants."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    """Represents a job in the queue system."""

    def __init__(
        self,
        job_id: str,
        job_type: str,
        task_id: str,
        function_name: str,
        args: tuple,
        meta: Optional[Dict[str, Any]] = None,
        queue: Optional[Queue] = None,
        created_at: Optional[str] = None,
    ):
        """Initialize a new job.

        Args:
            job_id: Unique job identifier.
            job_type: Type of job (data_processing, model_prediction, visualization_preparation).
            task_id: Associated task ID.
            function_name: Name of the function to execute.
            args: Arguments to pass to the function.
            meta: Additional metadata.
            queue: RQ queue instance.
            created_at: Job creation timestamp.
        """
        self.job_id = job_id
        self.job_type = job_type
        self.task_id = task_id
        self.function_name = function_name
        self.args = args
        self.meta = meta or {}
        self.queue = queue
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

        # Status tracking
        self.status = JobStatus.PENDING
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

    def enqueue(self, timeout: str = "4h", ttl: str = "2h") -> str:
        """Enqueue the job in the appropriate queue.

        Args:
            timeout: Job timeout.
            ttl: Job time to live.


        Returns:
            Job ID.
        """
        if self.queue is None:
            raise ValueError("Queue not set for job")

        # Create RQ job
        rq_job = self.queue.enqueue(
            self.function_name,
            args=self.args,
            meta=self.meta,
            job_timeout=timeout,
            ttl=ttl,
        )

        self.job_id = str(rq_job.id)
        self.save()

        return self.job_id

    def save(self) -> None:
        """Save job metadata to Redis."""
        job_meta = {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "task_id": self.task_id,
            "function_name": self.function_name,
            "args": str(self.args),  # Convert tuple to string for storage
            "meta": str(self.meta),  # Convert dict to string for storage
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at or "",
            "completed_at": self.completed_at or "",
            "result": str(self.result) if self.result else "",
            "error": self.error or "",
        }

        # Store job metadata in Redis
        redis_client.save_job(self.job_id, job_meta)

    def load(self) -> None:
        """Load job from Redis."""
        job_data = redis_client.load_job(self.job_id)

        if not job_data:
            raise ValueError(f"Job {self.job_id} not found")

        # Parse job data
        for key, value in job_data.items():
            # Handle empty strings
            if value == "":
                if key in ["started_at", "completed_at", "error"]:
                    value = None
                elif key in ["result", "args", "meta"]:
                    value = None

            setattr(self, key, value)

        # Set queue based on job type
        if self.job_type == JobType.DATA_PROCESSING:
            self.queue = data_processing_queue
        elif self.job_type == JobType.MODEL_PREDICTION:
            self.queue = model_prediction_queue
        elif self.job_type == JobType.VISUALIZATION_PREPARATION:
            self.queue = visualization_preparation_queue

    def update_status(
        self,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update job status.

        Args:
            status: New status.
            result: Job result.
            error: Job error.
        """
        self.status = status

        if status == JobStatus.RUNNING:
            self.started_at = datetime.now(timezone.utc).isoformat()
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            self.completed_at = datetime.now(timezone.utc).isoformat()

        if result:
            self.result = result

        if error:
            self.error = error

        self.save()

    def get_rq_job(self) -> Optional[RQJob]:
        """Get the underlying RQ job object.

        Returns:
            RQ Job object or None if not found.
        """
        if self.queue is None:
            return None

        return self.queue.fetch_job(self.job_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary.

        Returns:
            Job data as dictionary.
        """
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "task_id": self.task_id,
            "function_name": self.function_name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "meta": self.meta,
        }

    @classmethod
    def create_data_processing(
        cls,
        task_id: str,
        bboxes: List[List[float]],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "Job":
        """Create a new data processing job.

        Args:
            task_id: Associated task ID.
            bboxes: List of bounding box dictionaries.
            parameters: Optional processing parameters.

        Returns:
            New Job instance.
        """
        job_id = f"dp_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        meta = {
            "task_id": task_id,
            "bboxes": bboxes,
            "parameters": parameters,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "stage": "data_processing",
        }

        job = cls(
            job_id=job_id,
            job_type=JobType.DATA_PROCESSING,
            task_id=task_id,
            function_name=f"{PACKAGE_TASKS_PATH}.process_data_extraction_with_task",
            args=(task_id, bboxes, parameters),
            meta=meta,
            queue=data_processing_queue,
        )

        return job

    @classmethod
    def create_model_prediction(
        cls,
        task_id: str,
        processed_data: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "Job":
        """Create a new model prediction job.

        Args:
            task_id: Associated task ID.
            processed_data: Data from data processing stage.
            parameters: Optional model parameters.

        Returns:
            New Job instance.
        """
        job_id = f"mp_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        meta = {
            "task_id": task_id,
            "processed_data": processed_data,
            "parameters": parameters,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "stage": "model_prediction",
        }

        job = cls(
            job_id=job_id,
            job_type=JobType.MODEL_PREDICTION,
            task_id=task_id,
            function_name=f"{PACKAGE_TASKS_PATH}.process_model_prediction_with_task",
            args=(task_id, processed_data, parameters),
            meta=meta,
            queue=model_prediction_queue,
        )

        return job

    @classmethod
    def create_visualization_preparation(
        cls,
        task_id: str,
        processed_data: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "Job":
        """Create a visualization preparation job.

        Args:
            task_id: Task ID.
            processed_data: Data from data processing stage.
            prediction_results: Results from model prediction stage.
            parameters: Optional job parameters.

        Returns:
            Job instance.
        """
        job_id = f"vp_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return cls(
            job_id=job_id,
            job_type=JobType.VISUALIZATION_PREPARATION,
            task_id=task_id,
            queue=visualization_preparation_queue,
            function_name=f"{PACKAGE_TASKS_PATH}.process_visualization_preparation_with_task",
            args=(task_id, processed_data, parameters),
        )

    @classmethod
    def get(cls, job_id: str) -> "Job":
        """Get an existing job by ID.

        Args:
            job_id: Job ID.

        Returns:
            Job instance.
        """
        job = cls(job_id, "", "", "", (), {})
        job.load()
        return job


# Queue management functions
def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get status of a job from either queue.

    Args:
        job_id: Job ID to check.

    Returns:
        Job status dictionary.
    """
    try:
        job = Job.get(job_id)
        return job.to_dict()
    except ValueError:
        return {
            "job_id": job_id,
            "status": "not_found",
            "queue": None,
        }


def get_queues_status() -> Dict[str, Any]:
    """Get status of all job queues.

    Returns:
        Dictionary with queue status information.
    """
    return {
        "data_processing": {
            "name": "data-processing",
            "job_count": len(data_processing_queue),
            "is_empty": data_processing_queue.is_empty(),
        },
        "model_prediction": {
            "name": "model-prediction",
            "job_count": len(model_prediction_queue),
            "is_empty": model_prediction_queue.is_empty(),
        },
        "visualization_preparation": {
            "name": "visualization-preparation",
            "job_count": len(visualization_preparation_queue),
            "is_empty": visualization_preparation_queue.is_empty(),
        },
    }
