"""Task management for InstaGeo backend."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import redis  # type: ignore

from .cog_converter import COGConverter
from .data_processor import DataProcessor
from .jobs import Job, JobStatus

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Redis connection for task storage
redis_conn: Any = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
)

DATA_FOLDER = os.getenv("DATA_FOLDER", "/app/instageo-data")

# Initialize data processor
data_processor = DataProcessor(base_output_dir=DATA_FOLDER)

# Initialize cog converter
cog_converter = COGConverter()


class TaskStatus:
    """Task status constants."""

    DATA_PROCESSING = "data_processing"
    MODEL_PREDICTION = "model_prediction"
    VISUALIZATION_PREPARATION = "visualization_preparation"
    COMPLETED = "completed"
    FAILED = "failed"


class Task:
    """Represents a task with two jobs: data processing and model prediction."""

    def __init__(
        self,
        task_id: str,
        bboxes: List[List[float]],
        parameters: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
        is_new: bool = True,
    ):
        """Initialize a new task.

        Args:
            task_id: Unique task identifier.
            bboxes: List of bounding box dictionaries.
            parameters: Optional processing parameters.
            created_at: Task creation timestamp.
            is_new: If True, start jobs and save; if False, just initialize for loading.
        """
        logger.debug(
            f"Creating Task {task_id} with bboxes: {bboxes}, "
            f"parameters: {parameters}, is_new={is_new}"
        )

        self.task_id = task_id
        self.bboxes = bboxes
        self.parameters = parameters or {}
        self.created_at = created_at or datetime.utcnow().isoformat() + "Z"
        # Store timestamp for efficient Redis sorting
        # Handle timezone-aware timestamp for Redis sorting
        self.created_timestamp = (
            datetime.fromisoformat(self.created_at[:-1]).replace(tzinfo=None).timestamp()
        )

        logger.debug(
            f"Task {task_id} initialized with self.bboxes: {self.bboxes}, "
            f"self.parameters: {self.parameters}"
        )

        # Job instances
        self.data_processing_job: Optional[Job] = None
        self.model_prediction_job: Optional[Job] = None
        self.visualization_preparation_job: Optional[Job] = None

        # Task status - starts with data processing since it's created and immediately started
        self.status = TaskStatus.DATA_PROCESSING

        # Stages information
        self.stages: Dict[str, Dict[str, Union[str, None, Dict[str, Any]]]] = {
            stage: {
                "status": JobStatus.PENDING,
                "started_at": None,
                "completed_at": None,
                "result": None,
                "error": None,
            }
            for stage in [
                TaskStatus.DATA_PROCESSING,
                TaskStatus.MODEL_PREDICTION,
                TaskStatus.VISUALIZATION_PREPARATION,
            ]
        }

        if is_new:
            # Create and start data processing job
            self.start_data_processing()
            # Save task to Redis (after jobs are set up)
            logger.debug(
                f"Task {task_id} saving to Redis with self.bboxes: {self.bboxes}, "
                f"self.parameters: {self.parameters}"
            )
            self.save()

    def save(self) -> None:
        """Save task to Redis."""
        task_meta = {
            "task_id": self.task_id,
            "bboxes": json.dumps(self.bboxes),
            "parameters": json.dumps(self.parameters),
            "status": self.status,
            "created_at": self.created_at,
            "created_timestamp": self.created_timestamp,
            "data_processing_job_id": (
                self.data_processing_job.job_id if self.data_processing_job else ""
            ),
            "model_prediction_job_id": (
                self.model_prediction_job.job_id if self.model_prediction_job else ""
            ),
            "visualization_preparation_job_id": (
                self.visualization_preparation_job.job_id
                if self.visualization_preparation_job
                else ""
            ),
        }

        # Store task metadata in Redis
        redis_conn.hset(f"task:{self.task_id}", mapping=task_meta)
        # Add to sorted set for efficient date sorting
        redis_conn.zadd("tasks_by_created", {self.task_id: self.created_timestamp})

        # Store stages information
        for stage_name, stage_data in self.stages.items():
            stage_key = f"task:{self.task_id}:stage:{stage_name}"
            updates = {
                "status": stage_data["status"],
                "started_at": stage_data["started_at"] or "",
                "completed_at": stage_data["completed_at"] or "",
                "result": (json.dumps(stage_data["result"]) if stage_data["result"] else ""),
                "error": stage_data["error"] or "",
            }
            redis_conn.hset(stage_key, mapping=updates)

    def load(self) -> None:
        """Load task from Redis."""
        task_data = redis_conn.hgetall(f"task:{self.task_id}")
        logger.debug(f"Loading Task {self.task_id} from Redis: {task_data}")

        if not task_data:
            raise ValueError(f"Task {self.task_id} not found")

        # Convert bytes to strings and parse JSON fields
        for k, v in task_data.items():
            key = k.decode("utf-8")
            value = v.decode("utf-8") if v is not None else None
            logger.debug(f"Task {self.task_id} loading key={key}, value={value}")
            # Handle empty strings
            if value == "":
                if key in [
                    "data_processing_job_id",
                    "model_prediction_job_id",
                    "visualization_preparation_job_id",
                ]:
                    value = None
                elif key == "parameters":
                    value = {}
                elif key == "bboxes":
                    value = []
            # Handle numeric fields
            elif key == "created_timestamp" and value:
                value = float(value)
            # Parse JSON fields
            if key in ["bboxes", "parameters"] and value not in [None, [], {}]:
                try:
                    if isinstance(value, str):
                        parsed = json.loads(value)
                        logger.debug(f"Task {self.task_id} parsed {key}: {parsed}")
                        setattr(self, key, parsed)
                    else:
                        setattr(self, key, value)
                except json.JSONDecodeError:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)
        logger.debug(
            f"Task {self.task_id} after load: bboxes={self.bboxes}, "
            f"parameters={self.parameters}"
        )
        # Load job instances if job IDs exist
        if hasattr(self, "data_processing_job_id") and self.data_processing_job_id:
            try:
                self.data_processing_job = Job.get(self.data_processing_job_id)
            except ValueError:
                self.data_processing_job = None
        if hasattr(self, "model_prediction_job_id") and self.model_prediction_job_id:
            try:
                self.model_prediction_job = Job.get(self.model_prediction_job_id)
            except ValueError:
                self.model_prediction_job = None
        if (
            hasattr(self, "visualization_preparation_job_id")
            and self.visualization_preparation_job_id
        ):
            try:
                self.visualization_preparation_job = Job.get(self.visualization_preparation_job_id)
            except ValueError:
                self.visualization_preparation_job = None

        # Load stages information
        self.stages = self._load_stages()

    def _load_stages(self) -> Dict[str, Any]:
        """Load stages information from Redis."""
        stages = {}

        for stage in [
            TaskStatus.DATA_PROCESSING,
            TaskStatus.MODEL_PREDICTION,
            TaskStatus.VISUALIZATION_PREPARATION,
        ]:
            stage_data = redis_conn.hgetall(f"task:{self.task_id}:stage:{stage}")

            if stage_data:
                stage_info = {}
                for k, v in stage_data.items():
                    key = k.decode("utf-8")
                    value = v.decode("utf-8") if v is not None else None

                    # Handle empty strings
                    if value == "":
                        if key in ["started_at", "completed_at", "error"]:
                            value = None
                        elif key == "result":
                            value = None

                    # Parse JSON result
                    if key == "result" and value:
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            pass

                    stage_info[key] = value
                stages[stage] = stage_info
            else:
                stages[stage] = {
                    "status": JobStatus.PENDING,
                    "started_at": None,
                    "completed_at": None,
                    "result": None,
                    "error": None,
                }

        return stages

    def start_data_processing(self) -> str:
        """Start the data processing stage.

        Returns:
            Job ID.
        """
        self.status = TaskStatus.DATA_PROCESSING
        self.stages["data_processing"]["status"] = JobStatus.RUNNING
        self.stages["data_processing"]["started_at"] = datetime.utcnow().isoformat() + "Z"

        # Create and enqueue data processing job
        self.data_processing_job = Job.create_data_processing(
            self.task_id, self.bboxes, self.parameters
        )
        job_id = self.data_processing_job.enqueue(timeout="2h")

        self.save()

        return job_id

    def start_model_prediction(self, processed_data: Dict[str, Any]) -> str:
        """Start the model prediction stage.

        Args:
            processed_data: Data from data processing stage.

        Returns:
            Job ID.
        """
        self.status = TaskStatus.MODEL_PREDICTION
        self.stages["model_prediction"]["status"] = JobStatus.RUNNING
        self.stages["model_prediction"]["started_at"] = datetime.utcnow().isoformat() + "Z"

        # Create and enqueue model prediction job
        self.model_prediction_job = Job.create_model_prediction(
            self.task_id, processed_data, self.parameters
        )
        job_id = self.model_prediction_job.enqueue(timeout="1h")

        self.save()

        return job_id

    def start_visualization_preparation(self, processed_data: Dict[str, Any]) -> str:
        """Start the visualization preparation stage.

        Args:
            processed_data: Data from model prediction stage.

        Returns:
            Job ID.
        """
        self.status = TaskStatus.VISUALIZATION_PREPARATION
        self.stages["visualization_preparation"]["status"] = JobStatus.RUNNING
        self.stages["visualization_preparation"]["started_at"] = datetime.utcnow().isoformat() + "Z"

        # Create and enqueue visualization preparation job
        self.visualization_preparation_job = Job.create_visualization_preparation(
            self.task_id, processed_data, self.parameters
        )
        job_id = self.visualization_preparation_job.enqueue(timeout="1h")

        self.save()

        return job_id

    def complete_stage(
        self,
        stage: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark a stage as completed.

        Args:
            stage: Stage name ('data_processing' or 'model_prediction').
            result: Stage result.
            error: Stage error.
        """
        if stage not in self.stages:
            raise ValueError(f"Invalid stage: {stage}")

        self.stages[stage]["status"] = JobStatus.FAILED if error else JobStatus.COMPLETED
        self.stages[stage]["completed_at"] = datetime.utcnow().isoformat() + "Z"
        self.stages[stage]["result"] = result
        self.stages[stage]["error"] = error

        # Update corresponding job status
        if stage == "data_processing" and self.data_processing_job:
            self.data_processing_job.update_status(
                JobStatus.FAILED if error else JobStatus.COMPLETED,
                result=result,
                error=error,
            )
        elif stage == "model_prediction" and self.model_prediction_job:
            self.model_prediction_job.update_status(
                JobStatus.FAILED if error else JobStatus.COMPLETED,
                result=result,
                error=error,
            )
        elif stage == "visualization_preparation" and self.visualization_preparation_job:
            self.visualization_preparation_job.update_status(
                JobStatus.FAILED if error else JobStatus.COMPLETED,
                result=result,
                error=error,
            )
        if error:
            self.status = TaskStatus.FAILED
        elif stage == "visualization_preparation" and not error:
            self.status = TaskStatus.COMPLETED
        self.save()

    def get_job_status(self, job_type: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job.

        Args:
            job_type: Job type ('data_processing' or 'model_prediction').

        Returns:
            Job status dictionary or None if job doesn't exist.
        """
        if job_type == TaskStatus.DATA_PROCESSING and self.data_processing_job:
            return self.data_processing_job.to_dict()
        elif job_type == TaskStatus.MODEL_PREDICTION and self.model_prediction_job:
            return self.model_prediction_job.to_dict()
        elif (
            job_type == TaskStatus.VISUALIZATION_PREPARATION and self.visualization_preparation_job
        ):
            return self.visualization_preparation_job.to_dict()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for API response.

        Returns:
            Task data as dictionary.
        """
        return {
            "task_id": self.task_id,
            "status": self.status,
            "created_at": self.created_at,
            "data_processing_job_id": (
                self.data_processing_job.job_id if self.data_processing_job else None
            ),
            "model_prediction_job_id": (
                self.model_prediction_job.job_id if self.model_prediction_job else None
            ),
            "bboxes": self.bboxes,
            "parameters": self.parameters,
            "stages": self.stages,
            "jobs": {
                TaskStatus.DATA_PROCESSING: self.get_job_status(TaskStatus.DATA_PROCESSING),
                TaskStatus.MODEL_PREDICTION: self.get_job_status(TaskStatus.MODEL_PREDICTION),
                TaskStatus.VISUALIZATION_PREPARATION: self.get_job_status(
                    TaskStatus.VISUALIZATION_PREPARATION
                ),
            },
        }

    @classmethod
    def get(cls, task_id: str) -> "Task":
        """Get an existing task by ID.

        Args:
            task_id: Task ID.

        Returns:
            Task instance.
        """
        task = cls(task_id, [], {}, is_new=False)
        task.load()
        return task


def process_data_extraction_with_task(
    task_id: str,
    bboxes: List[List[float]],
    parameters: Dict[str, Any],
) -> None:
    """Process data extraction with task tracking.

    Args:
        task_id: Task ID.
        bboxes: List of bounding box dictionaries.
        parameters: Optional processing parameters.

    Returns:
        Job result dictionary.
    """
    try:
        task = Task.get(task_id)

        logger.info(
            f"Processing data extraction for task {task_id} with {len(bboxes)} " f"bounding boxes"
        )

        # Extract data from bounding boxes using the DataProcessor
        processed_data = data_processor.extract_data_from_bboxes(
            task_id=task_id,
            bboxes=bboxes,
            parameters=parameters,
        )

        # Add bounding boxes to processed data for model prediction
        processed_data["bboxes"] = bboxes
        processed_data["parameters"] = parameters

        # Add data paths for model prediction (not exposed to frontend)
        data_path = data_processor.get_data_path()
        dataset_csv_path = data_processor.get_dataset_csv_path()

        if data_path:
            processed_data["data_path"] = data_path
        if dataset_csv_path:
            processed_data["dataset_csv_path"] = dataset_csv_path

        # Create safe results (without internal paths)
        safe_results = {
            "chips_created": processed_data.get("chips_created"),
            "chip_size": processed_data.get("chip_size"),
            "processing_duration": processed_data.get("processing_duration"),
            "data_source": processed_data.get("data_source"),
            "target_date": processed_data.get("target_date"),
            "temporal_tolerance": processed_data.get("temporal_tolerance"),
            "bboxes_processed": processed_data.get("bboxes_processed"),
        }

        # Mark data processing as completed
        task.complete_stage("data_processing", result=safe_results)

        # Check if data is ready for model prediction
        if data_processor.check_data_ready_for_model():
            # Start model prediction stage
            model_job_id = task.start_model_prediction(processed_data)
            processed_data["model_job_id"] = model_job_id
        else:
            logger.warning(f"Data not ready for model prediction for task {task_id}")
            processed_data["model_job_id"] = None

        logger.info(
            {
                "status": "success",
                "message": "Data extraction completed successfully.",
                "processed_data": safe_results,
                "completed_at": datetime.utcnow().isoformat() + "Z",
                "model_job_id": processed_data.get("model_job_id"),
            }
        )

    except Exception as e:
        logger.error(f"Data extraction failed for task {task_id}: {str(e)}")

        # Mark data processing as failed
        task = Task.get(task_id)
        task.complete_stage("data_processing", error=str(e))

        logger.error(
            {
                "status": "error",
                "message": f"Data extraction failed: {str(e)}",
                "completed_at": datetime.utcnow().isoformat() + "Z",
            }
        )


def process_model_prediction_with_task(
    task_id: str,
    processed_data: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
) -> None:
    """Process model prediction with task tracking.

    Args:
        task_id: Task ID.
        processed_data: Data from data processing stage.
        parameters: Optional model parameters.

    Returns:
        Job result dictionary.
    """
    try:
        task = Task.get(task_id)

        logger.debug(f"Running model prediction for task {task_id}")

        # TODO: Implement actual model prediction logic
        import time

        logger.info(f"Running model prediction for task {task_id}")
        logger.info(f"Processed data: {processed_data}")
        time.sleep(10)

        # Example prediction results
        prediction_results = {
            "data_path": processed_data["data_path"],
            "dataset_csv_path": processed_data["dataset_csv_path"],
            "aod_values": [0.15, 0.22, 0.18, 0.31],
            "confidence_scores": [0.85, 0.78, 0.92, 0.71],
            "bboxes": processed_data["bboxes"],
            "prediction_date": datetime.utcnow().isoformat() + "Z",
            "chip_size": processed_data["chip_size"],
        }

        # Mark model prediction as completed
        task.complete_stage("model_prediction", result=prediction_results)
        task.start_visualization_preparation(prediction_results)

        logger.info(
            {
                "status": "success",
                "message": "Model prediction completed successfully.",
                "results": prediction_results,
                "completed_at": datetime.utcnow().isoformat() + "Z",
            }
        )

    except Exception as e:
        # Mark model prediction as failed
        task = Task.get(task_id)
        task.complete_stage("model_prediction", error=str(e))

        logger.error(
            {
                "status": "error",
                "message": f"Model prediction failed: {str(e)}",
                "completed_at": datetime.utcnow().isoformat() + "Z",
            }
        )


def process_visualization_preparation_with_task(
    task_id: str,
    processed_data: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
) -> None:
    """Process visualization preparation with task tracking.

    Args:
        task_id: Task ID.
        processed_data: Data from model prediction stage.
        parameters: Optional model parameters.

    Returns:
        Job result dictionary.
    """
    try:
        task = Task.get(task_id)
        logger.info(f"Running visualization preparation for task {task_id}")
        logger.info(f"Processed data: {processed_data}")
        results = cog_converter.merge_task_files_to_cog(
            data_path=processed_data["data_path"],
            chip_size=processed_data["chip_size"],
            compute_seg_stats=(processed_data.get("model_type") == "seg"),
        )
        logger.info(f"Chips merged cog path: {results['chips_merged_cog_path']}")
        logger.info(f"Predictions merged cog path: {results['predictions_merged_cog_path']}")
        logger.info(f"Segmentation stats: {results['segmentation_stats']}")

        safe_results = {
            "viz_data_ready": True,
            "processing_duration": results["processing_duration"],
            "segmentation_stats": results["segmentation_stats"],
        }

        # Mark visualization preparation as completed
        task.complete_stage(TaskStatus.VISUALIZATION_PREPARATION, result=safe_results)

        logger.info(
            {
                "status": "success",
                "message": "Visualization preparation completed successfully.",
                "results": results,
                "completed_at": datetime.utcnow().isoformat() + "Z",
            }
        )

    except Exception as e:
        # Mark visualization preparation as failed
        task = Task.get(task_id)
        task.complete_stage("visualization_preparation", error=str(e))

        logger.error(
            {
                "status": "error",
                "message": f"Visualization preparation failed: {str(e)}",
                "completed_at": datetime.utcnow().isoformat() + "Z",
            }
        )
