"""Task management for InstaGeo backend."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from instageo.model.configs.config_dataclasses import dict_to_chip_inference_config
from instageo.model.inference_pipeline import RayEvaluationPipeline
from instageo.model.registry.model_registry import ModelRegistry
from instageo.new_apps.backend.app.cog_converter import COGConverter
from instageo.new_apps.backend.app.data_processor import DataProcessor
from instageo.new_apps.backend.app.jobs import Job, JobStatus
from instageo.new_apps.backend.app.redis_client import redis_client

# Configure logging
logger = logging.getLogger(__name__)

DATA_FOLDER = os.getenv("DATA_FOLDER", "/app/instageo-data")

# Initialize data processor
data_processor = DataProcessor(base_output_dir=DATA_FOLDER)

# Initialize cog converter
cog_converter = COGConverter()
MODELS_PATH = os.getenv("MODELS_PATH", "/app/models")


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
        self.task_id = task_id
        self.bboxes = bboxes
        self.parameters = parameters or {}
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.created_timestamp = (
            datetime.fromisoformat(self.created_at).replace(tzinfo=None).timestamp()
        )

        self.data_processing_job: Optional[Job] = None
        self.model_prediction_job: Optional[Job] = None
        self.visualization_preparation_job: Optional[Job] = None

        self.status = TaskStatus.DATA_PROCESSING

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
            "created_timestamp": str(self.created_timestamp),
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
        redis_client.save_task(self.task_id, task_meta)
        # Add to sorted set for efficient date sorting
        redis_client.add_task_to_sorted_set(self.task_id, self.created_timestamp)

        # Store stages information
        for stage_name, stage_data in self.stages.items():
            stage_updates = {
                "status": stage_data["status"],
                "started_at": stage_data["started_at"] or "",
                "completed_at": stage_data["completed_at"] or "",
                "result": (json.dumps(stage_data["result"]) if stage_data["result"] else ""),
                "error": stage_data["error"] or "",
            }
            redis_client.save_task_stage(self.task_id, stage_name, stage_updates)

    def load(self) -> None:
        """Load task from Redis."""
        task_data = redis_client.load_task(self.task_id)
        logger.debug(f"Loading Task {self.task_id} from Redis: {task_data}")

        if not task_data:
            raise ValueError(f"Task {self.task_id} not found")

        # Parse task data
        for key, value in task_data.items():
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
            stage_data = redis_client.load_task_stage(self.task_id, stage)

            if stage_data:
                stage_info = {}
                for key, value in stage_data.items():
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
        self.stages["data_processing"]["started_at"] = datetime.now(timezone.utc).isoformat()

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
        self.stages["model_prediction"]["started_at"] = datetime.now(timezone.utc).isoformat()

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
        self.stages["visualization_preparation"]["started_at"] = datetime.now(
            timezone.utc
        ).isoformat()

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
        self.stages[stage]["completed_at"] = datetime.now(timezone.utc).isoformat()
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

        # Persist to database if task is completed or failed
        if self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            self._persist_to_database()

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

    def _persist_to_database(self) -> None:
        """Persist complete task metadata to database."""
        from instageo.new_apps.backend.app.crud import update_task_metadata
        from instageo.new_apps.backend.app.db import SessionLocal

        db = SessionLocal()
        try:
            success = update_task_metadata(
                db=db,
                task_id=self.task_id,
                status=self.status,
                stages=self.stages,
                model_short_name=self.parameters.get("model_short_name"),
                model_type=self.parameters.get("model_type"),
                model_name=self.parameters.get("model_name"),
                model_size=self.parameters.get("model_size"),
            )
            if success:
                logger.info(f"Persisted task {self.task_id} to database")
            else:
                logger.error(f"Failed to persist task {self.task_id} to database")
        finally:
            db.close()

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
        """Get an existing task by ID with Redis-first, database fallback.

        Args:
            task_id: Task ID.

        Returns:
            Task instance.
        """
        try:
            task = cls(task_id, [], {}, is_new=False)
            task.load()

            if task.stages and task.bboxes:
                return task
        except Exception as e:
            logger.warning(f"Failed to load task {task_id} from Redis: {e}")

        # Fallback to database
        logger.info(f"Loading task {task_id} from database")
        return cls._load_from_database(task_id)

    @classmethod
    def _load_from_database(cls, task_id: str) -> "Task":
        """Load task from database."""
        from instageo.new_apps.backend.app.crud import get_task_by_id
        from instageo.new_apps.backend.app.db import SessionLocal

        db = SessionLocal()
        try:
            db_task = get_task_by_id(db, task_id)
            if not db_task:
                raise ValueError(f"Task {task_id} not found in database")

            # Reconstruct task from database
            task = cls(task_id, [], {}, is_new=False)
            task.bboxes = json.loads(db_task.bboxes) if db_task.bboxes else []
            task.parameters = json.loads(db_task.parameters) if db_task.parameters else {}
            task.status = db_task.status
            task.created_at = db_task.created_at.isoformat()
            task.stages = json.loads(db_task.stages) if db_task.stages else {}

            return task
        finally:
            db.close()


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

        # Check if data is ready for model prediction
        if data_processor.check_data_ready_for_model():
            # Mark data processing as completed
            task.complete_stage("data_processing", result=safe_results)
            # Start model prediction stage
            model_job_id = task.start_model_prediction(processed_data)
            processed_data["model_job_id"] = model_job_id
        else:
            task.complete_stage("data_processing", error="Failed to extract data")
            logger.warning(f"Data not ready for model prediction for task {task_id}")
            processed_data["model_job_id"] = None

        logger.info(
            {
                "status": "success",
                "message": "Data extraction completed successfully.",
                "processed_data": safe_results,
                "completed_at": datetime.now(timezone.utc).isoformat(),
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
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )


def process_model_prediction_with_task(
    task_id: str,
    processed_data: Dict[str, Any],
    parameters: Dict[str, Any],
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

        model_type = parameters.get("model_type", "reg")
        model_key = parameters.get("model_key", "aod-estimator")
        model_size = parameters.get("model_size", "tiny")

        logger.info(f"Using model: {model_key}, type: {model_type}, size: {model_size}")

        model_registry: ModelRegistry = ModelRegistry()  # type: ignore
        model_metadata = model_registry.get_model_metadata_for_size(model_key, model_size)

        if model_metadata is None:
            raise ValueError(f"Model metadata not found for {model_key}/{model_size}")

        inference_configs = model_registry.get_model_config(model_key, model_size)
        model_path = os.path.join(MODELS_PATH, str(model_key), str(model_size))

        if inference_configs is None:
            raise ValueError(f"No configuration found for model {model_key}/{model_size}")

        inference_configs.update(
            {
                "root_dir": Path(str(processed_data.get("data_path"))),
                "test_filepath": Path(str(processed_data.get("dataset_csv_path"))),
                "mode": "chip_inference",
                "checkpoint_path": os.path.join(model_path, "instageo_best_checkpoint.ckpt"),
            }
        )
        configs = dict_to_chip_inference_config(inference_configs)

        logger.info(f"Running model prediction for task {task_id}")
        logger.info(f"Processed data: {processed_data}")

        pipeline = RayEvaluationPipeline(configs, processed_data, parameters, task_id)
        pipeline.start_evaluation_pipeline()

        try:
            raw_results = pipeline.evaluate()
        except Exception as pipeline_error:
            logger.error(f"Pipeline evaluation failed: {pipeline_error}")
            raise ValueError(f"Model prediction pipeline failed: {pipeline_error}")

        # Ensure results is a dictionary
        results_dict: Dict[str, Any] = (
            {} if raw_results is None or not isinstance(raw_results, dict) else raw_results
        )

        safe_results = {
            "classes_mapping": model_metadata.classes_mapping,
            "model/GFLOPs": results_dict.get("model/GFLOPs"),  # type: ignore[union-attr]
            "CO2_emissions": results_dict.get("CO2_emissions"),  # type: ignore[union-attr]
            "energy_consumed": results_dict.get("energy_consumed"),  # type: ignore[union-attr]
            "inference_time": results_dict.get("inference_time"),  # type: ignore[union-attr]
        }

        logger.info(f"Prediction results: {results_dict}")
        task.complete_stage("model_prediction", result=safe_results)

        # Start visualization preparation stage
        viz_job_id = task.start_visualization_preparation(processed_data)
        logger.info(f"Started visualization preparation job: {viz_job_id}")

        logger.info(
            {
                "status": "success",
                "message": "Model prediction completed successfully.",
                "results": safe_results,
                "completed_at": datetime.now(timezone.utc).isoformat(),
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
                "completed_at": datetime.now(timezone.utc).isoformat(),
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
            compute_seg_stats=(processed_data["parameters"].get("model_type") == "seg"),
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
                "completed_at": datetime.now(timezone.utc).isoformat(),
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
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
