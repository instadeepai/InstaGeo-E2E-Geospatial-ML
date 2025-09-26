"""Data processing module for InstaGeo backend.

This module provides a proxy interface to the bounding boxes data pipeline for processing
satellite data extraction tasks. It handles folder structure, parameter mapping,
and integration with the task system.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from absl import flags

from instageo.data.raster_chip_creator import main as bbox_chip_creator

logger = logging.getLogger(__name__)


class DataProcessor:
    """Proxy class for bounding boxes data pipeline integration."""

    def __init__(self, base_output_dir: str = "/app/instageo-data"):
        """Initialize the data processor.

        Args:
            base_output_dir: Base directory for storing task outputs.
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def extract_data_from_bboxes(
        self,
        task_id: str,
        bboxes: List[List[float]],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract data from bounding boxes using the bounding boxes data pipeline.

        Args:
            task_id: Unique task identifier.
            bboxes: List of bounding boxes.
            parameters: Dictionary of processing parameters.

        Returns:
            Dictionary containing processing results and metadata.
        """
        start_time = datetime.now()
        try:
            logger.info(f"Starting data extraction for task {task_id}")

            # Create task-specific directory structure
            self.task_dir = self.base_output_dir / task_id
            self.data_dir = self.task_dir / "data"
            self.data_dir.mkdir(parents=True, exist_ok=True)

            # Run the bounding boxes data pipeline
            logger.info(f"Running bounding boxes data pipeline with: {bboxes=} and {parameters=}")
            pipeline_params = self._prepare_pipeline_params(bboxes, parameters)
            self._run_pipeline(pipeline_params)

            # Collect results
            results = self._collect_processing_results()

            # Add processing metadata
            end_time = datetime.now()
            processing_duration = (end_time - start_time).total_seconds()
            results["processing_duration"] = f"{processing_duration:.1f}"
            results["data_source"] = parameters.get("data_source", "unknown")
            results["bboxes_processed"] = len(bboxes)
            results["target_date"] = parameters.get("date", "unknown")
            results["temporal_tolerance"] = parameters.get("temporal_tolerance", "unknown")
            results["chip_size"] = parameters.get("chip_size", "unknown")

            logger.info(f"Data extraction completed for task {task_id}")
            return results

        except Exception as e:
            logger.error(f"Data extraction failed for task {task_id}: {str(e)}")
            raise RuntimeError(f"Failed to extract data from bounding boxes: {str(e)}")

    def _prepare_pipeline_params(
        self,
        bboxes: List[List[float]],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare parameters for bounding boxes data pipeline.

        Args:
            bboxes: List of bounding box coordinates.
            parameters: Processing parameters including date.

        Returns:
            Dictionary of parameters for bounding boxes data pipeline.
        """
        bbox_file = self.data_dir / "bounding_boxes.json"
        with open(bbox_file, "w") as f:
            json.dump(bboxes, f)

        params = {
            "is_bbox_feature": True,
            "bbox_feature_path": str(bbox_file),
            "output_directory": str(self.data_dir),
            "temporal_tolerance": parameters["temporal_tolerance"],
            "temporal_step": parameters["temporal_step"],
            "num_steps": parameters["num_steps"],
            "data_source": parameters["data_source"],
            "cloud_coverage": parameters["cloud_coverage"],
            "date": parameters["date"],
        }
        return params

    def _run_pipeline(self, params: Dict[str, Any]) -> None:
        """Run the bounding boxes data pipeline.

        Args:
            params: Parameters for bounding boxes data pipeline.
        """
        # Build command line arguments
        args = ["raster_chip_creator"]
        for key, value in params.items():
            args.extend([f"--{key}", f"{value}"])
        flags.FLAGS(args)
        bbox_chip_creator(None)

    def _collect_processing_results(self) -> Dict[str, Any]:
        """Collect results from the data processing.

        Returns:
            Dictionary with processing results and metadata.
        """
        results = {
            "chips_created": 0,
        }

        # Count chips
        chips_dir = self.data_dir / "chips"
        if chips_dir.exists():
            chip_files = list(chips_dir.glob("*.tif"))
            results["chips_created"] = len(chip_files)
            self.chips_created = True if results["chips_created"] > 0 else False
        return results

    def check_data_ready_for_model(self) -> bool:
        """Check if data is ready for model prediction.

        Returns:
            True if data is ready, False otherwise.
        """
        return self.chips_created

    def get_data_path(self) -> Optional[str]:
        """Get the data path for a task.

        Returns:
            Path to the data directory if it exists, None otherwise.
        """
        if self.data_dir.exists():
            return str(self.data_dir)
        return None

    def get_dataset_csv_path(self) -> Optional[str]:
        """Get the dataset CSV path for a task (for model prediction).

        Returns:
            Path to the dataset CSV if it exists, None otherwise.
        """
        dataset_csv = self.data_dir / "hls_raster_dataset.csv"

        if dataset_csv.exists():
            return str(dataset_csv)
        return None
