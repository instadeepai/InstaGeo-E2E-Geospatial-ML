# ------------------------------------------------------------------------------
# This code is licensed under the Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License.
#
# You are free to:
# - Share: Copy and redistribute the material in any medium or format
# - Adapt: Remix, transform, and build upon the material
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license,
#   and indicate if changes were made. You may do so in any reasonable manner,
#   but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
# - ShareAlike: If you remix, transform, or build upon the material, you must
#   distribute your contributions under the same license as the original.
#
# For more details, see https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------

"""Ray Serve Model Evaluation Pipeline.

This module provides a comprehensive class for running model evaluation using Ray Serve.
It handles the entire pipeline from configuration setup to cleanup.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import ray
from omegaconf import DictConfig
from ray import serve
from torch.utils.data import DataLoader

from instageo.model.configs.config_dataclasses import (
    ChipInferenceConfig,
    DataSourceEnum,
    TrainConfig,
)
from instageo.model.dataloader import process_and_augment
from instageo.model.model_server import RayModelServer
from instageo.model.neptune_logger import AIchorNeptuneLogger
from instageo.model.pipeline_utils import (
    check_required_flags,
    create_dataloader,
    create_instageo_dataset,
    infer_collate_fn,
    init_neptune_logger,
)

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent))
# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def create_evaluation_pipeline(
    root_dir: str | Path,
    test_filepath: str | Path,
    checkpoint_path: str | Path,
    mode: str = "chip_inference",
    is_reg_task: bool = False,
    data_source: DataSourceEnum = DataSourceEnum.HLS,
) -> "RayEvaluationPipeline":
    """Create an evaluation pipeline with the given configuration.

    Args:
        root_dir: Root directory for data
        test_filepath: Path to test data file
        checkpoint_path: Path to model checkpoint
        mode: Evaluation mode (eval or chip_inference)
        is_reg_task: Whether this is a regression task
        data_source: Data source type

    Returns:
        Configured RayEvaluationPipeline instance
    """
    # Create configuration with default values
    config = ChipInferenceConfig(
        root_dir=str(root_dir),
        test_filepath=str(test_filepath),
        checkpoint_path=str(checkpoint_path),
        mode=mode,
        is_reg_task=is_reg_task,
        data_source=data_source,
        train=TrainConfig(batch_size=128),  # Default batch size
    )

    return RayEvaluationPipeline(config)


class RayEvaluationPipeline:
    """A comprehensive class for running model evaluation using Ray Serve.

    This class handles the entire pipeline including:
    - Configuration management
    - Environment setup
    - Data preprocessing
    - Ray Serve deployment
    - Model evaluation
    - Results logging
    - Cleanup
    """

    def __init__(
        self,
        config: ChipInferenceConfig,
        processed_data: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> None:
        """Initialize the evaluation pipeline.

        Args:
            config: Configuration object containing all necessary parameters
            processed_data: Optional dictionary containing processed data from previous stages
            parameters: Optional dictionary containing additional parameters
            task_id: Optional task identifier for tracking purposes
        """
        self.config = config
        self.ray_initialized: bool = False
        self.serve_started: bool = False
        self.model_server_handle: serve.Deployment = None
        self.neptune_logger: Optional[AIchorNeptuneLogger] = None
        self.test_loader: Optional[DataLoader] = None
        self.start_time: float = 0.0
        self.processed_data = processed_data
        self.parameters = parameters
        self.task_id = task_id

    def start_evaluation_pipeline(self) -> None:
        """Start the evaluation pipeline."""
        log.info("Starting Ray Serve model evaluation pipeline...")

        self._setup_environment()
        self._validate_config()

        self.test_loader = self._setup_data_preprocessing(self.config)
        self.neptune_logger = self._setup_neptune_logger(self.config)
        self._initialize_ray_and_serve()
        self._deploy_model_server(self.config)

    def _setup_environment(self) -> None:
        """Set up environment variables and paths."""
        log.info("Setting up environment variables...")
        if self.config.neptune.neptune_project:
            os.environ["NEPTUNE_PROJECT"] = self.config.neptune.neptune_project
        os.environ["NEPTUNE_MODE"] = self.config.neptune.neptune_mode

    def _validate_config(self) -> None:
        """Validate the configuration and check required files."""
        log.info("Validating configuration...")
        required_paths = [
            self.config.root_dir,
            self.config.test_filepath,
            self.config.checkpoint_path,
        ]

        missing_files = []
        for path in required_paths:
            if not Path(path).exists():
                missing_files.append(path)

        if missing_files:
            raise FileNotFoundError(
                f"Missing required files/directories: {missing_files}. "
                "Please ensure all required files exist."
            )

        valid_modes = ["eval", "chip_inference"]
        if self.config.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.config.mode}'. Valid modes are: {valid_modes}")

    def _setup_data_preprocessing(self, cfg: DictConfig) -> Optional[DataLoader]:
        """Set up data preprocessing pipeline."""
        log.info("Setting up data preprocessing pipeline...")

        from functools import partial

        MEAN = cfg.dataloader.mean
        STD = cfg.dataloader.std
        IM_SIZE = cfg.dataloader.img_size
        TEMPORAL_SIZE = cfg.dataloader.temporal_dim
        batch_size = cfg.train.batch_size
        root_dir = self.config.root_dir

        log.info(f"Inference pipeline: config: {cfg}")
        process_fn = partial(
            process_and_augment,
            mean=MEAN,
            std=STD,
            temporal_size=TEMPORAL_SIZE,
            im_size=IM_SIZE,
            augmentations=None,
        )

        if cfg.mode == "eval":
            check_required_flags(["root_dir", "test_filepath"], cfg)
            test_dataset = create_instageo_dataset(
                str(cfg.test_filepath),
                str(root_dir),
                process_fn,
                cfg,
                include_filenames=False,
            )
            test_loader = create_dataloader(
                test_dataset,
                batch_size=batch_size,
                num_workers=cfg.dataloader.num_workers,
            )
        elif cfg.mode == "chip_inference":
            check_required_flags(["root_dir", "test_filepath", "checkpoint_path"], cfg)
            test_dataset = create_instageo_dataset(
                str(cfg.test_filepath),
                str(root_dir),
                process_fn,
                cfg,
                include_filenames=True,
            )
            test_loader = create_dataloader(
                test_dataset,
                batch_size=batch_size,
                collate_fn=infer_collate_fn,
                num_workers=cfg.dataloader.num_workers,
            )
        else:
            test_loader = None
        log.info(f"Test loader: {test_loader}, length: {len(test_loader)}")
        log.info("Data preprocessing pipeline setup completed.")
        return test_loader

    def _initialize_ray_and_serve(self) -> None:
        """Initialize Ray and Ray Serve."""
        log.info("Initializing Ray and Ray Serve...")

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            self.ray_initialized = True

        serve.start(detached=True)
        self.serve_started = True

        log.info("Ray and Ray Serve initialized successfully.")

    def _deploy_model_server(self, cfg: DictConfig) -> None:
        """Deploy the model server using Ray Serve."""
        log.info(f"Deploying model server as '{self.config.app.app_name}'...")

        app = RayModelServer.bind(cfg=cfg)
        serve.run(
            app,
            name=self.config.app.app_name,
            route_prefix=self.config.app.route_prefix,
        )

        # Get deployment handle
        self.model_server_handle = serve.get_deployment_handle(
            "RayModelServer", app_name=self.config.app.app_name
        )

        log.info(f"Model server deployed successfully at {self.config.app.app_name}")

    def _setup_neptune_logger(self, cfg: DictConfig) -> Optional[AIchorNeptuneLogger]:
        """Set up Neptune logger if configured."""
        if self.config.mode == "eval" and self.config.neptune.neptune_project:
            log.info("Setting up Neptune logger...")
            neptune_logger = init_neptune_logger(cfg, cfg.test_filepath)
            log.info("Neptune logger setup completed.")
            return neptune_logger
        return None

    def _run_evaluation(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Run the evaluation pipeline."""
        log.info("Starting evaluation pipeline...")

        if self.config.mode == "eval":
            results = self.model_server_handle.evaluate.remote(test_loader).result()
            return {"status": "evaluation_completed", "results": results}

        elif self.config.mode == "chip_inference":
            output_dir = os.path.join(self.config.root_dir, "predictions")
            results = self.model_server_handle.chip_inference.remote(test_loader, output_dir)
            results = results.result()
            # Ensure results is a dictionary
            if isinstance(results, dict):
                return results
            else:
                return {"status": "chip_inference_completed", "results": results}

        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}")

    def _log_results(self, results: Dict[str, Any]) -> None:
        """Log results to Neptune if available."""
        if self.neptune_logger and results:
            if self.config.mode == "eval":
                self.neptune_logger.log_metrics(results)

                elapsed_time = time.time() - self.start_time
                self.neptune_logger.experiment["model/Eval duration"] = elapsed_time

    def _cleanup(self) -> None:
        """Clean up Ray and Serve resources."""
        log.info("Cleaning up resources...")

        try:
            if self.serve_started:
                serve.shutdown()
                self.serve_started = False

            if self.ray_initialized:
                ray.shutdown()
                self.ray_initialized = False

            if self.neptune_logger:
                self.neptune_logger.experiment.get_root_object().stop()

        except Exception as e:
            log.warning(f"Error during cleanup: {e}")

    def evaluate(self) -> Dict[str, Any]:
        """Run the complete evaluation pipeline.

        Returns:
            Dictionary containing evaluation results
        """
        self.start_time = time.time()

        try:
            # Run evaluation
            log.info("Running evaluation...")
            test_loader = self._setup_data_preprocessing(self.config)
            results = self._run_evaluation(test_loader)

            # Log results
            log.info("Logging results...")
            self._log_results(results)

            elapsed_time = time.time() - self.start_time
            log.info(f"Evaluation pipeline completed in {elapsed_time:.2f} seconds.")

            return results

        except Exception as e:
            log.error(f"Error in evaluation pipeline: {str(e)}")
            raise
        finally:
            self._cleanup()

    def __enter__(self) -> "RayEvaluationPipeline":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self._cleanup()
