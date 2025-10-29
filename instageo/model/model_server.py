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

"""InstaGeo Model Server for Ray Serve deployment."""

import logging
import os
import time
from typing import Any, Dict, Optional

import matplotlib
import torch
from omegaconf import DictConfig
from ray import serve
from ray.serve import Deployment
from torch.utils.data import DataLoader

from instageo.model.factory import create_model
from instageo.model.infer_utils import chip_inference
from instageo.model.pipeline_utils import create_trainer, get_device
from instageo.model.utils import get_model_complexity

matplotlib.use("Agg")

log = logging.getLogger(__name__)


@serve.deployment
class RayModelServer(Deployment):
    """InstaGeo Model Server for inference and evaluation."""

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        """Initialize the model server."""
        self.cfg = cfg

        self.device = get_device()

        self.model: Optional[Any] = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the model and load weights if provided."""
        try:
            log.info(f"Initializing model on device: {self.device}")
            self.model = create_model(self.cfg)

            if self.model is not None:
                self.model.to(self.device).eval()
        except Exception as e:
            log.error(f"Failed to initialize model: {str(e)}")
            raise

    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate the model using PyTorch Lightning trainer."""
        if not self.model:
            raise RuntimeError("Model not initialized")

        log.info("Starting model evaluation")
        start_time = time.time()

        try:
            trainer = create_trainer(self.cfg, logger=None)
            results = trainer.test(self.model, dataloaders=test_loader)
            evaluation_time = time.time() - start_time
            log.info(f"Evaluation completed in {evaluation_time:.2f}s")

            return results
        except Exception as e:
            log.error(f"Evaluation failed: {str(e)}")
            raise

    def chip_inference(self, test_loader: DataLoader, output_dir: str) -> Dict[str, Any]:
        """Run chip-based inference and save results."""
        if not self.model:
            raise RuntimeError("Model not initialized")

        log.info(f"Starting chip inference, output dir: {output_dir}")
        start_time = time.time()

        try:
            os.makedirs(output_dir, exist_ok=True)
            emissions_data = chip_inference(
                test_loader,
                output_dir,
                self.model,
                self.device,
            )
            inference_time = time.time() - start_time
            log.info(f"Chip inference completed in {inference_time:.2f}s")

        except Exception as e:
            log.error(f"Chip inference failed: {str(e)}")
            raise

        try:
            device_idx = 0 if self.device == "gpu" else 0
            complexity = get_model_complexity(self.model, self.cfg, device_idx)
        except Exception as e:
            log.warning(f"Could not calculate model complexity: {str(e)}")
            complexity = {"GFLOPs": "N/A", "Params": "N/A"}

        return {
            "status": "chip_inference_completed",
            "model/GFLOPs": complexity["GFLOPs"],
            "CO2_emissions": emissions_data.get("carbon/emissions (g CO₂)", "N/A"),
            "energy_consumed": emissions_data.get("carbon/energy_consumed (kWh)", "N/A"),
            "inference_time": inference_time,
        }

    def health_check(self) -> Dict[str, Any]:
        """Health check on the model server."""
        return {
            "status": "healthy" if self.model else "unhealthy",
            "model_loaded": self.model is not None,
            "device_info": self.get_device_info(),
        }

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        device_info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

        if torch.cuda.is_available():
            device_info.update(
                {
                    "cuda_device_name": torch.cuda.get_device_name(),
                    "cuda_memory_allocated": torch.cuda.memory_allocated(),
                    "cuda_memory_reserved": torch.cuda.memory_reserved(),
                }
            )

        return device_info
