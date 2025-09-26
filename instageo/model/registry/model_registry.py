"""Model registry for managing and accessing model metadata and configurations."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Union

from omegaconf import OmegaConf

from instageo.model.configs.config_dataclasses import ModelEnum, ModelInfo

logger = logging.getLogger(__name__)
MODELS_PATH = os.getenv("MODELS_PATH", "/app/models")


# Registry class
class ModelRegistry:
    """Model registry class for managing model metadata and configurations."""

    def __init__(self) -> None:
        """Initialize the model registry by loading the models.yaml file."""
        current_dir = Path(__file__).parent
        self.model_registry_path = current_dir / "models_registry.yaml"

        if not self.model_registry_path.exists():
            raise FileNotFoundError(f"Model registry not found at {self.model_registry_path}")

        self._all_models_metadata = OmegaConf.load(self.model_registry_path)
        logger.info(f"Model metadata: {self._all_models_metadata}")

    def get_model_metadata_for_size(
        self, model_key: Union[str, ModelEnum], model_size: str
    ) -> ModelInfo | None:
        """Get model metadata for a given model key and size."""
        if model_key not in self._all_models_metadata["models"]:
            raise ValueError(f"Model '{model_key}' not found in registry")

        model_data = self._all_models_metadata["models"][model_key]
        logger.info(f"Model data: {model_data}")

        size_specific_data = model_data["sizes"][model_size]

        model_config = self.get_model_config(model_key, model_size)
        logger.info(f"Model config: {model_config}")
        if model_config is None:
            return None
        classes_mapping = model_data.get("classes_mapping")
        if classes_mapping in ["None", "null", None]:
            classes_mapping = {}

        return ModelInfo(
            model_key=model_key,
            model_type=model_data.get("model_type", "unknown"),
            model_short_name=model_data.get("model_short_name", model_key),
            model_name=model_data.get("model_name", model_key),
            model_size=model_size,
            num_params=size_specific_data.get("num_params", 0.0),
            classes_mapping=classes_mapping,
            data_source=model_data.get("data_source", "unknown"),
            chip_size=model_config["dataloader"]["img_size"],
            num_steps=model_config["dataloader"]["temporal_dim"],
            temporal_step=model_data.get("temporal_step", 0),
            model_description=model_data.get("model_description", "unknown"),
        )

    def get_model_config(
        self, model_key: Union[str, ModelEnum], model_size: str
    ) -> Dict[str, Any] | None:
        """Get model configuration for a given model key and size."""
        model_path = os.path.join(
            MODELS_PATH, str(model_key), str(model_size), ".hydra/config.yaml"
        )
        logger.info(f"Model path: {model_path}")

        if not os.path.exists(model_path):
            return None
        return OmegaConf.load(model_path)

    def get_available_models(self) -> List[ModelInfo]:
        """Get list of all available models with their configurations."""
        models: List[ModelInfo] = []
        for key, model in self._all_models_metadata["models"].items():
            for size in model["sizes"].keys():
                model_metadata = self.get_model_metadata_for_size(key, size)
                if model_metadata is not None:
                    models.append(model_metadata)
        return models
