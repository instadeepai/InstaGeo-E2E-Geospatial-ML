"""Configuration dataclasses for InstaGeo models."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Model information for API responses and registry management."""

    model_key: str
    model_type: str
    model_short_name: str
    model_name: str
    model_size: str
    num_params: float
    classes_mapping: dict | None = None
    data_source: str | None = None
    chip_size: int = 224
    num_steps: int = 1
    temporal_step: int = 0
    model_description: str = ""


class ModelEnum(str, Enum):
    """Model names for list of available models."""

    PRITHVI_EO_TINY = "prithvi_eo_tiny"
    PRITHVI_EO_V1_100 = "prithvi_eo_v1_100"
    PRITHVI_EO_V2_100 = "prithvi_eo_v2_100"
    PRITHVI_EO_V2_300 = "prithvi_eo_v2_300"
    PRITHVI_EO_V2_300_TL = "prithvi_eo_v2_300_tl"
    PRITHVI_EO_V2_600 = "prithvi_eo_v2_600"
    PRITHVI_EO_V2_600_TL = "prithvi_eo_v2_600_tl"


class DataSourceEnum(str, Enum):
    """Data source names for list of data sources models are trained on."""

    HLS = "HLS"
    S1 = "S1"
    S2 = "S2"


@dataclass
class DataLoaderConfig:
    """Configuration for data loading and preprocessing."""

    bands: List[int] = field(default_factory=lambda: [1, 2, 3, 8, 11, 12])
    mean: List[float] = field(
        default_factory=lambda: [
            0.14245495,
            0.13921481,
            0.12434631,
            0.31420089,
            0.20743526,
            0.12046503,
        ]
    )
    std: List[float] = field(
        default_factory=lambda: [
            0.04036231,
            0.04186983,
            0.05267646,
            0.0822221,
            0.06834774,
            0.05294205,
        ]
    )
    img_size: int = 224
    temporal_dim: int = 1
    replace_label: List[int] = field(default_factory=lambda: [-1, 2])
    reduce_to_zero: bool = False
    no_data_value: int = -9999
    constant_multiplier: float = 1.0
    max_pixel_value: int = 10000
    num_workers: int = 1
    augmentations: Optional[List[Dict[str, Any]]] = None


@dataclass
class TrainConfig:
    """Configuration for training."""

    learning_rate: float = 0.0001
    num_epochs: int = 10
    batch_size: int = 8
    class_weights: List[float] = field(default_factory=lambda: [1, 1])
    ignore_index: int = -100
    weight_decay: float = 0.01
    scheduler: bool = False
    distillation: bool = False
    teacher_ckpt_path: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""

    model_name: ModelEnum = ModelEnum.PRITHVI_EO_TINY
    freeze_backbone: bool = False
    load_pretrained_weights: bool = True
    num_classes: int = 2
    use_log_scale: bool = False
    plot_reg_results: bool = False
    include_ee_metric: bool = False
    weight_clip_range: Optional[float] = None
    depth: int = -1


@dataclass
class NeptuneConfig:
    """Configuration for Neptune logging."""

    neptune_project: Optional[str] = None
    neptune_mode: str = "offline"
    neptune_experiment_id: Optional[str] = "InstaGeo-Inference-Server"


@dataclass
class AppConfig:
    """Configuration for app."""

    app_name: str = "instageo-inference-server"
    route_prefix: str = "/ray-infer"


@dataclass
class ChipInferenceConfig:
    """Configuration for chip inference."""

    # Core settings
    root_dir: str | Path = ""
    valid_filepath: str | Path = ""
    train_filepath: str | Path = ""
    test_filepath: str | Path = ""
    checkpoint_path: str | Path = ""
    data_source: DataSourceEnum = DataSourceEnum.HLS
    mode: str = "chip_inference"
    is_reg_task: bool = False

    # Component configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    neptune: NeptuneConfig = field(default_factory=NeptuneConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    app: AppConfig = field(default_factory=AppConfig)


def dict_to_chip_inference_config(config_dict: Dict[str, Any]) -> ChipInferenceConfig:
    """Convert dictionary to chip inference config dataclass."""
    # Handle nested configurations
    model_config = ModelConfig(**config_dict.get("model", {}))
    dataloader_config = DataLoaderConfig(**config_dict.get("dataloader", {}))
    train_config = TrainConfig(**config_dict.get("train", {}))

    # Convert Optional values to proper types
    root_dir = config_dict.get("root_dir")
    valid_filepath = config_dict.get("valid_filepath")
    train_filepath = config_dict.get("train_filepath")
    test_filepath = config_dict.get("test_filepath")
    checkpoint_path = config_dict.get("checkpoint_path")

    return ChipInferenceConfig(
        root_dir=root_dir if root_dir is not None else "",
        valid_filepath=valid_filepath if valid_filepath is not None else "",
        train_filepath=train_filepath if train_filepath is not None else "",
        test_filepath=test_filepath if test_filepath is not None else "",
        checkpoint_path=checkpoint_path if checkpoint_path is not None else "",
        data_source=config_dict.get("data_source", DataSourceEnum.HLS),
        mode=config_dict.get("mode", "chip_inference"),
        is_reg_task=config_dict.get("is_reg_task", False),
        model=model_config,
        dataloader=dataloader_config,
        neptune=NeptuneConfig(**config_dict.get("neptune", {})),
        train=train_config,
        app=AppConfig(**config_dict.get("app", {})),
    )


def get_dataloader_config(config_dict: Dict[str, Any]) -> DataLoaderConfig:
    """Get the dataloader configuration from the config dictionary."""
    return DataLoaderConfig(**config_dict.get("dataloader", {}))


def get_model_config(config_dict: Dict[str, Any]) -> ModelConfig:
    """Get the model configuration from the config dictionary."""
    return ModelConfig(**config_dict.get("model", {}))
