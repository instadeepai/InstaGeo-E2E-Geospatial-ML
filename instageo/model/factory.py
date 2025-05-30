"""Factory Module for Prithvi Models."""

from omegaconf import DictConfig

from instageo.model.base import PrithviBaseModule
from instageo.model.regression import PrithviRegressionModule
from instageo.model.segmentation import PrithviSegmentationModule


def create_model(cfg: DictConfig) -> PrithviBaseModule:
    """Create a model based on the configuration.

    Args:
        cfg (DictConfig): Configuration object containing model settings.

    Returns:
        PrithviBaseModule: The created model.
    """
    # Common parameters for both model types
    common_params = {
        "image_size": cfg.dataloader.img_size,
        "learning_rate": cfg.train.learning_rate,
        "freeze_backbone": cfg.model.freeze_backbone,
        "temporal_step": cfg.dataloader.temporal_dim,
        "ignore_index": cfg.train.ignore_index,
        "weight_decay": cfg.train.weight_decay,
        "model_name": cfg.model.model_name,
        "load_pretrained_weights": cfg.model.load_pretrained_weights,
        "scheduler": cfg.train.scheduler,
        "weight_clip_range": cfg.model.weight_clip_range,
    }

    if cfg.is_reg_task:
        # Regression-specific parameters
        return PrithviRegressionModule(
            **common_params,
            use_log_scale=cfg.model.use_log_scale,
            plot_reg_results=cfg.model.plot_reg_results,
            include_ee=cfg.model.include_ee_metric,
        )
    else:
        # Segmentation-specific parameters
        return PrithviSegmentationModule(
            **common_params,
            num_classes=cfg.model.num_classes,
            class_weights=cfg.train.class_weights,
        )
