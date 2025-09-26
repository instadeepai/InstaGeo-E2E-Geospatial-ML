"""Factory Module for Prithvi Models."""

import torch
from omegaconf import DictConfig

from instageo.model.base import PrithviBaseModule
from instageo.model.regression import (
    PrithviDistillationRegressionModule,
    PrithviRegressionModule,
)
from instageo.model.segmentation import (
    PrithviDistillationSegmentationModule,
    PrithviSegmentationModule,
)


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
        "scheduler": cfg.train.scheduler,
        "weight_clip_range": cfg.model.weight_clip_range,
        "depth": cfg.model.depth,
    }
    if cfg.mode == "train":
        if cfg.is_reg_task:
            # Regression-specific parameters

            if cfg.train.distillation:
                model = PrithviDistillationRegressionModule(
                    teacher_ckpt_path=cfg.train.teacher_ckpt_path,
                    **common_params,
                    load_pretrained_weights=cfg.model.load_pretrained_weights,
                    use_log_scale=cfg.model.use_log_scale,
                    plot_reg_results=cfg.model.plot_reg_results,
                    include_ee=cfg.model.include_ee_metric,
                )
            else:
                model = PrithviRegressionModule(
                    **common_params,
                    load_pretrained_weights=cfg.model.load_pretrained_weights,
                    use_log_scale=cfg.model.use_log_scale,
                    plot_reg_results=cfg.model.plot_reg_results,
                    include_ee=cfg.model.include_ee_metric,
                )

        else:
            # Segmentation-specific parameters
            if cfg.train.distillation:
                model = PrithviDistillationSegmentationModule(
                    teacher_ckpt_path=cfg.train.teacher_ckpt_path,
                    **common_params,
                    load_pretrained_weights=cfg.model.load_pretrained_weights,
                    num_classes=cfg.model.num_classes,
                    class_weights=cfg.train.class_weights,
                )
            else:
                model = PrithviSegmentationModule(
                    **common_params,
                    load_pretrained_weights=cfg.model.load_pretrained_weights,
                    num_classes=cfg.model.num_classes,
                    class_weights=cfg.train.class_weights,
                )
    else:
        if cfg.is_reg_task:
            model = PrithviRegressionModule(
                **common_params,
                use_log_scale=cfg.model.use_log_scale,
                plot_reg_results=cfg.model.plot_reg_results,
                load_pretrained_weights=False,
                include_ee=cfg.model.include_ee_metric,
            )
        else:
            model = PrithviSegmentationModule(
                **common_params,
                num_classes=cfg.model.num_classes,
                class_weights=cfg.train.class_weights,
                load_pretrained_weights=False,
            )

        model.load_state_dict(
            torch.load(cfg.checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        )
    return model
