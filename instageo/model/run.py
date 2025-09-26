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

"""Run Module Containing Training, Evaluation and Inference Logic."""

import json
import logging
import os
import time
from functools import partial

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from instageo.model.dataloader import process_and_augment, process_test
from instageo.model.factory import create_model
from instageo.model.infer_utils import chip_inference
from instageo.model.pipeline_utils import (
    check_required_flags,
    compute_stats,
    create_dataloader,
    create_instageo_dataset,
    create_trainer,
    eval_collate_fn,
    get_augmentations,
    get_device,
    infer_collate_fn,
    init_neptune_logger,
)
from instageo.model.utils import log_model_complexity

pl.seed_everything(seed=1042, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)


@hydra.main(config_path="configs", version_base=None, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runner Entry Point.

    Performs training, evaluation or inference/prediction depending on the selected mode.

    Arguments:
        cfg (DictConfig): Dict-like object containing necessary values used to configure runner.

    Returns:
        None.
    """
    log.info(f"Script: {__file__}")
    log.info(f"Imported hydra config:\n{OmegaConf.to_yaml(cfg)}")

    start_time = time.time()

    # Common configuration parameters
    MEAN = cfg.dataloader.mean
    STD = cfg.dataloader.std
    IM_SIZE = cfg.dataloader.img_size
    TEMPORAL_SIZE = cfg.dataloader.temporal_dim
    AUGMENTATIONS = get_augmentations(cfg)
    batch_size = cfg.train.batch_size
    root_dir = cfg.root_dir
    valid_filepath = cfg.valid_filepath
    train_filepath = cfg.train_filepath
    test_filepath = cfg.test_filepath

    if cfg.mode == "stats":
        train_dataset = create_instageo_dataset(
            train_filepath,
            root_dir,
            partial(
                process_and_augment,
                mean=[0] * len(MEAN),
                std=[1] * len(STD),
                temporal_size=TEMPORAL_SIZE,
                im_size=IM_SIZE,
                augmentations=None,
            ),
            cfg,
        )
        train_loader = create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_workers,
        )
        mean, std, class_weights = compute_stats(train_loader, is_reg_task=cfg.is_reg_task)
        print(json.dumps({"mean": mean, "std": std, "class_weights": class_weights}))
        exit(0)
    model = create_model(cfg)

    if cfg.mode == "train":
        check_required_flags(["root_dir", "train_filepath", "valid_filepath"], cfg)
        neptune_logger = init_neptune_logger(cfg)
        # Create train dataset
        train_dataset = create_instageo_dataset(
            train_filepath,
            root_dir,
            partial(
                process_and_augment,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                im_size=IM_SIZE,
                augmentations=AUGMENTATIONS,
                chip_no_data_value=cfg.dataloader.no_data_value,
                label_no_data_value=cfg.train.ignore_index,
                max_pixel_value=cfg.dataloader.max_pixel_value,
            ),
            cfg,
        )

        # Create validation dataset
        valid_dataset = create_instageo_dataset(
            valid_filepath,
            root_dir,
            partial(
                process_and_augment,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                im_size=IM_SIZE,
            ),
            cfg,
        )

        # Create dataloaders
        train_loader = create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_workers,
        )
        valid_loader = create_dataloader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
        )

        monitor = "val_RMSE" if cfg.is_reg_task else "val_IoU"
        mode = "min" if cfg.is_reg_task else "max"

        trainer = create_trainer(cfg, neptune_logger, monitor, mode)
        trainer.fit(model, train_loader, valid_loader)

        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        neptune_logger.experiment["model/Training duration"] = elapsed_time
        neptune_logger.experiment["model/One training epoch duration"] = (
            elapsed_time / cfg.train.num_epochs
        )

        log_model_complexity(model, cfg, neptune_logger)

    elif cfg.mode == "eval":
        check_required_flags(["root_dir", "test_filepath"], cfg)
        neptune_logger = init_neptune_logger(cfg, test_filepath)
        test_dataset = create_instageo_dataset(
            test_filepath,
            root_dir,
            partial(
                process_test,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                img_size=cfg.test.img_size,
                crop_size=cfg.test.crop_size,
                stride=cfg.test.stride,
            ),
            cfg,
            include_filenames=True,
        )

        test_loader = create_dataloader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=eval_collate_fn,
            num_workers=cfg.dataloader.num_workers,
        )

        trainer = create_trainer(cfg, neptune_logger)
        result = trainer.test(model, dataloaders=test_loader)
        log.info(f"Evaluation results:\n{result}")

        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        neptune_logger.experiment["model/Eval duration"] = elapsed_time

        log_model_complexity(model, cfg, neptune_logger)

    elif cfg.mode == "chip_inference":
        check_required_flags(["root_dir", "test_filepath", "checkpoint_path"], cfg)

        model.eval()

        output_dir = os.path.join(root_dir, "predictions")
        os.makedirs(output_dir, exist_ok=True)

        test_dataset = create_instageo_dataset(
            test_filepath,
            root_dir,
            partial(
                process_and_augment,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                im_size=cfg.test.img_size,
                augmentations=None,
            ),
            cfg,
            include_filenames=True,
        )

        test_loader = create_dataloader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=infer_collate_fn,
            num_workers=cfg.dataloader.num_workers,
        )

        carbon_info = chip_inference(test_loader, output_dir, model, device=get_device())
        print(f"Carbon tracking information: {carbon_info}")


if __name__ == "__main__":
    main()
