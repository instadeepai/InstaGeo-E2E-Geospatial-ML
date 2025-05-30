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
from collections import Counter
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import hydra
import neptune
import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from instageo.model.dataloader import (
    InstaGeoDataset,
    process_and_augment,
    process_data,
    process_test,
)
from instageo.model.factory import create_model
from instageo.model.infer_utils import chip_inference, sliding_window_inference
from instageo.model.neptune_logger import AIchorNeptuneLogger, set_neptune_api_token

pl.seed_everything(seed=1042, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)


def check_required_flags(required_flags: List[str], config: DictConfig) -> None:
    """Check if required flags are provided.

    Args:
        required_flags: A list of required command line arguments.

    Raises:
        An exception if at least one of the arguments is not set
    """
    for flag_name in required_flags:
        if getattr(config, flag_name) == "None":
            raise RuntimeError(f"Flag --{flag_name} is required.")


def get_device() -> str:
    """Selects available device."""
    try:
        import torch_xla.core.xla_model as xm  # noqa: F401

        device = "tpu"
        logging.info("TPU is available. Using TPU...")
    except ImportError:
        if torch.cuda.is_available():
            device = "gpu"
            logging.info("GPU is available. Using GPU...")
        else:
            device = "cpu"
            logging.info("Neither GPU nor TPU is available. Using CPU...")
    return device


def eval_collate_fn(batch: tuple[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluation DataLoader Collate Function.

    Args:
        batch (Tuple[Tensor]): A list of tuples containing features and labels.

    Returns:
        Tuple of (x,y) concatenated into separate tensors
    """
    data = torch.cat([a[0][0] for a in batch], 0)
    labels = torch.cat([a[0][1] for a in batch], 0)
    return data, labels


def infer_collate_fn(batch: tuple[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Inference DataLoader Collate Function.

    Args:
        batch (Tuple[Tensor]): A list of tuples containing features and labels.

    Returns:
        Tuple of (x,y) concatenated into separate tensors
    """
    data = torch.stack([a[0][0] for a in batch], 0)
    labels = [a[0][1] for a in batch]
    filepaths = [a[1] for a in batch]
    return (data, labels), filepaths


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int | None = 1,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for the given dataset.

    This function is a convenient wrapper around the PyTorch DataLoader class,
    allowing for easy setup of various DataLoader parameters.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): How many samples per batch to load.
        shuffle (bool): Set to True to have the data reshuffled at every epoch.
        num_workers (int): How many subprocesses to use for data loading.
        collate_fn (Optional[Callable]): Merges a list of samples to form a mini-batch.
        pin_memory (bool): If True, the data loader will copy tensors into CUDA pinned
            memory.

    Returns:
        DataLoader: An instance of the PyTorch DataLoader.
    """
    num_workers = num_workers if num_workers is not None else 1
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


def get_augmentations(cfg: DictConfig) -> Optional[List[Dict[str, Any]]]:
    """Get list of augmentations from config.

    Args:
        cfg (DictConfig): Configuration object containing augmentation settings.

    Returns:
        Optional[List[Dict[str, Any]]]: List of augmentation configurations to apply,
        or None if augmentations are disabled.
    """
    if cfg.dataloader.augmentations is None:
        return None

    augmentations = []
    for aug_name, aug_params in cfg.dataloader.augmentations.items():
        if not aug_params["use"]:
            continue
        aug_config = {"name": aug_name, "parameters": {"p": aug_params["p"]}}
        if aug_name == "rotate":
            aug_config["parameters"]["degrees"] = aug_params["degrees"]
        elif aug_name == "brightness":
            aug_config["parameters"]["brightness_range"] = aug_params[
                "brightness_range"
            ]
            aug_config["parameters"]["contrast_range"] = aug_params["contrast_range"]
        elif aug_name == "blur":
            aug_config["parameters"]["kernel_size"] = aug_params["kernel_size"]
            aug_config["parameters"]["sigma_range"] = tuple(aug_params["sigma_range"])
        elif aug_name == "noise":
            aug_config["parameters"]["noise_std"] = aug_params["noise_std"]
        elif (aug_name != "hflip") and (aug_name.strip() != "vflip"):
            log.warning(f"Unknown augmentation: {aug_name} skipping...")
            continue

        augmentations.append(aug_config)

    if len(augmentations) == 0:
        log.warning(
            "No valid augmentations specified. No augmentations will be applied."
        )
        return None

    return augmentations


def compute_class_weights(counts: dict[int, int]) -> list[float]:
    """Compute Class Weights.

    Args:
        counts (dict[int, int]): Dictionary mapping class index to counts.

    Returns:
        list[int]: List containing the class weight at the corresponding list index
    """
    total_samples = sum(counts.values())
    num_classes = len(counts)

    class_weights_dict = {}
    for cls, cnt in counts.items():
        class_weights_dict[cls] = total_samples / (num_classes * cnt)

    max_class_label = int(max(counts.keys()))
    class_weights_list = [0.0] * (max_class_label + 1)
    for cls, weight in class_weights_dict.items():
        class_weights_list[int(cls)] = weight
    return class_weights_list


def compute_stats(
    data_loader: DataLoader, is_reg_task: bool = False
) -> Tuple[List[float], List[float], List[float]]:
    """Compute the statistics of train dataset.

    Statistics computed includes mean, standard deviation and class weights.

    Args:
        data_loader (DataLoader): PyTorch DataLoader.
        is_reg_task (bool): Flag to indicate if the task is a regression task.

    Returns:
        mean (list): List of means for each channel.
        std (list): List of standard deviations for each channel.
        class_weights (list[float]): List of class weights.
    """
    mean = 0.0
    var = 0.0
    nb_samples = 0
    class_counts: dict[int, int] = Counter()
    for data, label in data_loader:
        # Reshape data to (B, C, T*H*W)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)

        nb_samples += batch_samples

        # Sum over batch, timestep, height and width
        mean += data.mean(2).sum(0)

        var += data.var(2, unbiased=False).sum(0)
        if not is_reg_task:
            class_counts.update(
                {k: v for k, v in zip(*np.unique(label.numpy(), return_counts=True))}
            )
    mean /= nb_samples
    var /= nb_samples
    std = torch.sqrt(var)
    class_weights = None
    if not is_reg_task:
        try:
            class_counts.pop(-1)  # remove ignore_index
        except KeyError:
            logging.info("No label pixel with value -1")
        logging.info(f"Class Counts: {class_counts}")
        class_weights = compute_class_weights(class_counts)
        logging.info(f"Normalized Counts: {class_weights}")
    return mean.tolist(), std.tolist(), class_weights  # type:ignore


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

    BANDS = cfg.dataloader.bands
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
    checkpoint_path = cfg.checkpoint_path

    if cfg.mode == "stats":
        train_dataset = InstaGeoDataset(
            filename=train_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_and_augment,
                mean=[0] * len(MEAN),
                std=[1] * len(STD),
                temporal_size=TEMPORAL_SIZE,
                im_size=IM_SIZE,
                augmentations=None,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            chip_no_data_value=cfg.dataloader.no_data_value,
            label_no_data_value=cfg.train.ignore_index,
            constant_multiplier=cfg.dataloader.constant_multiplier,
        )
        train_loader = create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_workers,
        )
        mean, std, class_weights = compute_stats(
            train_loader, is_reg_task=cfg.is_reg_task
        )
        print(json.dumps({"mean": mean, "std": std, "class_weights": class_weights}))
        exit(0)

    elif cfg.mode == "train":
        check_required_flags(["root_dir", "train_filepath", "valid_filepath"], cfg)
        neptune_run = neptune.init_run(
            api_token=set_neptune_api_token(),
            project=os.environ["NEPTUNE_PROJECT"],
            with_id=(
                cfg.neptune_experiment_id
                if hasattr(cfg, "neptune_experiment_id")
                else None
            ),
        )
        neptune_logger = AIchorNeptuneLogger(run=neptune_run)
        neptune_logger.experiment["config"] = stringify_unsupported(
            OmegaConf.to_yaml(cfg)
        )
        train_dataset = InstaGeoDataset(
            filename=train_filepath,
            input_root=root_dir,
            preprocess_func=partial(
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
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            chip_no_data_value=cfg.dataloader.no_data_value,
            label_no_data_value=cfg.train.ignore_index,
            constant_multiplier=cfg.dataloader.constant_multiplier,
        )

        valid_dataset = InstaGeoDataset(
            filename=valid_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_and_augment,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                im_size=IM_SIZE,
                augmentations=None,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            chip_no_data_value=cfg.dataloader.no_data_value,
            label_no_data_value=cfg.train.ignore_index,
            constant_multiplier=cfg.dataloader.constant_multiplier,
        )
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
        model = create_model(cfg)
        hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        checkpoint_callback = ModelCheckpoint(
            monitor="val_RMSE" if cfg.is_reg_task else "val_IoU",
            dirpath=hydra_out_dir,
            filename="instageo_best_checkpoint",
            auto_insert_metric_name=False,
            mode="min" if cfg.is_reg_task else "max",
            save_top_k=1,
        )

        trainer = pl.Trainer(
            accelerator=get_device(),
            max_epochs=cfg.train.num_epochs,
            callbacks=[checkpoint_callback],
            logger=neptune_logger,
            deterministic=True,
        )

        # run training and validation
        trainer.fit(model, train_loader, valid_loader)

    elif cfg.mode == "eval":
        check_required_flags(["root_dir", "test_filepath", "checkpoint_path"], cfg)
        neptune_run = neptune.init_run(
            api_token=set_neptune_api_token(),
            project=os.environ["NEPTUNE_PROJECT"],
            with_id=(
                cfg.neptune_experiment_id
                if hasattr(cfg, "neptune_experiment_id")
                else None
            ),
        )
        neptune_logger = AIchorNeptuneLogger(
            run=neptune_run[
                f"eval-{os.path.splitext(os.path.basename(test_filepath))[0]}"
            ]
        )
        neptune_logger.experiment["config"] = stringify_unsupported(
            OmegaConf.to_yaml(cfg)
        )
        test_dataset = InstaGeoDataset(
            filename=test_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_test,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                img_size=cfg.test.img_size,
                crop_size=cfg.test.crop_size,
                stride=cfg.test.stride,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            chip_no_data_value=cfg.dataloader.no_data_value,
            label_no_data_value=cfg.train.ignore_index,
            constant_multiplier=cfg.dataloader.constant_multiplier,
            include_filenames=True,
        )
        test_loader = create_dataloader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=eval_collate_fn,
            num_workers=cfg.dataloader.num_workers,
        )
        model = create_model(cfg)
        model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
        trainer = pl.Trainer(
            accelerator=get_device(),
            logger=neptune_logger,
        )
        result = trainer.test(model, dataloaders=test_loader)
        log.info(f"Evaluation results:\n{result}")

    elif cfg.mode == "sliding_inference":
        model = create_model(cfg)
        model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
        model.eval()
        infer_filepath = os.path.join(root_dir, cfg.test_filepath)
        assert (
            os.path.splitext(infer_filepath)[-1] == ".json"
        ), f"Test file path expects a json file but got {infer_filepath}"
        output_dir = os.path.join(root_dir, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(infer_filepath)) as json_file:
            hls_dataset = json.load(json_file)
        for key, hls_tile_path in tqdm(
            hls_dataset.items(), desc="Processing HLS Dataset"
        ):
            try:
                hls_tile, _ = process_data(
                    hls_tile_path,
                    None,
                    bands=cfg.dataloader.bands,
                    no_data_value=cfg.dataloader.no_data_value,
                    constant_multiplier=cfg.dataloader.constant_multiplier,
                    mask_cloud=cfg.test.mask_cloud,
                    replace_label=cfg.dataloader.replace_label,
                    reduce_to_zero=cfg.dataloader.reduce_to_zero,
                )
            except rasterio.RasterioIOError:
                continue
            nan_mask = hls_tile == cfg.dataloader.no_data_value
            nan_mask = np.any(nan_mask, axis=0).astype(int)
            hls_tile, _ = process_and_augment(
                hls_tile,
                None,
                mean=cfg.dataloader.mean,
                std=cfg.dataloader.std,
                temporal_size=cfg.dataloader.temporal_dim,
                augmentations=None,
            )
            prediction = sliding_window_inference(
                hls_tile,
                model,
                window_size=(cfg.test.img_size, cfg.test.img_size),
                stride=cfg.test.stride,
                batch_size=cfg.train.batch_size,
                device=get_device(),
            )
            prediction = np.where(nan_mask == 1, np.nan, prediction)
            prediction_filename = os.path.join(output_dir, f"{key}_prediction.tif")
            with rasterio.open(hls_tile_path["tiles"]["B02_0"]) as src:
                crs = src.crs
                transform = src.transform
            with rasterio.open(
                prediction_filename,
                "w",
                driver="GTiff",
                height=prediction.shape[0],
                width=prediction.shape[1],
                count=1,
                dtype=str(prediction.dtype),
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(prediction, 1)

    # TODO: Add support for chips that are greater than image size used for training
    elif cfg.mode == "chip_inference":
        check_required_flags(["root_dir", "test_filepath", "checkpoint_path"], cfg)
        output_dir = os.path.join(root_dir, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        test_dataset = InstaGeoDataset(
            filename=test_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_and_augment,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                im_size=cfg.test.img_size,
                augmentations=None,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            chip_no_data_value=cfg.dataloader.no_data_value,
            label_no_data_value=cfg.train.ignore_index,
            constant_multiplier=cfg.dataloader.constant_multiplier,
            include_filenames=True,
        )
        test_loader = create_dataloader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=infer_collate_fn,
            num_workers=cfg.dataloader.num_workers,
        )
        model = create_model(cfg)
        model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
        chip_inference(test_loader, output_dir, model, device=get_device())


if __name__ == "__main__":
    main()
