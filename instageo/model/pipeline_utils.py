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

"""Pipeline utilities for InstaGeo model training, evaluation and inference."""

import logging
import os
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import hydra
import neptune
import numpy as np
import pytorch_lightning as pl
import torch
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from instageo.model.dataloader import InstaGeoDataset
from instageo.model.neptune_logger import AIchorNeptuneLogger, set_neptune_api_token
from instageo.model.utils import CarbonTrackerCallback

log = logging.getLogger(__name__)


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
        elif torch.backends.mps.is_available():
            device = "mps"
            logging.info("MPS is available. Using MPS...")
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
            aug_config["parameters"]["brightness_range"] = aug_params["brightness_range"]
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
        log.warning("No valid augmentations specified. No augmentations will be applied.")
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
    return mean.tolist(), std.tolist(), class_weights  # type: ignore


def create_instageo_dataset(
    filename: str,
    root_dir: str,
    preprocess_func: Callable,
    cfg: DictConfig,
    include_filenames: bool = False,
) -> InstaGeoDataset:
    """Create an InstaGeoDataset with common parameters.

    Args:
        filename (str): Path to the dataset file
        root_dir (str): Root directory for data
        preprocess_func (Callable): Function to preprocess data
        cfg (DictConfig): Configuration object
        include_filenames (bool): Whether to include filenames in dataset

    Returns:
        InstaGeoDataset: Created dataset
    """
    # Get config loader for fallback values

    return InstaGeoDataset(
        filename=filename,
        input_root=root_dir,
        preprocess_func=preprocess_func,
        bands=cfg.dataloader.bands,
        replace_label=cfg.dataloader.replace_label,
        reduce_to_zero=cfg.dataloader.reduce_to_zero,
        chip_no_data_value=cfg.dataloader.no_data_value,
        label_no_data_value=cfg.train.ignore_index,
        constant_multiplier=cfg.dataloader.constant_multiplier,
        include_filenames=include_filenames,
    )


def init_neptune_logger(cfg: DictConfig, test_filepath: str | None = None) -> AIchorNeptuneLogger:
    """Initialize Neptune logger with common parameters.

    Args:
        cfg (DictConfig): Configuration object
        test_filepath (str | None): Optional test filepath for evaluation runs

    Returns:
        AIchorNeptuneLogger: Initialized logger
    """
    # Check if we're in offline mode
    neptune_mode = os.environ.get("NEPTUNE_MODE", "online")

    if neptune_mode == "offline":
        # In offline mode, create a new run without specific ID
        neptune_run = neptune.init_run(
            api_token=set_neptune_api_token(),
            project=os.environ["NEPTUNE_PROJECT"],
        )
    else:
        # In online mode, use the experiment ID if available
        with_id = None
        if hasattr(cfg, "neptune_experiment_id"):
            with_id = cfg.neptune_experiment_id

        neptune_run = neptune.init_run(
            api_token=set_neptune_api_token(),
            project=os.environ["NEPTUNE_PROJECT"],
            with_id=with_id,
        )

    if test_filepath:
        neptune_run = neptune_run[f"eval-{os.path.splitext(os.path.basename(test_filepath))[0]}"]
    neptune_logger = AIchorNeptuneLogger(run=neptune_run, log_model_checkpoints=False)
    neptune_logger.experiment["config"] = stringify_unsupported(OmegaConf.to_yaml(cfg))
    return neptune_logger


def create_trainer(
    cfg: DictConfig,
    logger: Optional[AIchorNeptuneLogger] = None,
    monitor: str = "val_IoU",
    mode: str = "max",
) -> pl.Trainer:
    """Create a PyTorch Lightning trainer with common parameters.

    Args:
        cfg (DictConfig): Configuration object
        logger (Optional[AIchorNeptuneLogger]): Neptune logger (optional)
        monitor (str): Metric to monitor for checkpointing
        mode (str): Mode for checkpointing (min/max)

    Returns:
        pl.Trainer: Created trainer
    """
    hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=hydra_out_dir,
        filename="instageo_best_checkpoint",
        auto_insert_metric_name=False,
        mode=mode,
        save_top_k=1,
    )

    callbacks = [checkpoint_callback]

    # Add carbon tracking callback only if logger is provided
    if logger is not None:
        carbon_cb = CarbonTrackerCallback(
            total_epochs=cfg.train.num_epochs if hasattr(cfg, "train") else None,
            neptune_run=logger,
            log_dir=hydra_out_dir,
        )
        callbacks.append(carbon_cb)

    return pl.Trainer(
        accelerator=get_device(),
        max_epochs=cfg.train.num_epochs if hasattr(cfg, "train") else None,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
    )


def create_hydra_config(
    hydra_config_path: Optional[str] = None,
    override_config: Optional[DictConfig] = None,
) -> DictConfig:
    """Create Hydra configuration from model metadata.

    Args:
        hydra_config_path: Optional custom config path, defaults to model metadata config_path
        override_config: Optional configuration overrides

    Returns:
        DictConfig: Hydra configuration object
    """
    # Use provided config_path or fall back to metadata config_path
    if hydra_config_path is None:
        raise ValueError("Hydra config path is required")

    # Initialize Hydra with the configuration
    with hydra.initialize_config_dir(config_dir=str(hydra_config_path), version_base=None):
        cfg = hydra.compose(config_name="config")
    if override_config:
        cfg = OmegaConf.merge(cfg, override_config)
    return cfg
