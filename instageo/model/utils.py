# Copyright (c) IBM Corp. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# transformers: https://github.com/huggingface/transformers
# --------------------------------------------------------

"""Utility functions."""

import logging
from enum import Enum
from typing import Any, Dict, KeysView, Optional, Tuple

import torch
import torch.nn as nn
from carbontracker import parser
from carbontracker.tracker import CarbonTracker
from neptune import Run
from ptflops import get_model_complexity_info
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)


class HLSBands(Enum):
    """Enum class representing the various bands in the Hyperion Landsat Sentinel (HLS) dataset."""

    BLUE = "BLUE"
    GREEN = "GREEN"
    RED = "RED"
    NIR_NARROW = "NIR_NARROW"
    SWIR_1 = "SWIR_1"
    SWIR_2 = "SWIR_2"


PRETRAINED_BANDS = [
    HLSBands.BLUE,
    HLSBands.GREEN,
    HLSBands.RED,
    HLSBands.NIR_NARROW,
    HLSBands.SWIR_1,
    HLSBands.SWIR_2,
]


def patch_embed_weights_are_compatible(
    model_patch_embed: torch.Tensor, checkpoint_patch_embed: torch.Tensor
) -> bool:
    """Checks if the patch embedding weights from the model and checkpoint are compatible.

    Compares the dimensions of the patch embedding tensors, excluding the channel dimension,
    to determine if they are compatible for weight transfer.

    Args:
        model_patch_embed (torch.Tensor): The patch embedding tensor from the model.
        checkpoint_patch_embed (torch.Tensor): The patch embedding tensor from the checkpoint.

    Returns:
        bool: True if the patch embedding dimensions are compatible, False otherwise.
    """
    # check all dimensions are the same except for channel dimension
    if len(model_patch_embed.shape) != len(checkpoint_patch_embed.shape):
        return False

    model_shape = [
        model_patch_embed.shape[i]
        for i in range(len(model_patch_embed.shape))
        if i != 1
    ]
    checkpoint_shape = [
        checkpoint_patch_embed.shape[i]
        for i in range(len(checkpoint_patch_embed.shape))
        if i != 1
    ]
    return model_shape == checkpoint_shape


def get_state_dict(state_dict: dict) -> dict:
    """Extracts the state dict from the provided dictionary.

    Searches for the key that ends with "state_dict" and returns its
    corresponding value.
    If no such key exists, the original state_dict is returned.

    Args:
        state_dict (dict): The state dictionary containing model weights.

    Returns:
        dict: The state dictionary corresponding to the "state_dict" key,
        or the original state_dict.
    """

    def search_state_dict(keys: KeysView[str]) -> str:
        key = ""
        for k in keys:
            if k.endswith("state_dict"):
                key = k
                break
        return key

    state_dict_key = search_state_dict(state_dict.keys())

    if state_dict_key:
        return state_dict[state_dict_key]
    else:
        return state_dict


def get_common_prefix(keys: KeysView[str]) -> str:
    """Finds the common prefix of a list of keys.

    The function compares all keys and returns the common prefix, which is the part
    of the key string shared by all elements in the list.

    Args:
        keys (list): A list of keys (strings) to compare.

    Returns:
        str: The common prefix shared by the input keys.
    """
    keys_big_list = []

    keys_ = list(keys)
    keys_.pop(-1)

    for k in keys_:
        keys_big_list.append(set(k.split(".")))
    prefix_list = set.intersection(*keys_big_list)

    if len(prefix_list) > 1:
        prefix = ".".join(prefix_list)
    else:
        prefix = prefix_list.pop()

    return prefix + "."


def get_proj_key(state_dict: dict, return_prefix: bool = False) -> Tuple:
    """Finds the projection key for patch embedding weights in the state_dict.

    Searches for keys that match the patch embedding projection weight names and optionally
    returns the prefix used for the projection key.

    Args:
        state_dict (dict): The state dictionary.
        return_prefix (bool, optional): If True, the prefix of the key will also be returned.

    Returns:
        tuple: The projection key and the optional prefix, if requested.
    """
    proj_key = None

    for key in state_dict.keys():
        if key.endswith("patch_embed.proj.weight") or key.endswith(
            "patch_embed.projection.weight"
        ):
            proj_key = key
            break

    if return_prefix and proj_key:
        for sufix in ["patch_embed.proj.weight", "patch_embed.projection.weight"]:
            if proj_key.endswith(sufix):
                prefix = proj_key.replace(sufix, "")
                break

    else:
        prefix = None

    return proj_key, prefix


def remove_prefixes(state_dict: dict, prefix: str) -> dict:
    """Removes the specified prefix from all keys in the state_dict.

    This function modifies the state_dict by removing the given prefix from all the keys.

    Args:
        state_dict (dict): The state dictionary.
        prefix (str): The prefix to remove from the keys.

    Returns:
        dict: A new state dictionary with the prefix removed from the keys.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace(prefix, "")] = v
    return new_state_dict


def select_patch_embed_weights(
    state_dict: dict,
    model: torch.nn.Module,
    pretrained_bands: list,
    model_bands: list,
    proj_key: str | None = None,
) -> dict:
    """Filter out the patch embedding weights according to the bands being used.

    If a band exists in the pretrained_bands, but not in model_bands, drop it.
    If a band exists in model_bands, but not pretrained_bands, randomly
    initialize those weights.

    Args:
        state_dict (dict): State Dict
        model (nn.Module): Model to load the weights onto.
        pretrained_bands (list[HLSBands | int]): List of bands the
        model was pretrained on, in the correct order.
        model_bands (list[HLSBands | int]): List of bands the model is going
        to be finetuned on, in the correct order
        proj_key (str, optional): Key to patch embedding
        projection weight in state_dict.

    Returns:
        dict: New state dict
    """
    if (
        (isinstance(pretrained_bands, type(model_bands)))
        | isinstance(pretrained_bands, int)
        | isinstance(model_bands, int)
    ):
        state_dict = get_state_dict(state_dict)
        prefix = None  # we expect no prefix will be necessary in principle

        if proj_key is None:
            # Search for patch embedding weight in state dict
            proj_key, prefix = get_proj_key(state_dict, return_prefix=True)
        if proj_key is None or proj_key not in state_dict:
            raise Exception("Could not find key for patch embed weight in state_dict.")

        patch_embed_weight = state_dict[proj_key]

        # It seems `proj_key` can have different names for
        # the checkpoint and the model instance
        proj_key_, _ = get_proj_key(model.state_dict())

        if proj_key_:
            temp_weight = model.state_dict()[proj_key_].clone()
        else:
            temp_weight = model.state_dict()[proj_key].clone()

        # only do this if the patch size and tubelet size match. If not, start with random weights
        if patch_embed_weights_are_compatible(temp_weight, patch_embed_weight):
            torch.nn.init.xavier_uniform_(temp_weight.view([temp_weight.shape[0], -1]))
            for index, band in enumerate(model_bands):
                if band in pretrained_bands:
                    logging.info(
                        f"Loaded weights for {band} in position {index} of patch embed"
                    )
                    temp_weight[:, index] = patch_embed_weight[
                        :, pretrained_bands.index(band)
                    ]
        else:
            log.warning(
                f"Incompatible shapes between patch embedding of model {temp_weight.shape} and\
                of checkpoint {patch_embed_weight.shape}",
            )

        state_dict[proj_key] = temp_weight

        if prefix:
            state_dict = remove_prefixes(state_dict, prefix)

    return state_dict


def checkpoint_filter_fn_vit(
    state_dict: dict,
    model: torch.nn.Module,
    pretrained_bands: list,
    model_bands: list[HLSBands],
) -> dict:
    """Filters the state dictionary for Vision Transformer (ViT) checkpoints.

    Removes unnecessary keys from the state dictionary, such as decoder weights and
    embeddings that don't match the model's configuration. It also selects the appropriate
    patch embedding weights based on the bands being used.

    Args:
        state_dict (dict): The state dictionary to filter.
        model (nn.Module): The model to load the filtered weights onto.
        pretrained_bands (list): List of bands the model was pretrained on.
        model_bands (list): List of bands the model will be fine-tuned on.

    Returns:
        dict: The filtered state dictionary.
    """
    clean_dict = {}
    for k, v in state_dict.items():
        if "_timm_module." in k:  # Backwards compatibility for old model checkpoints
            k = k.replace("_timm_module.", "")

        if "pos_embed" in k:
            v = model.pos_embed  # pos_embed depends on num_frames and is fixed.
        if "decoder" in k or "_dec" in k or k == "mask_token":
            continue  # Drop decoder weights
        if not model.temporal_encoding and "temporal_embed" in k:
            continue
        if not model.location_encoding and "location_embed" in k:
            continue

        if k.startswith("encoder."):
            clean_dict[
                k.replace("encoder.", "")
            ] = v  # Convert Prithvi MAE to Prithvi ViT
        else:
            clean_dict[k] = v

    state_dict = clean_dict

    state_dict = select_patch_embed_weights(
        state_dict, model, pretrained_bands, model_bands
    )

    return state_dict


class CarbonTrackerCallback(Callback):
    """Callback for tracking carbon emissions.

    PyTorch Lightning callback for tracking carbon emissions using CarbonTracker during
    training and test phases. Logs results to Neptune if a run is provided.

    Parameters:
        total_epochs (int): Total number of training epochs.
        neptune_run (optional): Neptune run object for logging results.
        log_dir (str): Directory to store CarbonTracker logs.
    """

    def __init__(
        self,
        total_epochs: int,
        neptune_run: Optional[Run] = None,
        log_dir: str = "carbon_logs",
    ):
        """Initialize the CarbonTrackerCallback."""
        self.total_epochs = total_epochs
        self.neptune_run = neptune_run
        self.log_dir = log_dir
        self.tracker = CarbonTracker(
            epochs=total_epochs,
            monitor_epochs=1,
            components="gpu",
            log_dir=log_dir,
            ignore_errors=True,
        )
        self.eval_tracker = CarbonTracker(
            epochs=1,
            monitor_epochs=1,
            components="gpu",
            log_dir=log_dir,
            ignore_errors=True,
        )

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Start carbon tracking for a training epoch."""
        self.tracker.epoch_start()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """End carbon tracking for a training epoch."""
        self.tracker.epoch_end()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Stop training tracker.

        Stop training tracker and log carbon consumption metrics to Neptune if provided.
        """
        self.tracker.stop()

        logs = parser.parse_all_logs(log_dir=self.log_dir)
        first_log = logs[0]
        if self.neptune_run is not None:
            log_carbon_info(self.neptune_run, first_log)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start carbon tracking for a test epoch."""
        self.eval_tracker.epoch_start()

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """End carbon tracking for a test epoch."""
        self.eval_tracker.epoch_end()

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Stop test tracker.

        Stop test tracker and log carbon metrics to Neptune under `carbon/test/*`.
        Assumes the last parsed log is for test.
        """
        self.eval_tracker.stop()

        logs = parser.parse_all_logs(log_dir=self.log_dir)
        test_log = logs[-1]
        if self.neptune_run is not None:
            log_carbon_info(self.neptune_run, test_log)


def log_model_complexity(
    model: nn.Module, cfg: Any, neptune_logger: Optional[Run], device: int = 0
) -> None:
    """Logs model computational complexity (MACs, GFLOPs, Params) to Neptune.

    Args:
        model (torch.nn.Module): The PyTorch model.
        cfg (object): Configuration object with dataloader attributes.
        neptune_logger: Neptune logger object.
        device (int): GPU device index. Default is 0.
    """
    input_shape = (
        len(cfg.dataloader.bands),
        cfg.dataloader.temporal_dim,
        cfg.dataloader.img_size,
        cfg.dataloader.img_size,
    )

    with torch.cuda.device(device):
        macs, params = get_model_complexity_info(
            model,
            input_shape,
            as_strings=True,
            print_per_layer_stat=True,
            verbose=True,
        )

    if neptune_logger is not None:
        neptune_logger.experiment["model/Computational complexity"] = macs
        neptune_logger.experiment["model/GFLOPs"] = float(macs.split()[0]) * 2
        neptune_logger.experiment["model/params"] = params


def log_carbon_info(neptune_run: Optional[Run], first_log: Dict[str, Any]) -> None:
    """Logs carbon tracking information to Neptune.

    Args:
        neptune_run: Neptune run object (e.g., self.neptune_run).
        first_log (dict): Dictionary containing carbon log data.
    """
    if neptune_run is not None:
        run = neptune_run.experiment

        run["carbon/output_filename"] = first_log["output_filename"]
        run["carbon/standard_filename"] = first_log["standard_filename"]
        run["carbon/early_stop"] = first_log["early_stop"]
        run["carbon/actual"] = first_log["actual"]
        run["carbon/pred"] = first_log["pred"]
        run["carbon/components"] = first_log["components"]["gpu"]["devices"]
