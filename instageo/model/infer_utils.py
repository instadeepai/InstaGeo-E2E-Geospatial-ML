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

"""Utils for Running Inference."""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import numpy as np
import rasterio
import torch
from codecarbon import EmissionsTracker
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from tqdm import tqdm

from instageo.model.utils import get_carbon_info


def save_prediction(
    prediction: np.ndarray, file_name: str, output_folder: str, profile: Dict[str, Any]
) -> None:
    """Save a single prediction as a TIFF file.

    Args:
        prediction: The prediction array to be saved.
        file_name: The original input file name.
        output_folder: Directory where the prediction file will be saved.
        profile: Metadata profile for the TIFF file.

    Returns:
        None
    """
    output_basename = os.path.basename(file_name).replace("chip", "prediction")
    output_file_path = os.path.join(output_folder, output_basename)
    with rasterio.open(output_file_path, "w", **profile) as dst:
        dst.write(prediction, 1)


def chip_inference(
    dataloader: DataLoader,
    output_folder: str,
    model: LightningModule,
    device: str = "gpu",
    num_workers: int = 4,
) -> dict:
    """Chip Inference with optimizations.

    Performs inference on chips and saves corresponding predictions as TIFF files.

    Args:
        dataloader: Dataloader that yields input, label and input filenames.
        model: Trained model for inference.
        output_folder: Path to save predictions.
        device: Device used for inference.
        num_workers: Number of workers for concurrent file saving.

    Returns:
        Dict containing carbon tracking information.
    """
    device = "cuda" if device == "gpu" else device
    model.eval()
    model.to(device)

    tracker = EmissionsTracker(measure_power_secs=5, tracking_mode="machine", log_level="error")
    tracker.start()

    with torch.no_grad():
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for (data, _), file_names in tqdm(dataloader, desc="Running Inference"):
                data = data.to(device)
                prediction_batch = model(data)

                if prediction_batch.shape[1] == 1:  # Regression (single output channel)
                    prediction_batch = prediction_batch.cpu().numpy().squeeze(1)
                else:
                    prediction_batch = (
                        torch.argmax(prediction_batch, dim=1).cpu().numpy().astype(np.int8)
                    )

                profiles = []
                for file_name in file_names:
                    with rasterio.open(file_name) as src:
                        profile = src.profile
                        profile.update(
                            count=1,
                            dtype=rasterio.int8
                            if prediction_batch.dtype == np.int8
                            else rasterio.float32,
                        )
                        profiles.append(profile)

                futures = [
                    executor.submit(
                        save_prediction,
                        prediction,
                        file_name,
                        output_folder,
                        profile,
                    )
                    for prediction, file_name, profile in zip(
                        prediction_batch, file_names, profiles
                    )
                ]
                for future in futures:
                    future.result()

        tracker.stop()
        emissions_data = tracker._prepare_emissions_data()
        carbon_info = get_carbon_info(emissions_data)

    return carbon_info
