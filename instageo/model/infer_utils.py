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
from typing import Any, Dict, Generator, List

import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from instageo.model.dataloader import crop_array


def sliding_window_inference(
    hls_tile: torch.Tensor,
    model: pl.LightningModule,
    window_size: tuple[int, int] = (224, 224),
    stride: int = 224,
    batch_size: int = 32,
    device: str = "gpu",
) -> np.ndarray:
    """Sliding Window Inference.

    Performs sliding window inference on large inputs using a given model with batching,
    and reassemble the output to match the original image size.

    Args:
        image_path: Path to the large image.
        model: Trained model for inference.
        window_size: Size of the window (default is 224x224).
        stride: Step size for sliding the window (default is 224).
        batch_size: Number of patches to process in one batch.
        device: Device used for training.

    Returns:
        Final prediction image of the same size as the original image.
    """
    device = "cuda" if device == "gpu" else device
    _, _, width, height = hls_tile.shape

    final_prediction = np.zeros((height, width), dtype=np.float32)
    patch_coords = [
        (x, y)
        for y in range(0, height - window_size[1] + 1, stride)
        for x in range(0, width - window_size[0] + 1, stride)
    ]

    def get_batches(
        patches: List[torch.Tensor], batch_size: int
    ) -> Generator[List[torch.Tensor], None, None]:
        """Splits a list of patches into batches of the specified size.

        Args:
            patches: List of tensors representing patches.
            batch_size: Number of patches in each batch.

        Yields:
            Batches of patches as lists of tensors.
        """
        for i in range(0, len(patches), batch_size):
            yield patches[i : i + batch_size]

    with torch.no_grad():
        for batch_coords in get_batches(patch_coords, batch_size):
            batch_patches = [
                crop_array(hls_tile, x, y, x + window_size[0], y + window_size[1])
                for x, y in batch_coords
            ]
            batch_tensor = torch.stack(batch_patches, dim=0).to(device)
            batch_results = model.predict_step(batch_tensor).detach().cpu().numpy()

            for (x, y), result in zip(batch_coords, batch_results):
                final_prediction[
                    y : y + window_size[1], x : x + window_size[0]
                ] = result

    return final_prediction


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
    model: pl.LightningModule,
    device: str = "gpu",
    num_workers: int = 4,
) -> None:
    """Chip Inference with optimizations.

    Performs inference on chips and saves corresponding predictions as TIFF files.

    Args:
        dataloader: Dataloader that yields input, label and input filenames.
        model: Trained model for inference.
        output_folder: Path to save predictions.
        device: Device used for inference.
        num_workers: Number of workers for concurrent file saving.

    Returns:
        None.
    """
    device = "cuda" if device == "gpu" else device
    model.eval()
    model.to(device)

    with torch.no_grad():
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for (data, _), file_names in tqdm(dataloader, desc="Running Inference"):
                data = data.to(device)
                prediction_batch = model(data)
                prediction_cls = (
                    torch.nn.functional.softmax(prediction_batch, dim=1)
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )

                profiles = []
                for file_name in file_names:
                    with rasterio.open(file_name) as src:
                        profile = src.profile
                        profile.update(count=1, dtype=rasterio.float32)
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
                        prediction_cls, file_names, profiles
                    )
                ]
                for future in futures:
                    future.result()
