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

import numpy as np
import pytorch_lightning as pl
import torch

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

    patch_coords = []
    current_batch = []

    for y in range(0, height - window_size[1] + 1, stride):
        for x in range(0, width - window_size[0] + 1, stride):
            patch_array = crop_array(
                hls_tile, x, y, x + window_size[0], y + window_size[1]
            )
            patch_coords.append((x, y))
            current_batch.append(patch_array)

            if len(current_batch) == batch_size:
                batch_array = torch.stack(current_batch, dim=0)
                batch_results = (
                    model.predict_step(batch_array.to(device)).detach().cpu().numpy()
                )

                for i, (px, py) in enumerate(patch_coords):
                    final_prediction[
                        py : py + window_size[1], px : px + window_size[0]
                    ] = batch_results[i]

                current_batch = []
                patch_coords = []

    if current_batch:
        batch_array = torch.stack(current_batch, dim=0)
        batch_results = (
            model.predict_step(batch_array.to(device)).detach().cpu().numpy()
        )

        for i, (px, py) in enumerate(patch_coords):
            final_prediction[
                py : py + window_size[1], px : px + window_size[0]
            ] = batch_results[i]

    return final_prediction
