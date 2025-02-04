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

"""Model Module."""

import os
import time
from pathlib import Path

import numpy as np
import requests  # type: ignore
import torch
import torch.nn as nn
import yaml  # type: ignore
from absl import logging

from instageo.model.Prithvi import ViTEncoder


def download_file(url: str, filename: str | Path, retries: int = 3) -> None:
    """Downloads a file from the given URL and saves it to a local file.

    Args:
        url (str): The URL from which to download the file.
        filename (str): The local path where the file will be saved.
        retries (int, optional): The number of times to retry the download
                                 in case of failure. Defaults to 3.

    Raises:
        Exception: If the download fails after the specified number of retries.

    Returns:
        None
    """
    if os.path.exists(filename):
        logging.info(f"File '{filename}' already exists. Skipping download.")
        return

    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                logging.info(f"Download successful on attempt {attempt + 1}")
                break
            else:
                logging.warning(
                    f"Attempt {attempt + 1} failed with status code {response.status_code}"  # noqa
                )
        except requests.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed with error: {e}")

        if attempt < retries - 1:
            time.sleep(2)

    else:
        raise Exception("Failed to download the file after several attempts.")


class Norm2D(nn.Module):
    """A normalization layer for 2D inputs.

    This class implements a 2D normalization layer using Layer Normalization.
    It is designed to normalize 2D inputs (e.g., images or feature maps in a
    convolutional neural network).

    Attributes:
        ln (nn.LayerNorm): The layer normalization component.

    Args:
        embed_dim (int): The number of features of the input tensor (i.e., the number of
            channels in the case of images).

    Methods:
        forward: Applies normalization to the input tensor.
    """

    def __init__(self, embed_dim: int):
        """Initializes the Norm2D module.

        Args:
            embed_dim (int): The number of features of the input tensor.
        """
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the normalization process to the input tensor.

        Args:
            x (torch.Tensor): A 4D input tensor with shape
                (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The normalized tensor, having the same shape as the input.
        """
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class PrithviSeg(nn.Module):
    def __init__(self, temporal_step=3, image_size=224, num_classes=2, freeze_backbone=True):
        super().__init__()

        # Load Pretrained Weights
        weights_dir = Path.home() / ".instageo" / "prithvi"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "Prithvi_EO_V1_100M.pt"
        cfg_path = weights_dir / "config.yaml"

        download_file("https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M/resolve/main/Prithvi_EO_V1_100M.pt?download=true", weights_path)
        download_file("https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/raw/main/config.yaml", cfg_path)

        checkpoint = torch.load(weights_path, map_location="cpu")
        with open(cfg_path) as f:
            model_config = yaml.safe_load(f)

        model_args = model_config["model_args"]
        model_args["num_frames"] = temporal_step
        model_args["img_size"] = image_size
        self.model_args = model_args

        # Load ViT Encoder
        model = ViTEncoder(**model_args)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        filtered_checkpoint_state_dict = {
            key[len("encoder.") :]: value
            for key, value in checkpoint.items()
            if key.startswith("encoder.")
        }
        filtered_checkpoint_state_dict["pos_embed"] = torch.zeros(
            1, (temporal_step * (image_size // 16) ** 2 + 1), 768
        )
        _ = model.load_state_dict(filtered_checkpoint_state_dict)

        self.prithvi_100M_backbone = model

        # **Fix: Reduce Temporal Dimension before Decoder**
        self.temporal_merge = nn.Conv2d(temporal_step * 768, 768, kernel_size=1)

        # Improved U-Net Decoder
        self.decoder = nn.Sequential(
            self.upsample_block(768, 384),  # 32x32 → 64x64
            self.upsample_block(384, 192),  # 64x64 → 128x128
            self.upsample_block(192, 96),   # 128x128 → 256x256
            nn.Conv2d(96, num_classes, kernel_size=1)  # Final segmentation output
        )

        # **Fix: Final Upsampling to Match Ground Truth (256x256)**
        self.final_upsample = nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True)

    def upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, img):
        features = self.prithvi_100M_backbone(img)
        reshaped_features = features[:, 1:, :]
        feature_img_side_length = int((reshaped_features.shape[1] // self.model_args["num_frames"]) ** 0.5)

        # **Fix: Merge Temporal Dimension (2304 → 768 channels)**
        reshaped_features = reshaped_features.permute(0, 2, 1).reshape(
            features.shape[0], -1, feature_img_side_length, feature_img_side_length
        )
        reshaped_features = self.temporal_merge(reshaped_features)  # Merge temporal steps into 1 feature map

        out = self.decoder(reshaped_features)

        # **Fix: Ensure Output Matches Target Mask Size**
        out = self.final_upsample(out)  # Upsample to (batch_size, num_classes, 256, 256)
        return out