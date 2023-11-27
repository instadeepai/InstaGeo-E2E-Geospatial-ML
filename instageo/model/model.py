import os
import time

import numpy as np
import requests  # type: ignore
import torch
import torch.nn as nn
import yaml  # type: ignore

from instageo.model.Prithvi import MaskedAutoencoderViT


def download_file(url: str, filename: str, retries: int = 3) -> None:
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
        print(f"File '{filename}' already exists. Skipping download.")
        return

    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"Download successful on attempt {attempt + 1}")
                break
            else:
                print(
                    f"Attempt {attempt + 1} failed with status code {response.status_code}"  # noqa
                )
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")

        if attempt < retries - 1:
            time.sleep(2)

    else:
        raise Exception("Failed to download the file after several attempts.")


class PrithviSeg(nn.Module):
    def __init__(self, temporal_step: int = 1) -> None:
        """Initialize the PrithviSeg model.

        This model is designed for image segmentation tasks on remote sensing data.
        It loads Prithvi configuration and weights and sets up a MaskedAutoencoderViT
        backbone along with a segmentation head.

        Args:
            temporal_step (int): Size of temporal dimension.
        """
        super().__init__()
        weights_path = "./prithvi/Prithvi_100M.pt"
        cfg_path = "./prithvi/Prithvi_100M_config.yaml"
        download_file(
            "https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/resolve/main/Prithvi_100M.pt?download=true",  # noqa
            weights_path,
        )
        download_file(
            "https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/raw/main/Prithvi_100M_config.yaml",  # noqa
            cfg_path,
        )
        checkpoint = torch.load(weights_path, map_location="cpu")
        with open(cfg_path) as f:
            model_config = yaml.safe_load(f)

        model_args = model_config["model_args"]

        model_args["num_frames"] = temporal_step
        self.model_args = model_args
        # instantiate model
        model = MaskedAutoencoderViT(**model_args)
        for param in model.parameters():
            param.requires_grad = False
        del checkpoint["pos_embed"]
        del checkpoint["decoder_pos_embed"]
        _ = model.load_state_dict(checkpoint, strict=False)

        self.prithvi_100M_backbone = model

        def upscaling_block(in_channels: int, out_channels: int) -> nn.Module:
            """Upscaling block.

            Args:
                in_channels (int): number of input channels.
                out_channels (int): number of output channels.

            Returns:
                An upscaling block configured to upscale spatially.
            """
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(
                    kernel_size=3,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    padding=1,
                ),
                nn.ReLU(),
            )

        embed_dims = [model_args["embed_dim"] // (2**i) for i in range(5)]
        self.segmentation_head = nn.Sequential(
            *[upscaling_block(embed_dims[i], embed_dims[i + 1]) for i in range(4)],
            nn.Conv2d(kernel_size=1, in_channels=embed_dims[-1], out_channels=2),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the model.

        Args:
            img (torch.Tensor): The input tensor representing the image.

        Returns:
            torch.Tensor: Output tensor after image segmentation.
        """
        features, _, _ = self.prithvi_100M_backbone.forward_encoder(img, mask_ratio=0)
        # drop cls token
        reshaped_features = features[:, 1:, :]
        feature_img_side_length = int(np.sqrt(reshaped_features.shape[1]))
        reshaped_features = reshaped_features.view(
            -1,
            feature_img_side_length,
            feature_img_side_length,
            self.model_args["embed_dim"],
        )
        # channels first
        reshaped_features = reshaped_features.permute(0, 3, 1, 2)
        out = self.segmentation_head(reshaped_features)
        return out
