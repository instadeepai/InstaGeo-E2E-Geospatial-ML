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

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from instageo.model.pritvhi import PrithviViT
from instageo.model.utils import PRETRAINED_BANDS, HLSBands, checkpoint_filter_fn_vit

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)


@dataclass
class PrithviConfig:
    """Configuration class for Prithvi model.

    Attributes:
        img_size: Size of input images
        num_frames: Number of temporal frames
        patch_size: Size of patches in [t, h, w] format
        in_chans: Number of input channels
        embed_dim: Embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        decoder_embed_dim: Embedding dimension for decoder
        decoder_depth: Number of decoder layers
        decoder_num_heads: Number of decoder attention heads
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        coords_encoding: List of coordinate encodings to use
        coords_scale_learn: Whether to learn coordinate scaling
        bands: List of bands to use
        mask_ratio: Masking ratio for MAE
        norm_pix_loss: Whether to normalize pixel loss
    """

    img_size: int = 224
    num_frames: int = 4
    patch_size: List[int] = field(default_factory=lambda: [1, 16, 16])
    in_chans: int = 6
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    mlp_ratio: int = 4
    coords_encoding: List[str] = field(default_factory=lambda: [])
    coords_scale_learn: bool = False
    bands: List[HLSBands] = field(default_factory=lambda: PRETRAINED_BANDS)
    mask_ratio: float = 0.75
    norm_pix_loss: bool = False

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary.

        Returns:
            dict: Dictionary representation of the configuration.
        """
        return {
            "img_size": self.img_size,
            "num_frames": self.num_frames,
            "patch_size": self.patch_size,
            "in_chans": self.in_chans,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "decoder_embed_dim": self.decoder_embed_dim,
            "decoder_depth": self.decoder_depth,
            "decoder_num_heads": self.decoder_num_heads,
            "mlp_ratio": self.mlp_ratio,
            "coords_encoding": self.coords_encoding,
            "coords_scale_learn": self.coords_scale_learn,
            "bands": self.bands,
            "mask_ratio": self.mask_ratio,
            "norm_pix_loss": self.norm_pix_loss,
        }


pretrained_weights = {
    "prithvi_eo_v1_100": {
        "hf_hub_id": "ibm-nasa-geospatial/Prithvi-EO-1.0-100M",
        "hf_hub_filename": "Prithvi_EO_V1_100M.pt",
    },
    "prithvi_eo_v2_300": {
        "hf_hub_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        "hf_hub_filename": "Prithvi_EO_V2_300M.pt",
    },
    "prithvi_eo_v2_300_tl": {
        "hf_hub_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
        "hf_hub_filename": "Prithvi_EO_V2_300M_TL.pt",
    },
    "prithvi_eo_v2_600": {
        "hf_hub_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
        "hf_hub_filename": "Prithvi_EO_V2_600M.pt",
    },
    "prithvi_eo_v2_600_tl": {
        "hf_hub_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL",
        "hf_hub_filename": "Prithvi_EO_V2_600M_TL.pt",
    },
}

prithvi_cfgs = {
    "prithvi_eo_tiny": PrithviConfig(
        num_frames=1,
        embed_dim=256,
        depth=4,
        num_heads=4,
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_num_heads=4,
    ),
    "prithvi_eo_v1_100": PrithviConfig(
        num_frames=3,
    ),
    "prithvi_eo_v2_100": PrithviConfig(),
    "prithvi_eo_v2_300": PrithviConfig(
        embed_dim=1024,
        depth=24,
        num_heads=16,
    ),
    "prithvi_eo_v2_300_tl": PrithviConfig(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        coords_encoding=["time", "location"],
        coords_scale_learn=True,
    ),
    "prithvi_eo_v2_600": PrithviConfig(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        patch_size=[1, 14, 14],
    ),
    "prithvi_eo_v2_600_tl": PrithviConfig(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        patch_size=[1, 14, 14],
        coords_encoding=["time", "location"],
        coords_scale_learn=True,
    ),
}
seg_head_kernel_sizes = {
    "prithvi_eo_tiny": [3, 3, 3, 3],
    "prithvi_eo_v1_100": [3, 3, 3, 3],
    "prithvi_eo_v2_100": [3, 3, 3, 3],
    "prithvi_eo_v2_300": [3, 3, 3, 3],
    "prithvi_eo_v2_300_tl": [3, 3, 3, 3],
    "prithvi_eo_v2_600": [5, 5, 5, 7],
    "prithvi_eo_v2_600_tl": [5, 5, 5, 7],
}


def create_prithvi(
    variant: str,
    model_bands: list[HLSBands],
    pretrained_bands: list[HLSBands],
    pretrained: bool = False,
    num_frames: int = 1,
    img_size: int = 224,
    depth: int = -1,
    **kwargs: Optional[dict],
) -> PrithviViT:
    """Builds the PrithviViT model.

    This function constructs a PrithviViT model using the specified
    variant and configurations.
    Args:
        variant (str): The variant of the PrithviViT model to build.
        pretrained (bool): Whether to load pretrained weights.
        model_bands (list[HLSBands]): A list of bands the model will use.
        If None, it uses the pretrained bands.
        pretrained_bands (list[HLSBands]): A list of pretrained bands to be used.
        num_frames (int, optional): The number of frames for the model.
        img_size: Image size used in the dataset.

    Returns:
        PrithviViT: A fully constructed PrithviViT model.
    """
    # Load default config
    model_args = prithvi_cfgs[variant].to_dict()
    if depth != -1:
        model_args["depth"] = depth

    model_args.update(kwargs)
    pretrained_bands = pretrained_bands or model_args.get("bands", PRETRAINED_BANDS)

    model_kwargs = {}
    model_kwargs["in_chans"] = len(model_bands)
    model_kwargs["num_frames"] = num_frames
    model_kwargs["img_size"] = img_size
    model_args.update(model_kwargs)
    model = PrithviViT(**model_args)

    if pretrained:
        assert variant in pretrained_weights, (
            f"No pre-trained model found for variant {variant} "
            f"(pretrained models: {pretrained_weights.keys()})"
        )

        try:
            # Download config.json to count model downloads
            _ = hf_hub_download(
                repo_id=pretrained_weights[variant]["hf_hub_id"],
                filename="config.json",
            )
            # Load model from Hugging Face
            pretrained_path = hf_hub_download(
                repo_id=pretrained_weights[variant]["hf_hub_id"],
                filename=pretrained_weights[variant]["hf_hub_filename"],
            )
            state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint_filter_fn_vit(state_dict, model, pretrained_bands, model_bands)

            # Only keep blocks from 0 to depth-1
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if not k.startswith("blocks.")
                or (k.startswith("blocks.") and int(k.split(".")[1]) < model_args["depth"])
            }
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            log.error(f"Failed to load the pre-trained weights for {variant}.")
            raise e

    model.model_bands = model_bands
    model.pretrained_bands = pretrained_bands

    return model


class Norm2D(nn.Module):
    """A normalization layer for 2D inputs.

    This class implements a 2D normalization layer using Layer Normalization.
    It is designed to normalize 2D inputs (e.g., images or feature maps in a
    convolutional neural network).
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
    """Prithvi Segmentation Model."""

    def __init__(
        self,
        temporal_step: int = 1,
        image_size: int = 224,
        num_classes: int = 2,
        load_pretrained_weights: bool = True,
        freeze_backbone: bool = True,
        model_bands: list[int] = list(range(6)),
        variant: str = "prithvi_eo_v1_100",
        embed_dims: list[int] | None = None,
        depth: int = -1,
        **kwargs: Any,
    ) -> None:
        """Initialize the PrithviSeg model.

        This model is designed for image segmentation tasks on remote sensing data.
        It loads Prithvi configuration and weights and sets up a ViTEncoder backbone
        along with a segmentation head.

        Args:
            temporal_step (int): Size of temporal dimension.
            image_size (int): Size of input image.
            num_classes (int): Number of target classes.
            load_pretrained_weights (bool): Flag to whether use the pretrained weights or not.
            freeze_backbone (bool): Flag to freeze ViT transformer backbone weights.
            model_bands (list): Bands used in the dataset.
            variant (str): The model architecture to use.
            embed_dims (list[int]): List of embedding dimensions for the segmentation head.
            depth (int): The depth of the model.
            load_pretrained_weights (bool): Whether to load pretrained weights.
            **kwargs: Additional keyword arguments.

        """
        super().__init__()

        model_bands_ = PRETRAINED_BANDS * (len(model_bands) // len(PRETRAINED_BANDS))
        model = create_prithvi(
            variant=variant,
            pretrained=load_pretrained_weights,
            model_bands=model_bands_,
            pretrained_bands=PRETRAINED_BANDS,
            num_frames=temporal_step,
            img_size=image_size,
            depth=depth,
            **kwargs,
        )
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        self.prithvi_encoder = model
        model_args = prithvi_cfgs[variant].to_dict()
        self.model_args = model_args
        model_args["num_frames"] = temporal_step

        def upscaling_block(in_channels: int, out_channels: int, kernel_size: int) -> nn.Module:
            """Upscaling block.

            Args:
                in_channels (int): number of input channels.
                out_channels (int): number of output channels.
                kernel_size (int): kernel size

            Returns:
                An upscaling block configured to upscale spatially.
            """
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.Dropout(0.1),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        if embed_dims is None:
            embed_dims = [
                (model_args["embed_dim"] * model_args["num_frames"]) // (2**i) for i in range(5)
            ]
        kernel_sizes = seg_head_kernel_sizes[variant]

        self.segmentation_head = nn.Sequential(
            *[upscaling_block(embed_dims[i], embed_dims[i + 1], kernel_sizes[i]) for i in range(4)],
            nn.Dropout(0.1),
            nn.Conv2d(kernel_size=1, in_channels=embed_dims[-1], out_channels=num_classes),
        )

    def forward(
        self, img: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Define the forward pass of the model.

        Args:
            img (torch.Tensor): The input tensor representing the image.

        Returns:
            torch.Tensor: Output tensor after image segmentation.
        """
        features = self.prithvi_encoder(img)

        # drop cls token
        reshaped_features = features[:, 1:, :]

        feature_img_side_length = int(
            np.sqrt(reshaped_features.shape[1] // self.model_args["num_frames"])
        )
        reshaped_features = reshaped_features.permute(0, 2, 1).reshape(
            features.shape[0], -1, feature_img_side_length, feature_img_side_length
        )
        out = self.segmentation_head(reshaped_features)

        if return_features:
            return out, reshaped_features
        else:
            return out
