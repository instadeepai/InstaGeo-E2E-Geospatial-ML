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
"""Vision Transformer Module."""
import logging
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from timm.layers import to_2tuple
from timm.models.vision_transformer import Block

logger = logging.getLogger(__name__)


def get_1d_sincos_embed_from_grid_torch(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """Generate a 1D sinusoidal positional embedding from the given positions.

    This function computes a 1D sinusoidal encoding for a list of positions
    using sine and cosine functions.
    The positional embeddings are commonly used in transformer models to provide
    information about the order
    of tokens in the input sequence.

    Args:
        embed_dim (int): The output dimension of the embeddings.
        pos (torch.Tensor): The tensor of positions to be encoded.

    Returns:
        torch.Tensor: The 1D sinusoidal position embeddings.
    """
    assert embed_dim % 2 == 0
    assert pos.dtype in [torch.float32, torch.float16, torch.bfloat16]

    omega = torch.arange(embed_dim // 2, dtype=pos.dtype).to(pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)

    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generates a 1D sinusoidal position embedding from a list of positions.

    Args:
        embed_dim (int): Output dimension for each position.
        pos (np.ndarray): A list of positions to be encoded, size (M,).

    Returns:
        np.ndarray: The sinusoidal position embedding (M, D).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: tuple[int, int, int],
    cls_token: bool = False,
) -> np.ndarray:
    """Generates a 3D sinusoidal position embedding from a given grid size.

    Args:
        embed_dim (int): Output dimension for each position.
        grid_size (list): grid size (t, h, w).
        cls_token (bool): Whether to include a class token.

    Returns:
        np.ndarray: The sinusoidal position embedding (L, D).
    """
    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def init_weights(module: nn.Module) -> None:
    """Initialize the weights of the given module.

    This function initializes the weights of a module based on its type:
    - For `nn.Linear`, it uses Xavier uniform initialization for weights and zeros for biases.
    - For `nn.LayerNorm`, it sets the weight to 1 and the bias to 0.

    Args:
        module (nn.Module): The module whose weights are to be initialized.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def interpolate_pos_encoding(
    pos_embed: torch.Tensor,
    grid_size: tuple[int, int, int],
    patch_size: tuple[int, int, int] | list[int],
    shape: tuple[int, int, int],
    embed_dim: int,
) -> torch.Tensor:
    """Interpolates the position encoding based on the new shape of the input tensor.

    The function adapts the position encoding if the input shape (after patching)
    differs from the original grid size.
    It handles resizing by generating a new position encoding for the new grid size.

    Args:
        pos_embed (torch.Tensor): The original position embedding tensor.
        grid_size (tuple[int, int, int] ): The grid size of the input tensor.
        patch_size (tuple[int, int, int] | list[int]): The size of patches that the
        image is divided into.
        shape (tuple[int, int, int]): The shape of the new input tensor (num_frames,
        height, width).
        embed_dim (int): The dimensionality of the position embeddings.

    Returns:
        torch.Tensor: The interpolated position encoding tensor.
    """
    t, h, w = shape
    t_patches = t // patch_size[0]
    h_patches = h // patch_size[1]
    w_patches = w // patch_size[2]

    if (t_patches, h_patches, w_patches) == grid_size:
        # No interpolation needed
        return pos_embed
    if t_patches != grid_size[0]:
        # Re-compute pos embedding to handle changed num_frames
        new_grid_size = (t_patches, grid_size[1], grid_size[2])
        new_pos_embed = get_3d_sincos_pos_embed(pos_embed.shape[-1], new_grid_size, cls_token=True)
        new_pos_embed = torch.from_numpy(new_pos_embed).float().unsqueeze(0)
    else:
        new_grid_size = grid_size
        new_pos_embed = pos_embed

    class_pos_embed, patch_pos_embed = new_pos_embed[:, :1], new_pos_embed[:, 1:]

    patch_pos_embed = patch_pos_embed.reshape(*new_grid_size, embed_dim).permute(0, 3, 1, 2)

    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed,
        size=(h_patches, w_patches),
        mode="bicubic",
        align_corners=True,
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, embed_dim)

    return torch.cat((class_pos_embed, patch_pos_embed), dim=1)


class PatchEmbed(nn.Module):
    """3D version of timm.models.vision_transformer.PatchEmbed."""

    def __init__(
        self,
        input_size: tuple[int, int, int] = (1, 224, 224),
        patch_size: tuple[int, int, int] = (1, 16, 16),
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: nn.Module | None = None,
        flatten: bool = True,
        bias: bool = True,
    ):
        """Initializes the PatchEmbed module.

        Args:
        input_size (Tuple[int, int, int]): The input size as a tuple
        of (T, H, W).
        patch_size (Tuple[int, int, int]): The size of each patch to
        extract from the input.
        in_chans (int): The number of input channels.
        embed_dim (int): The embedding dimension of the output patches.
        norm_layer (Optional[nn.Module]): A normalization layer to apply
        to the output embeddings..
        flatten (bool): Whether to flatten the patches into a 2D tensor
        of shape (B, L, embed_dim) or not.
        bias (bool): Whether to use a bias in the convolutional layer..
        """
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        grid_size_info = [s // p for s, p in zip(self.input_size, self.patch_size)]
        self.grid_size = (grid_size_info[0], grid_size_info[1], grid_size_info[2])
        assert self.grid_size >= (1, 1, 1), "Patch size is bigger than input size."
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute patch embeddings from the input tensor.

        Args:
            x (torch.Tensor): The input tensor with shape
            (batch_size, channels, frames, height, width).

        Returns:
            torch.Tensor: The patch embeddings, which are flattened and normalized.
        """
        B, C, T, H, W = x.shape

        if T / self.patch_size[0] % 1 or H / self.patch_size[1] % 1 or W / self.patch_size[2] % 1:
            warnings.warn(
                f"Input {x.shape[-3:]} is not divisible by patch size {self.patch_size}."
                f"The border will be ignored, add backbone_padding for pixel-wise tasks."
            )

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x


class TemporalEncoder(nn.Module):
    """Temporal encoding module for handling time-related information."""

    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        """Initializes the TemporalEncoder.

        Args:
        embed_dim (int): The embedding dimension for both year and day of the year information.
        trainable_scale (bool): Whether to make the scale parameter for the embeddings trainable.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.year_embed_dim = embed_dim // 2
        self.julian_day_embed_dim = embed_dim - self.year_embed_dim

        # If trainable, initialize scale with small number
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer("scale", torch.ones(1))

    def forward(
        self, temporal_coords: torch.Tensor, tokens_per_frame: int | None = None
    ) -> torch.Tensor:
        """Generate temporal embeddings for the given coordinates.

        Args:
            temporal_coords (torch.Tensor): A tensor with shape (B, T, 2)
            representing year and day-of-year information.
            tokens_per_frame (Optional[int]): If provided, the
            embeddings are repeated across the time dimension.

        Returns:
            torch.Tensor: The generated temporal embeddings.
        """
        shape = temporal_coords.shape[:2] + (-1,)  # B, T, -1

        year = get_1d_sincos_embed_from_grid_torch(
            self.year_embed_dim, temporal_coords[:, :, 0].flatten()
        ).reshape(shape)
        julian_day = get_1d_sincos_embed_from_grid_torch(
            self.julian_day_embed_dim, temporal_coords[:, :, 1].flatten()
        ).reshape(shape)

        embedding = self.scale * torch.cat([year, julian_day], dim=-1)

        if tokens_per_frame is not None:
            embedding = torch.repeat_interleave(embedding, tokens_per_frame, dim=1)

        return embedding  # B, T*tokens_per_frame, embed_dim


class LocationEncoder(nn.Module):
    """Location encoding module for handling geographical coordinates."""

    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        """Initializes the LocationEncoder.

        Args:
        embed_dim (int): The embedding dimension for the latitude and longitude information.
        trainable_scale (bool): Whether to make the scale parameter for the embeddings trainable.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.lat_embed_dim = embed_dim // 2
        self.lon_embed_dim = embed_dim - self.lat_embed_dim

        # If trainable, initialize scale with small number
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer("scale", torch.ones(1))

    def forward(self, location_coords: torch.Tensor) -> torch.Tensor:
        """Generate location embeddings for the given coordinates.

        Args:
            location_coords (torch.Tensor): A tensor with shape (B, 2)
            representing latitude and longitude information.

        Returns:
            torch.Tensor: The generated location embeddings.
        """
        shape = location_coords.shape[:1] + (1, -1)  # B, 1, -1

        lat = get_1d_sincos_embed_from_grid_torch(
            self.lat_embed_dim, location_coords[:, 0].flatten()
        ).reshape(shape)
        lon = get_1d_sincos_embed_from_grid_torch(
            self.lon_embed_dim, location_coords[:, 1].flatten()
        ).reshape(shape)

        embedding = self.scale * torch.cat([lat, lon], dim=-1)

        return embedding


class PrithviViT(nn.Module):
    """Prithvi ViT Encoder."""

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int, int] = (1, 14, 14),
        num_frames: int = 1,
        in_chans: int = 3,
        embed_dim: int = 1280,
        depth: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        coords_encoding: list[str] | None = None,
        coords_scale_learn: bool = False,
        drop_path: float = 0.0,
        **kwargs: Optional[dict],
    ):
        """Initializes the PrithviViT model.

        Args:
        img_size (Union[int, tuple[int, int]]): The size of the input image.
        patch_size (Union[int, tuple[int, int, int]]): The size of
        patches used in the embedding.
        num_frames (int): The number of frames in the input sequence (time dimension).
        in_chans (int): The number of input channels, typically 3 for RGB images.
        embed_dim (int): The embedding dimension of each patch after transformation.
        depth (int): The number of transformer blocks (layers).
        num_heads (int): The number of attention heads in each transformer block.
        mlp_ratio (float): The ratio of the hidden layer size to the embedding size in the
        MLP part of the transformer.
        norm_layer (nn.Module): The normalization layer applied in the transformer.
        coords_encoding (Optional[List[str]]): The list of coordinate types to include.
        coords_scale_learn (bool): Whether to make the scale parameter of coordinate
        embeddings trainable.
        drop_path (float): The drop path rate for regularization..
        **kwargs (Any): Additional arguments for customization.
        """
        super().__init__()

        self.in_chans = in_chans
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.img_size = to_2tuple(img_size)
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)

        # 3D patch embedding
        self.patch_embed = PatchEmbed(
            input_size=(num_frames,) + self.img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.out_channels = [embed_dim * self.patch_embed.grid_size[0]] * depth

        # Optional temporal and location embedding
        coords_encoding = coords_encoding or []
        self.temporal_encoding = "time" in coords_encoding
        self.location_encoding = "location" in coords_encoding
        if self.temporal_encoding:
            assert (
                patch_size[0] == 1
            ), f"With temporal encoding, patch_size[0] must be 1, received {patch_size[0]}"
            self.temporal_embed_enc = TemporalEncoder(embed_dim, coords_scale_learn)
        if self.location_encoding:
            self.location_embed_enc = LocationEncoder(embed_dim, coords_scale_learn)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer(
            "pos_embed", torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )

        # Transformer layers
        self.blocks = []
        for i in range(depth):
            self.blocks.append(
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                )
            )
        self.blocks = nn.ModuleList(self.blocks)

        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize weights for position embedding, patch embeddings, and transformer layers."""
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(init_weights)

    def interpolate_pos_encoding(self, sample_shape: tuple[int, int, int]) -> torch.Tensor:
        """Interpolate position encodings to match the input shape.

        Args:
            sample_shape (tuple[int, int, int]): The shape of the
            input tensor (num_frames, height, width).

        Returns:
            torch.Tensor: The interpolated position encodings.
        """
        pos_embed = interpolate_pos_encoding(
            pos_embed=self.pos_embed,
            grid_size=self.patch_embed.grid_size,
            patch_size=self.patch_embed.patch_size,
            shape=sample_shape,
            embed_dim=self.embed_dim,
        )
        return pos_embed

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through the Prithvi Vision Transformer.

        Args:
            x (torch.Tensor): The input tensor (batch_size, channels, num_frames, height, width).

        Returns:
            list[torch.Tensor]: The output of the transformer layers.
        """
        if len(x.shape) == 4 and self.patch_embed.input_size[0] == 1:
            # add time dim
            x = x.unsqueeze(2)
        sample_shape = x.shape[-3:]

        # embed patches
        x = self.patch_embed(x)

        pos_embed = self.interpolate_pos_encoding(sample_shape)
        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x
