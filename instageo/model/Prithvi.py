"""Vision Transformer Module."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import Block


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


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generates a 2D sinusoidal position embedding from a grid of positions.

    Args:
        embed_dim (int): Output dimension for each position.
        grid (np.ndarray): A 2D grid of positions.

    Returns:
        np.ndarray: The sinusoidal position embedding (H*W, D).
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_3d_sincos_pos_embed(
    embed_dim: int, grid_size: Tuple[int, int, int], cls_token: bool = False
) -> np.ndarray:
    """Generates a 3D sinusoidal position embedding from a given grid size.

    Args:
        embed_dim (int): Output dimension for each position.
        grid_size (Tuple[int, int, int]): 3D tuple of grid size (t, h, w).
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


class PatchEmbed(nn.Module):
    """Converts frames of 2D images to patch embedding.

    This is a 3D version of timm.models.vision_transformer.PatchEmbed.
    """

    def __init__(
        self,
        img_size_int: int = 224,
        patch_size_int: int = 16,
        num_frames: int = 3,
        tubelet_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: nn.Module | None = None,
        flatten: bool = True,
        bias: bool = True,
    ):
        """Initializes the PatchEmbed module.

        Args:
            img_size_int (int): Size of the image.
            patch_size_int (int): Size of each patch.
            num_frames (int): Number of frames.
            tubelet_size (int): Size of the tubelet.
            in_chans (int): Number of input channels.
            embed_dim (int): Dimension of the embedding.
            norm_layer (Optional[nn.Module]): Normalization layer.
            flatten (bool): Whether to flatten the output.
            bias (bool): Whether to use bias in the convolution layer.
        """
        super().__init__()
        img_size = to_2tuple(img_size_int)
        patch_size = to_2tuple(patch_size_int)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1]),
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        B, C, T, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x


class ViTEncoder(nn.Module):
    """Masked Autoencoder with a Vision Transformer (ViT) backbone.

    This class represents an autoencoder architecture suitable for vision tasks,
    particularly utilizing a Transformer-based approach for encoding and decoding.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 3,
        tubelet_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        norm_pix_loss: bool = False,
        **kwargs: int,
    ) -> None:
        """Initializes the ViTEncoder model.

        Args:
            img_size (int): Size of the input images.
            patch_size (int): Size of the patches to be extracted from the input images.
            num_frames (int): Number of frames to be considered for the input.
            tubelet_size (int): Size of the tubelet.
            in_chans (int): Number of input channels.
            embed_dim (int): Dimensionality of the token embeddings.
            depth (int): Number of layers in the Transformer encoder.
            num_heads (int): Number of attention heads in the Transformer encoder.
            decoder_embed_dim (int): Dimensionality of the token embeddings in the
                decoder.
            decoder_depth (int): Number of layers in the Transformer decoder.
            decoder_num_heads (int): Number of attention heads in the Transformer
                decoder.
            mlp_ratio (float): Ratio of feed-forward layer size to Transformer block
                size.
            norm_layer (Optional[nn.Module]): Normalization layer to be used in the
                Transformer.
            norm_pix_loss (bool): Whether to normalize pixel loss.

        The class includes methods for patchifying images, applying random masking,
        encoding with a Transformer-based architecture, and decoding with a separate
        Transformer-based architecture.
        """
        del kwargs  # unused kwargs
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialization.

        Initialize (and freeze) pos_embed by sin-cos embedding.
        """
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.) # noqa
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights.

        Args:
            m: nn.Module layer in model
        """
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes the input patches and applies random masking.

        Args:
            x (torch.Tensor): The input tensor of patches.
            mask_ratio (float): The proportion of the sequence to be masked.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The encoded tensor, the
                binary mask, and the indices to restore the original order.
        """
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
