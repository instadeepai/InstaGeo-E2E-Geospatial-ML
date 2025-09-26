from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from instageo.model.pritvhi import (
    LocationEncoder,
    PatchEmbed,
    TemporalEncoder,
    get_1d_sincos_embed_from_grid_torch,
    get_1d_sincos_pos_embed_from_grid,
    get_3d_sincos_pos_embed,
    interpolate_pos_encoding,
)


def test_get_1d_sincos_pos_embed_from_grid():
    embed_dim = 8
    pos = np.array([0, 1, 2, 3])
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

    assert emb.shape == (4, 8)  # (M, D)

    pos2 = np.array([0.5, 1.5])
    emb2 = get_1d_sincos_pos_embed_from_grid(embed_dim, pos2)
    assert emb2.shape == (2, 8)


def test_get_1d_sincos_embed_from_grid_torch():
    embed_dim = 8
    pos = torch.tensor([0, 1, 2, 3], dtype=torch.float32)
    emb = get_1d_sincos_embed_from_grid_torch(embed_dim, pos)

    assert emb.shape == (4, 8)

    pos2 = torch.tensor([0.5, 1.5], dtype=torch.float32)
    emb2 = get_1d_sincos_embed_from_grid_torch(embed_dim, pos2)
    assert emb2.shape == (2, 8)


def test_get_3d_sincos_pos_embed():
    embed_dim = 32
    grid_size = (2, 3, 4)

    emb = get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
    total_positions = grid_size[0] * grid_size[1] * grid_size[2]
    assert emb.shape == (total_positions, embed_dim)

    emb_with_cls = get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=True)
    assert emb_with_cls.shape == (total_positions + 1, embed_dim)
    assert np.allclose(emb_with_cls[0], np.zeros(embed_dim))


def test_interpolate_pos_encoding():
    embed_dim = 32
    grid_size = (2, 3, 4)
    patch_size = (1, 1, 1)
    shape = (4, 3, 4)

    pos_embed = torch.randn(1, 25, embed_dim)

    result2 = interpolate_pos_encoding(pos_embed, grid_size, patch_size, shape, embed_dim)
    assert result2.shape == (1, 49, embed_dim)


def test_patch_embed():
    patch_embed = PatchEmbed(
        input_size=(4, 64, 64),
        patch_size=(2, 16, 16),
        in_chans=3,
        embed_dim=128,
        flatten=True,
    )

    x = torch.randn(2, 3, 4, 64, 64)
    out = patch_embed(x)

    expected_num_patches = (4 // 2) * (64 // 16) * (64 // 16)
    assert out.shape == (
        2,
        expected_num_patches,
        128,
    ), f"Unexpected output shape: {out.shape}"

    patch_embed = PatchEmbed(
        input_size=(4, 64, 64),
        patch_size=(2, 16, 16),
        in_chans=3,
        embed_dim=128,
        flatten=False,
    )

    x = torch.randn(2, 3, 4, 64, 64)
    out = patch_embed(x)

    assert out.shape == (
        2,
        128,
        2,
        4,
        4,
    ), f"Unexpected shape when flatten=False: {out.shape}"

    norm_layer = lambda dim: nn.LayerNorm(dim)
    patch_embed = PatchEmbed(
        input_size=(4, 64, 64),
        patch_size=(2, 16, 16),
        in_chans=3,
        embed_dim=32,
        flatten=True,
        norm_layer=norm_layer,
    )

    x = torch.randn(1, 3, 4, 64, 64)
    out = patch_embed(x)

    assert isinstance(patch_embed.norm, nn.LayerNorm), "Norm layer not applied"
    assert out.shape == (1, 2 * 4 * 4, 32), "Incorrect output shape with LayerNorm"


def test_temporal_encoder():
    embed_dim = 32
    temporal_encoder = TemporalEncoder(embed_dim, trainable_scale=False)

    batch_size = 2
    num_frames = 3
    temporal_coords = torch.randn(batch_size, num_frames, 2)
    output = temporal_encoder(temporal_coords)

    assert output.shape == (batch_size, num_frames, embed_dim)

    tokens_per_frame = 4
    output2 = temporal_encoder(temporal_coords, tokens_per_frame)
    assert output2.shape == (batch_size, num_frames * tokens_per_frame, embed_dim)

    temporal_encoder_trainable = TemporalEncoder(embed_dim, trainable_scale=True)
    output3 = temporal_encoder_trainable(temporal_coords)
    assert output3.shape == (batch_size, num_frames, embed_dim)


def test_location_encoder():
    embed_dim = 32
    location_encoder = LocationEncoder(embed_dim, trainable_scale=False)

    batch_size = 2
    location_coords = torch.randn(batch_size, 2)
    output = location_encoder(location_coords)

    assert output.shape == (batch_size, 1, embed_dim)

    location_encoder_trainable = LocationEncoder(embed_dim, trainable_scale=True)
    output2 = location_encoder_trainable(location_coords)
    assert output2.shape == (batch_size, 1, embed_dim)
