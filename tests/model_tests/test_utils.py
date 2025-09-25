import torch

from instageo.model.utils import (
    HLSBands,
    checkpoint_filter_fn_vit,
    get_common_prefix,
    get_proj_key,
    get_state_dict,
    patch_embed_weights_are_compatible,
    remove_prefixes,
)


def test_get_state_dict():
    test_dict = {
        "model.state_dict": {"layer1.weight": torch.randn(2, 2)},
        "other_key": "value",
    }
    result = get_state_dict(test_dict)
    assert result == test_dict["model.state_dict"]

    test_dict2 = {"layer1.weight": torch.randn(2, 2), "layer1.bias": torch.randn(2)}
    result2 = get_state_dict(test_dict2)
    assert result2 == test_dict2

    test_dict3 = {
        "first.state_dict": {"layer1.weight": torch.randn(2, 2)},
        "second.state_dict": {"layer2.weight": torch.randn(2, 2)},
    }
    result3 = get_state_dict(test_dict3)
    assert result3 == test_dict3["first.state_dict"]


def test_get_common_prefix():
    keys = {
        "model.layer1.weight",
        "model.layer2.bias",
        "model.layer3.activation_func",
    }
    result = get_common_prefix(keys)
    assert result == "model."


def test_get_proj_key():
    test_dict = {
        "model.patch_embed.proj.weight": torch.randn(2, 2),
        "model.patch_embed.proj.bias": torch.randn(2),
    }
    proj_key, prefix = get_proj_key(test_dict, return_prefix=True)
    assert proj_key == "model.patch_embed.proj.weight"
    assert prefix == "model."

    test_dict2 = {
        "model.patch_embed.projection.weight": torch.randn(2, 2),
        "model.patch_embed.projection.bias": torch.randn(2),
    }
    proj_key2, prefix2 = get_proj_key(test_dict2, return_prefix=True)
    assert proj_key2 == "model.patch_embed.projection.weight"
    assert prefix2 == "model."

    proj_key3, prefix3 = get_proj_key(test_dict)
    assert proj_key3 == "model.patch_embed.proj.weight"
    assert prefix3 is None

    test_dict4 = {"model.layer1.weight": torch.randn(2, 2)}
    proj_key4, prefix4 = get_proj_key(test_dict4)
    assert proj_key4 is None
    assert prefix4 is None


def test_remove_prefixes():
    test_dict = {
        "model.layer1.weight": torch.randn(2, 2),
        "model.layer1.bias": torch.randn(2),
    }
    result = remove_prefixes(test_dict, "model.")
    assert "layer1.weight" in result
    assert "layer1.bias" in result
    assert "model.layer1.weight" not in result
    assert "model.layer1.bias" not in result


def test_patch_embed_weights_are_compatible():
    # Test case 1: Compatible tensors with different channel dimensions
    model_embed = torch.randn(64, 6, 16, 16)  # 6 channels
    checkpoint_embed = torch.randn(64, 3, 16, 16)  # 3 channels
    assert patch_embed_weights_are_compatible(model_embed, checkpoint_embed) == True

    # Test case 2: Incompatible tensors with different spatial dimensions
    model_embed = torch.randn(64, 6, 16, 16)
    checkpoint_embed = torch.randn(64, 3, 32, 32)
    assert patch_embed_weights_are_compatible(model_embed, checkpoint_embed) == False

    # Test case 3: Incompatible tensors with different batch dimensions
    model_embed = torch.randn(32, 6, 16, 16)
    checkpoint_embed = torch.randn(64, 3, 16, 16)
    assert patch_embed_weights_are_compatible(model_embed, checkpoint_embed) == False

    # Test case 4: Different number of dimensions
    model_embed = torch.randn(64, 6, 16, 16)
    checkpoint_embed = torch.randn(64, 3, 16)
    assert patch_embed_weights_are_compatible(model_embed, checkpoint_embed) == False


class MockViTModel(torch.nn.Module):
    def __init__(self, temporal_encoding=True, location_encoding=True):
        super().__init__()
        self.temporal_encoding = temporal_encoding
        self.location_encoding = location_encoding
        self.pos_embed = torch.randn(1, 196, 768)  # Example pos_embed shape
        self.patch_embed = torch.nn.Module()
        self.patch_embed.proj = torch.nn.Conv2d(6, 768, kernel_size=16, stride=16)


def test_checkpoint_filter_fn_vit():
    # Create a mock model and state dict
    model = MockViTModel(temporal_encoding=True, location_encoding=True)

    # Create a sample state dict with various types of weights
    state_dict = {
        "patch_embed.proj.weight": torch.randn(768, 3, 16, 16),
        "pos_embed": torch.randn(1, 196, 768),
        "temporal_embed": torch.randn(1, 12, 768),
        "location_embed": torch.randn(1, 2, 768),
        "decoder.blocks.0.weight": torch.randn(768, 768),
        "encoder.blocks.0.weight": torch.randn(768, 768),
        "mask_token": torch.randn(1, 1, 768),
    }

    # Define band configurations
    pretrained_bands = [HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE]
    model_bands = [
        HLSBands.RED,
        HLSBands.GREEN,
        HLSBands.BLUE,
        HLSBands.NIR_NARROW,
        HLSBands.SWIR_1,
        HLSBands.SWIR_2,
    ]

    # Filter the state dict
    filtered_dict = checkpoint_filter_fn_vit(state_dict, model, pretrained_bands, model_bands)

    # Test that decoder weights are removed
    assert "decoder.blocks.0.weight" not in filtered_dict
    assert "mask_token" not in filtered_dict

    # Test that encoder weights are properly renamed
    assert "blocks.0.weight" in filtered_dict

    # Test with different encoding settings
    model_no_temporal = MockViTModel(temporal_encoding=False, location_encoding=True)
    filtered_dict_no_temporal = checkpoint_filter_fn_vit(
        state_dict, model_no_temporal, pretrained_bands, model_bands
    )
    assert "temporal_embed" not in filtered_dict_no_temporal

    model_no_location = MockViTModel(temporal_encoding=True, location_encoding=False)
    filtered_dict_no_location = checkpoint_filter_fn_vit(
        state_dict, model_no_location, pretrained_bands, model_bands
    )
    assert "location_embed" not in filtered_dict_no_location

    # Test backwards compatibility
    state_dict_with_timm = {
        "_timm_module.patch_embed.proj.weight": torch.randn(768, 3, 16, 16),
        "_timm_module.blocks.0.weight": torch.randn(768, 768),
    }
    filtered_dict_timm = checkpoint_filter_fn_vit(
        state_dict_with_timm, model, pretrained_bands, model_bands
    )
    assert "patch_embed.proj.weight" in filtered_dict_timm
    assert "blocks.0.weight" in filtered_dict_timm
