import torch

from instageo.model.utils import (
    get_common_prefix,
    get_proj_key,
    get_state_dict,
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
