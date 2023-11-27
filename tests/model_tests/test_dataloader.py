import numpy as np
import torch
from PIL import Image

from instageo.model.dataloader import (
    get_arr_flood,
    normalize_and_convert_to_tensor,
    process_and_augment,
    process_data,
    random_crop_and_flip,
)


def test_random_crop_and_flip():
    # Create dummy images and label
    ims = [Image.new("L", (256, 256)) for _ in range(3)]
    label = Image.new("L", (256, 256))

    # Apply function
    transformed_ims, transformed_label = random_crop_and_flip(ims, label, im_size=224)

    # Check output types and dimensions
    assert isinstance(transformed_ims, list)
    assert isinstance(transformed_label, Image.Image)
    assert all(isinstance(im, Image.Image) for im in transformed_ims)
    assert all(im.size == (224, 224) for im in transformed_ims)
    assert transformed_label.size == (224, 224)


def test_normalize_and_convert_to_tensor():
    # Create dummy images and label
    ims = [Image.new("L", (224, 224)) for _ in range(3)]
    label = Image.new("L", (224, 224))

    tensor_ims, tensor_label = normalize_and_convert_to_tensor(
        ims, label, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    )

    assert isinstance(tensor_ims, torch.Tensor)
    assert isinstance(tensor_label, torch.Tensor)
    assert tensor_ims.shape == torch.Size([3, 1, 224, 224])
    assert tensor_label.shape == torch.Size([224, 224])


def test_process_and_augment():
    x = np.random.rand(6, 256, 256)
    y = np.random.rand(256, 256)
    processed_ims, processed_label = process_and_augment(
        x, y, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], temporal_size=2, im_size=224
    )

    assert processed_ims.shape == torch.Size([3, 2, 224, 224])
    assert processed_label.shape == torch.Size([224, 224])


def test_get_arr_flood():
    test_fname = "tests/data/sample.tif"
    result = get_arr_flood(test_fname, is_label=False)
    assert isinstance(result, np.ndarray)


def test_process_data():
    im_test_fname = "tests/data/sample.tif"
    mask_test_fname = "tests/data/sample.tif"

    arr_x, arr_y = process_data(im_test_fname, mask_test_fname)

    assert isinstance(arr_x, np.ndarray)
    assert isinstance(arr_y, np.ndarray)
    assert arr_x.shape[:-2] == arr_y.shape[:-2]
