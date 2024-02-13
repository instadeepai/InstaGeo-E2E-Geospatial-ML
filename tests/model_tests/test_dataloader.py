import numpy as np
import pytest
import torch
from PIL import Image

from instageo.model.dataloader import (
    crop_array,
    get_raster_data,
    normalize_and_convert_to_tensor,
    process_and_augment,
    process_data,
    process_test,
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
    result = get_raster_data(test_fname, is_label=False)
    assert isinstance(result, np.ndarray)


def test_process_data():
    im_test_fname = "tests/data/sample.tif"
    mask_test_fname = "tests/data/sample.tif"

    arr_x, arr_y = process_data(im_test_fname, mask_test_fname)

    assert isinstance(arr_x, np.ndarray)
    assert isinstance(arr_y, np.ndarray)
    assert arr_x.shape[:-2] == arr_y.shape[:-2]


def test_crop_2d_array():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    cropped = crop_array(arr, 1, 1, 3, 3)
    expected = np.array([[5, 6], [8, 9]])
    assert np.array_equal(cropped, expected)


def test_crop_3d_array():
    arr = np.zeros((3, 3, 3))  # Example 3D array
    arr[:, 1, 1] = 1  # Modify the array for testing
    cropped = crop_array(arr, 1, 1, 3, 3)
    expected = np.zeros((3, 2, 2))
    expected[:, 0, 0] = 1
    assert np.array_equal(cropped, expected)


def test_invalid_dimensions():
    arr = np.array([1, 2, 3])  # 1D array, not valid
    with pytest.raises(ValueError):
        crop_array(arr, 0, 0, 1, 1)


def test_boundary_conditions():
    arr = np.array([[1, 2], [3, 4]])
    cropped = crop_array(arr, 0, 0, 2, 2)
    assert np.array_equal(cropped, arr)


def test_non_integer_indices():
    arr = np.array([[1, 2], [3, 4]])
    with pytest.raises(TypeError):
        crop_array(arr, 0.5, 0.5, 1.5, 1.5)


def test_output_types_and_shapes():
    x = np.random.rand(3, 512, 512)
    y = np.random.rand(512, 512)
    mean = [0.5, 0.5, 0.5]
    std = [0.1, 0.1, 0.1]
    imgs, labels = process_test(x, y, mean, std)
    assert isinstance(imgs, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert imgs.shape == torch.Size([4, 3, 1, 224, 224])
    assert labels.shape == torch.Size([4, 224, 224])


def test_invalid_inputs():
    x = np.random.rand(3, 512, 512)
    y = np.random.rand(512, 512)
    mean = [0.5, 0.5, "invalid"]  # Invalid mean value
    std = [0.1, 0.1, 0.1]
    with pytest.raises(Exception):
        process_test(x, y, mean, std)
