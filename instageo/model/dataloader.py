"""Dataloader Module."""

import os
import random
from functools import partial
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
from absl import logging
from PIL import Image
from torchvision import transforms

from instageo.data.geo_utils import open_mf_tiff_dataset


def random_crop_and_flip(
    ims: List[Image.Image], label: Image.Image, im_size: int
) -> Tuple[List[Image.Image], Image.Image]:
    """Apply random cropping and flipping transformations to the given images and label.

    Args:
        ims (List[Image.Image]): List of PIL Image objects representing the images.
        label (Image.Image): A PIL Image object representing the label.

    Returns:
        Tuple[List[Image.Image], Image.Image]: A tuple containing the transformed list of
        images and label.
    """
    i, j, h, w = transforms.RandomCrop.get_params(ims[0], (im_size, im_size))

    ims = [transforms.functional.crop(im, i, j, h, w) for im in ims]
    label = transforms.functional.crop(label, i, j, h, w)

    if random.random() > 0.5:
        ims = [transforms.functional.hflip(im) for im in ims]
        label = transforms.functional.hflip(label)

    if random.random() > 0.5:
        ims = [transforms.functional.vflip(im) for im in ims]
        label = transforms.functional.vflip(label)

    return ims, label


def normalize_and_convert_to_tensor(
    ims: List[Image.Image],
    label: Image.Image | None,
    mean: List[float],
    std: List[float],
    temporal_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize the images and label and convert them to PyTorch tensors.

    Args:
        ims (List[Image.Image]): List of PIL Image objects representing the images.
        label (Image.Image | None): A PIL Image object representing the label.
        mean (List[float]): The mean of each channel in the image
        std (List[float]): The standard deviation of each channel in the image
        temporal_size: The number of temporal steps

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors representing the normalized
        images and label.
    """
    norm = transforms.Normalize(mean, std)
    ims_tensor = torch.stack([transforms.ToTensor()(im).squeeze() for im in ims])
    _, h, w = ims_tensor.shape
    ims_tensor = ims_tensor.reshape([temporal_size, -1, h, w])  # T*C,H,W -> T,C,H,W
    ims_tensor = torch.stack([norm(im) for im in ims_tensor]).permute(
        [1, 0, 2, 3]
    )  # T,C,H,W -> C,T,H,W
    if label:
        label = torch.from_numpy(np.array(label)).squeeze()
    return ims_tensor, label


def process_and_augment(
    x: np.ndarray,
    y: np.ndarray | None,
    mean: List[float],
    std: List[float],
    temporal_size: int = 1,
    im_size: int = 224,
    train: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process and augment the given images and labels.

    Args:
        x (np.ndarray): Numpy array representing the images.
        y (np.ndarray): Numpy array representing the label.
        mean (List[float]): The mean of each channel in the image
        std (List[float]): The standard deviation of each channel in the image
        temporal_size: The number of temporal steps
        train: To identify training mode.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors representing the processed
        and augmented images and label.
    """
    ims = x.copy()
    label = None
    # convert to PIL for easier transforms
    ims = [Image.fromarray(im) for im in ims]
    if not (y is None):
        label = y.copy()
        label = Image.fromarray(label.squeeze())
    if train:
        ims, label = random_crop_and_flip(ims, label, im_size)
    ims, label = normalize_and_convert_to_tensor(ims, label, mean, std, temporal_size)
    return ims, label


def crop_array(
    arr: np.ndarray, left: int, top: int, right: int, bottom: int
) -> np.ndarray:
    """Crop Numpy Image.

    Crop a given array (image) using specified left, top, right, and bottom indices.

    This function supports cropping both grayscale (2D) and color (3D) images.

    Args:
        arr (np.ndarray): The input array (image) to be cropped.
        left (int): The left boundary index for cropping.
        top (int): The top boundary index for cropping.
        right (int): The right boundary index for cropping.
        bottom (int): The bottom boundary index for cropping.

    Returns:
        np.ndarray: The cropped portion of the input array (image).

    Raises:
        ValueError: If the input array is not 2D or 3D.
    """
    if len(arr.shape) == 2:  # Grayscale image (2D array)
        return arr[top:bottom, left:right]
    elif len(arr.shape) == 3:  # Color image (3D array)
        return arr[:, top:bottom, left:right]
    elif len(arr.shape) == 4:  # Color image (3D array)
        return arr[:, :, top:bottom, left:right]
    else:
        raise ValueError("Input array must be a 2D, 3D or 4D array")


def process_test(
    x: np.ndarray,
    y: np.ndarray,
    mean: List[float],
    std: List[float],
    temporal_size: int = 1,
    img_size: int = 512,
    crop_size: int = 224,
    stride: int = 224,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process and augment test data.

    Args:
        x (np.ndarray): Input image array.
        y (np.ndarray): Corresponding mask array.
        mean (List[float]): Mean values for normalization.
        std: (List[float]): Standard deviation values for normalization.
        temporal_size (int, optional): Temporal dimension size. Defaults to 1.
        img_size (int, optional): Size of the input images. Defaults to
            512.
        crop_size (int, optional): Size of the crops to be extracted from the
            images. Defaults to 224.
        stride (int, optional): Stride for cropping. Defaults to 224.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of tensors containing the processed
            images and masks.
    """
    preprocess_func = partial(
        process_and_augment,
        mean=mean,
        std=std,
        temporal_size=temporal_size,
        train=False,
    )

    img_crops, mask_crops = [], []
    width, height = img_size, img_size

    for top in range(0, height - crop_size + 1, stride):
        for left in range(0, width - crop_size + 1, stride):
            bottom = top + crop_size
            right = left + crop_size

            img_crops.append(crop_array(x, left, top, right, bottom))
            mask_crops.append(crop_array(y, left, top, right, bottom))

    samples = [preprocess_func(x, y) for x, y in zip(img_crops, mask_crops)]
    imgs = torch.stack([sample[0] for sample in samples])
    labels = torch.stack([sample[1] for sample in samples])
    return imgs, labels


def get_raster_data(
    fname: str | dict[str, dict[str, str]],
    is_label: bool = True,
    bands: List[int] | None = None,
    no_data_value: int | None = -9999,
    mask_cloud: bool = True,
) -> np.ndarray:
    """Load and process raster data from a file.

    Args:
        fname (str): Filename to load data from.
        is_label (bool): Whether the file is a label file.
        bands (List[int]): Index of bands to select from array.
        no_data_value (int | None): NODATA value in image raster.
        mask_cloud (bool): Perform cloud masking.

    Returns:
        np.ndarray: Numpy array representing the processed data.
    """
    if isinstance(fname, dict):
        data, _ = open_mf_tiff_dataset(fname, mask_cloud)
        data = data.fillna(no_data_value)
        data = data.band_data.values
    else:
        with rasterio.open(fname) as src:
            data = src.read()
    if (not is_label) and bands:
        data = data[bands, ...]
    # For some reasons, some few HLS tiles are not scaled. In the following lines,
    # we find and scale them
    bands = []
    for band in data:
        if band.max() > 10:
            band *= 0.0001
        bands.append(band)
    data = np.stack(bands, axis=0)
    return data


def process_data(
    im_fname: str,
    mask_fname: str | None = None,
    no_data_value: int | None = -9999,
    reduce_to_zero: bool = False,
    replace_label: Tuple | None = None,
    bands: List[int] | None = None,
    constant_multiplier: float = 1.0,
    mask_cloud: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process image and mask data from filenames.

    Args:
        im_fname (str): Filename for the image data.
        mask_fname (str | None): Filename for the mask data.
        bands (List[int]): Indices of bands to select from array.
        no_data_value (int | None): NODATA value in image raster.
        reduce_to_zero (bool): Reduces the label index to start from Zero.
        replace_label (Tuple): Tuple of value to replace and the replacement value.
        constant_multiplier (float): Constant multiplier for image.
        mask_cloud (bool): Perform cloud masking.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of numpy arrays representing the processed
        image and mask data.
    """
    arr_x = get_raster_data(
        im_fname,
        is_label=False,
        bands=bands,
        no_data_value=no_data_value,
        mask_cloud=mask_cloud,
    )
    arr_x = arr_x * constant_multiplier
    if mask_fname:
        arr_y = get_raster_data(mask_fname)
        if replace_label:
            arr_y = np.where(arr_y == replace_label[0], replace_label[1], arr_y)
        if reduce_to_zero:
            arr_y -= 1
    else:
        arr_y = None
    return arr_x, arr_y


def load_data_from_csv(fname: str, input_root: str) -> List[Tuple[str, str]]:
    """Load data file paths from a CSV file.

    Args:
        fname (str): Filename of the CSV file.
        input_root (str): Root directory for input images and labels.

    Returns:
        List[Tuple[str, str]]: A list of tuples, each containing file paths for input
        image and label image.
    """
    file_paths = []
    data = pd.read_csv(fname)
    for _, row in data.iterrows():
        im_path = os.path.join(input_root, row["Input"])
        mask_path = os.path.join(input_root, row["Label"])
        if os.path.exists(im_path):
            try:
                with rasterio.open(im_path) as src:
                    _ = src.crs
                file_paths.append((im_path, mask_path))
            except Exception as e:
                logging.error(e)
                continue
    return file_paths


class InstaGeoDataset(torch.utils.data.Dataset):
    """InstaGeo PyTorch Dataset for Loading and Handling HLS Data."""

    def __init__(
        self,
        filename: str,
        input_root: str,
        preprocess_func: Callable,
        no_data_value: int | None,
        replace_label: Tuple,
        reduce_to_zero: bool,
        constant_multiplier: float,
        bands: List[int] | None = None,
    ):
        """Dataset Class for loading and preprocessing Sentinel11Floods dataset.

        Args:
            filename (str): Filename of the CSV file containing data paths.
            input_root (str): Root directory for input images and labels.
            preprocess_func (Callable): Function to preprocess the data.
            bands (List[int]): Indices of bands to select from array.
            no_data_value (int | None): NODATA value in image raster.
            reduce_to_zero (bool): Reduces the label index to start from Zero.
            replace_label (Tuple): Tuple of value to replace and the replacement value.
            constant_multiplier (float): Constant multiplier for image.

        """
        self.input_root = input_root
        self.preprocess_func = preprocess_func
        self.bands = bands
        self.file_paths = load_data_from_csv(filename, input_root)
        self.no_data_value = no_data_value
        self.replace_label = replace_label
        self.reduce_to_zero = reduce_to_zero
        self.constant_multiplier = constant_multiplier

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves a sample from dataset.

        Args:
            i (int): Sample index to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors representing the
            processed images and label.
        """
        im_fname, mask_fname = self.file_paths[i]
        arr_x, arr_y = process_data(
            im_fname,
            mask_fname,
            no_data_value=self.no_data_value,
            replace_label=self.replace_label,
            reduce_to_zero=self.reduce_to_zero,
            bands=self.bands,
            constant_multiplier=self.constant_multiplier,
        )
        return self.preprocess_func(arr_x, arr_y)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.file_paths)
