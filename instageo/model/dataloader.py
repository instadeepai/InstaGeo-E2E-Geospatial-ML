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

"""Dataloader Module."""

import os
import random
from functools import partial
from typing import Any, Callable, List, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
import xarray as xr
from absl import logging
from PIL import Image
from rasterio.crs import CRS
from torchvision import transforms
from torch.utils.data import Dataset


def open_mf_tiff_dataset(
    band_files: dict[str, Any], load_masks: bool
) -> tuple[xr.Dataset, xr.Dataset | None, CRS]:
    """Open multiple TIFF files as an xarray Dataset.

    Args:
        band_files (Dict[str, Dict[str, str]]): A dictionary mapping band names to file paths.
        load_masks (bool): Whether or not to load the masks files.

    Returns:
        (xr.Dataset, xr.Dataset | None, CRS): A tuple of xarray Dataset combining data from all the
            provided TIFF files, (optionally) the masks, and the CRS
    """
    band_paths = list(band_files["tiles"].values())
    bands_dataset = xr.open_mfdataset(
        band_paths,
        concat_dim="band",
        combine="nested",
        mask_and_scale=False,  # Scaling will be applied manually
    )
    bands_dataset.band_data.attrs["scale_factor"] = 1
    mask_paths = list(band_files["fmasks"].values())
    mask_dataset = (
        xr.open_mfdataset(
            mask_paths,
            concat_dim="band",
            combine="nested",
        )
        if load_masks
        else None
    )
    with rasterio.open(band_paths[0]) as src:
        crs = src.crs
    return bands_dataset, mask_dataset, crs








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


def compute_indices(image):
    """Compute NDVI, EVI, NDWI, and NDSI from Sentinel-2 bands."""
    B2, B3, B4, B8, B11, B12 = image

    NDVI = (B8 - B4) / (B8 + B4 + 1e-6)
    EVI = 2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1 + 1e-6)
    NDWI = (B3 - B8) / (B3 + B8 + 1e-6)
    NDSI = (B11 - B8) / (B11 + B8 + 1e-6)

    return np.stack([NDVI, EVI, NDWI, NDSI], axis=0)

# Function to compute temporal differences
def compute_differences(current, previous):
    """Compute ΔNDVI and ΔNDWI (temporal changes)."""
    delta_NDVI = current[0] - previous[0]
    delta_NDWI = current[2] - previous[2]
    return np.stack([delta_NDVI, delta_NDWI], axis=0)

# Function to process image time-series
def process_image(image_series):
    indices_series = [compute_indices(img) for img in image_series]
    differences = [compute_differences(indices_series[i], indices_series[i-1]) for i in range(1, len(indices_series))]

    final_input = np.concatenate([image_series[2], indices_series[2], differences[-1]], axis=0)
    return final_input  # Shape: (12, H, W)

# Function to load and process raster data
def get_raster_data(fname, is_label=True, bands=None, no_data_value=-9999):
    if isinstance(fname, dict):
        data, _, _ = open_mf_tiff_dataset(fname, load_masks=False)
        data = data.fillna(no_data_value).band_data.values
    else:
        with rasterio.open(fname) as src:
            data = src.read()
    if not is_label and bands:
        data = data[bands, ...]
    return data

# Function to process image and mask
def process_data(im_fname, mask_fname=None, bands=None, no_data_value=-9999):
    arr_x = get_raster_data(im_fname, is_label=False, bands=bands, no_data_value=no_data_value)
    assert arr_x.shape[0] == 3 and arr_x.shape[1] == 6, "Expected shape (3, 6, H, W)"

    arr_x = process_image(arr_x)

    if mask_fname:
        arr_y = get_raster_data(mask_fname)
    else:
        arr_y = None

    return arr_x, arr_y

# Function for data augmentation
def random_crop_and_flip(ims, label, im_size):
    i, j, h, w = transforms.RandomCrop.get_params(ims[0], (im_size, im_size))
    ims = [transforms.functional.crop(im, i, j, h, w) for im in ims]
    label = transforms.functional.crop(label, i, j, h, w) if label is not None else None

    if random.random() > 0.5:
        ims = [transforms.functional.hflip(im) for im in ims]
        label = transforms.functional.hflip(label) if label is not None else None

    if random.random() > 0.5:
        ims = [transforms.functional.vflip(im) for im in ims]
        label = transforms.functional.vflip(label) if label is not None else None

    return ims, label

# Function to normalize images and convert to tensor
def normalize_and_convert_to_tensor(ims, label, mean, std):
    norm = transforms.Normalize(mean, std)
    ims_tensor = torch.stack([transforms.ToTensor()(im).squeeze() for im in ims])
    ims_tensor = torch.stack([norm(im) for im in ims_tensor])
    
    if label is not None:
        label = torch.from_numpy(np.array(label)).squeeze()
    
    return ims_tensor, label

# Main processing function
def process_and_augment(x, y, mean, std, im_size=224, augment=True):
    ims = [Image.fromarray(im) for im in x]
    label = Image.fromarray(y.squeeze()) if y is not None else None

    if augment:
        ims, label = random_crop_and_flip(ims, label, im_size)

    ims, label = normalize_and_convert_to_tensor(ims, label, mean, std)
    return ims, label

# Function to process test-time images
def process_test(x, y, mean, std, temporal_size=1, img_size=512, crop_size=224, stride=224):
    preprocess_func = partial(
        process_and_augment,
        mean=mean,
        std=std,
        augment=False,
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

# PyTorch Dataset class
class InstaGeoDataset(Dataset):
    def __init__(self, filename, input_root, preprocess_func, bands=None, include_filenames=False, replace_label=None, reduce_to_zero=None):
        self.input_root = input_root
        self.preprocess_func = preprocess_func
        self.bands = bands
        self.file_paths = load_data_from_csv(filename, input_root)
        self.include_filenames = include_filenames

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        im_fname, mask_fname = self.file_paths[idx]
        arr_x, arr_y = process_data(im_fname, mask_fname, bands=self.bands)

        if self.include_filenames:
            return self.preprocess_func(arr_x, arr_y), im_fname
        else:
            return self.preprocess_func(arr_x, arr_y)

# Function to load data from CSV
def load_data_from_csv(fname, input_root):
    file_paths = []
    data = pd.read_csv(fname)
    label_present = "Label" in data.columns
    for _, row in data.iterrows():
        im_path = os.path.join(input_root, row["Input"])
        mask_path = None if not label_present else os.path.join(input_root, row["Label"])
        if os.path.exists(im_path):
            try:
                with rasterio.open(im_path) as src:
                    _ = src.crs
                file_paths.append((im_path, mask_path))
            except Exception as e:
                logging.error(e)
    return file_paths
