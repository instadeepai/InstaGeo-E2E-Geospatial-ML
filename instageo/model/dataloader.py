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

def compute_indices(image):
    """Compute NDVI, EVI, NDWI, and NDSI from Sentinel-2 bands."""
    B2, B3, B4, B8, B11, B12 = image

    NDVI = (B8 - B4) / (B8 + B4 + 1e-6)
    EVI = 2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1 + 1e-6)
    NDWI = (B3 - B8) / (B3 + B8 + 1e-6)
    NDSI = (B11 - B8) / (B11 + B8 + 1e-6)

    return np.stack([NDVI, EVI, NDWI, NDSI], axis=0)

def compute_differences(current, previous):
    """Compute ∆NDVI and ∆NDWI (temporal changes)."""
    delta_NDVI = current[0] - previous[0]
    delta_NDWI = current[2] - previous[2]
    return np.stack([delta_NDVI, delta_NDWI], axis=0)

def process_image(image_series):
    """
    Process an input time-series image with indices and differences.
    image_series = (3, 6, H, W) -> 3 time-steps, 6 spectral bands
    Returns (18, H, W) replacing original bands with computed indices and differences and reshaping.
    """
    indices_series = [compute_indices(img) for img in image_series]
    differences = [compute_differences(indices_series[i], indices_series[i-1]) for i in range(1, len(indices_series))]
    processed_bands = [np.concatenate([indices_series[i], differences[i-1]], axis=0) for i in range(1, 3)]
    processed_bands.insert(0, np.concatenate([indices_series[0], np.zeros_like(differences[0])], axis=0))  # First timestep has no difference
    final_input = np.concatenate(processed_bands, axis=0)  # Reshape to (18, H, W)
    return final_input

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
    augment: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process and augment the given images and labels.

    Args:
        x (np.ndarray): Numpy array representing the images.
        y (np.ndarray): Numpy array representing the label.
        mean (List[float]): The mean of each channel in the image
        std (List[float]): The standard deviation of each channel in the image
        temporal_size: The number of temporal steps
        augment: Flag to perform augmentations in training mode.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors representing the processed
        and augmented images and label.
    """
    ims = x.copy()
    label = None
    # convert to PIL for easier transforms
    ims = [Image.fromarray(im) for im in ims]
    if y is not None:
        label = Image.fromarray(y.copy().squeeze())
    if augment:
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


def get_raster_data(fname, is_label=True, bands=None, no_data_value=-9999, mask_cloud=True, water_mask=False):
    """Load and process raster data from a file."""
    if isinstance(fname, dict):
        data, _, _ = open_mf_tiff_dataset(fname, load_masks=False)
        data = data.fillna(no_data_value).band_data.values
    else:
        with rasterio.open(fname) as src:
            data = src.read()

    if not is_label and bands:
        data = data[bands, ...]


    # Fix: Ensure the shape is (3, 6, H, W)
    if data.shape[0] == 18:  
        data = data.reshape(3, 6, data.shape[1], data.shape[2])  
    
    print(f"Before Data shape is {data.shape}")
    # Call process_image to replace the original 6 bands
    if not is_label:
        data = process_image(data)  # Now returns (6, H, W), replacing the old bands
    print(f"After Data shape is {data.shape}")
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
        water_mask=False,
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


def load_data_from_csv(fname: str, input_root: str) -> List[Tuple[str, str | None]]:
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
    label_present = True if "Label" in data.columns else False
    for _, row in data.iterrows():
        im_path = os.path.join(input_root, row["Input"])
        mask_path = (
            None if not label_present else os.path.join(input_root, row["Label"])
        )
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
        include_filenames: bool = False,
    ):
        """Dataset Class for loading and preprocessing the dataset.

        Args:
            filename (str): Filename of the CSV file containing data paths.
            input_root (str): Root directory for input images and labels.
            preprocess_func (Callable): Function to preprocess the data.
            bands (List[int]): Indices of bands to select from array.
            no_data_value (int | None): NODATA value in image raster.
            reduce_to_zero (bool): Reduces the label index to start from Zero.
            replace_label (Tuple): Tuple of value to replace and the replacement value.
            constant_multiplier (float): Constant multiplier for image.
            include_filenames (bool): Flag that determines whether to return filenames.

        """
        self.input_root = input_root
        self.preprocess_func = preprocess_func
        self.bands = bands
        self.file_paths = load_data_from_csv(filename, input_root)
        self.no_data_value = no_data_value
        self.replace_label = replace_label
        self.reduce_to_zero = reduce_to_zero
        self.constant_multiplier = constant_multiplier
        self.include_filenames = include_filenames

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
        if self.include_filenames:
            return self.preprocess_func(arr_x, arr_y), im_fname
        else:
            return self.preprocess_func(arr_x, arr_y)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.file_paths)

