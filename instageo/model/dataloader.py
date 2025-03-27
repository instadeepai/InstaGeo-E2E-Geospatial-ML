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
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
import xarray as xr
from absl import logging
from PIL import Image
from torchvision import transforms


def open_mf_tiff_dataset(band_files: dict[str, Any]) -> xr.Dataset:
    """Open multiple TIFF files as an xarray Dataset.

    Args:
        band_files (Dict[str, Dict[str, str]]): A dictionary mapping band names to file paths.

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
    return bands_dataset


def random_brightness_contrast(
    ims: list[Image.Image],
    brightness_factor_range: tuple[float, float] = (0.8, 1.2),
    contrast_factor_range: tuple[float, float] = (0.8, 1.2),
    max_pixel_value: float = 10000.0,
) -> List[Image.Image]:
    """Applies Random Brightness and Contrast.

    Args:
        ims (List[PIL.Image]): List of single-band PIL images with values in [0,max_pixel_value]
        brightness_factor_range (tuple, optional): Brightness factor range. Defaults to (0.8, 1.2).
        contrast_factor_range (tuple, optional): Contrast factor range. Defaults to (0.8, 1.2).
        max_pixel_value (float): maximum pixel value in `ims`.

    Returns:
        List[PIL.Image]: List of images (same shape and range) with random brightness and contrast
            applied.
    """
    new_ims = []
    bright_factor = random.uniform(*brightness_factor_range)
    contrast_factor = random.uniform(*contrast_factor_range)
    for im in ims:
        arr = torch.from_numpy(np.array(im, dtype=np.float32))
        # Apply brightness shift
        arr = arr * bright_factor
        # Apply contrast shift around the mean
        mean_val = arr.mean()
        arr = (arr - mean_val) * contrast_factor + mean_val
        arr.clamp_(0, max_pixel_value)
        new_ims.append(Image.fromarray(arr.numpy()))
    return new_ims


def add_gaussian_noise(
    ims: list[Image.Image], noise_std: float = 0.05, max_pixel_value: float = 10000.0
) -> List[Image.Image]:
    """Add random Gaussian noise to satellite images.

    Args:
        ims (List[PIL.Image]): List of single-band PIL images with values in [0,max_pixel_value].
        noise_std (float): Standard deviation of the Gaussian noise in the [0,1] range.
            e.g., 0.05 means noise with std dev = 5% of max (which is 1 in normalized space).
        max_pixel_value (float): maximum pixel value in `ims`.

    Returns:
        List[PIL.Image]: List of noisy images (same shape and range).
    """
    noisy_ims = []
    for im in ims:
        arr = np.array(im, dtype=np.float32)
        arr = np.clip(arr, 0, max_pixel_value) / max_pixel_value
        arr_tensor = torch.from_numpy(arr)

        # Add Gaussian noise
        noise = torch.randn_like(arr_tensor) * noise_std
        arr_noisy = arr_tensor + noise

        arr_noisy = torch.clamp(arr_noisy, 0.0, 1.0)
        arr_noisy = arr_noisy * max_pixel_value
        arr_noisy_uint16 = arr_noisy.numpy().astype(np.uint16)
        im_noisy = Image.fromarray(arr_noisy_uint16)

        noisy_ims.append(im_noisy)
    return noisy_ims


def add_gaussian_blur(
    ims: list[Image.Image],
    kernel_size: int = 3,
    sigma: tuple[float, float] = (0.1, 2.0),
    max_pixel_value: float = 10000.0,
) -> List[Image.Image]:
    """Apply Gaussian blur to each satellite band image in [0, max_pixel_value].

    Args:
        ims (List[PIL.Image]): List of single-band PIL images (range [0,max_pixel_value]).
        kernel_size (int): Gaussian kernel size for the blur.
        sigma (tuple[float]): Standard deviation range for the Gaussian kernel.
        max_pixel_value (float): maximum pixel value in `ims`.

    Returns:
        List[PIL.Image]: List of blurred images (same shape and range).
    """
    blurred_ims = []
    for im in ims:
        arr = np.array(im, dtype=np.float32)
        # Normalize [0,max_pixel_value] to [0,1]
        arr = np.clip(arr, 0, max_pixel_value) / max_pixel_value
        arr_tensor = torch.from_numpy(arr).unsqueeze(0)

        # Apply gaussian blur
        arr_blurred = transforms.functional.gaussian_blur(
            arr_tensor, kernel_size=kernel_size, sigma=sigma
        )

        # Un-Normalize
        arr_blurred = torch.clamp(arr_blurred, 0.0, 1.0)
        arr_blurred = arr_blurred.squeeze(0) * max_pixel_value

        arr_blurred = arr_blurred.numpy().astype(np.uint16)
        im_blurred = Image.fromarray(arr_blurred)
        blurred_ims.append(im_blurred)
    return blurred_ims


def crop_image_and_label(
    ims: List[Image.Image], label: Image.Image, im_size: int
) -> Tuple[List[Image.Image], Image.Image]:
    """Crop `ims` and `label` to `im_size`.

    Args:
        ims (List[Image.Image]): List of PIL Image objects representing the images.
        label (Image.Image): A PIL Image object representing the label.
        im_size (int): The crop size (height/width).

    Returns:
        Tuple[List[Image.Image], Image.Image]: A tuple containing list of cropped images and
            the label.
    """
    # Crop to im_size
    i, j, h, w = transforms.RandomCrop.get_params(ims[0], (im_size, im_size))
    ims = [transforms.functional.crop(im, i, j, h, w) for im in ims]
    label = transforms.functional.crop(label, i, j, h, w)
    return ims, label


def random_augs(
    ims: List[Image.Image],
    label: Image.Image,
    label_no_data_value: int,
    chip_no_data_value: int,
    max_pixel_value: float = 10000.0,
) -> Tuple[List[Image.Image], Image.Image]:
    """Apply Random Augmentations.

    Apply a series of random transformations (cropping, flipping, rotation, color jitter,
    blur, etc.) to a list of PIL Images and a segmentation label.

    Args:
        ims (List[Image.Image]): List of PIL Image objects representing the images.
        label (Image.Image): A PIL Image object representing the label.
        chip_no_data_value (int): Value of no_data pixels in chip.
        label_no_data_value (int): Value of no_data pixels in label.
        max_pixel_value (float): maximum pixel value in `ims`.

    Returns:
        Tuple[List[Image.Image], Image.Image]: A tuple containing the transformed list of
        images and the label.
    """
    # Random Horizontal Flip
    if random.random() < 0.5:
        ims = [transforms.functional.hflip(im) for im in ims]
        label = transforms.functional.hflip(label)

    # Random Vertical Flip
    if random.random() < 0.5:
        ims = [transforms.functional.vflip(im) for im in ims]
        label = transforms.functional.vflip(label)

    # Random Rotation
    if random.random() < 0.5:
        angle = random.uniform(-45, 45)
        ims = [
            transforms.functional.rotate(im, angle, fill=chip_no_data_value)
            for im in ims
        ]
        label = transforms.functional.rotate(label, angle, fill=label_no_data_value)

    # Color Jitter
    if random.random() < 0.5:
        ims = random_brightness_contrast(ims, max_pixel_value=max_pixel_value)

    # Random Gaussian Blur
    if random.random() < 0.5:
        ims = add_gaussian_blur(
            ims, kernel_size=3, sigma=(0.1, 2.0), max_pixel_value=max_pixel_value
        )

    # Add Random Noise
    if random.random() < 0.5:
        ims = add_gaussian_noise(ims, noise_std=0.05, max_pixel_value=max_pixel_value)
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
    ims_tensor = torch.stack(
        [transforms.ToTensor()(im).float().squeeze() for im in ims]
    )
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
    crop: bool = True,
    label_no_data_value: int = -1,
    chip_no_data_value: int = 0,
    max_pixel_value: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process and augment the given images and labels.

    Args:
        x (np.ndarray): Numpy array representing the images.
        y (np.ndarray): Numpy array representing the label.
        mean (List[float]): The mean of each channel in the image
        std (List[float]): The standard deviation of each channel in the image
        temporal_size: The number of temporal steps
        augment: Flag to perform augmentations in training mode.
        chip_no_data_value (int): Value of no_data pixels in chip.
        label_no_data_value (int): Value of no_data pixels in label.
        max_pixel_value (float): maximum pixel value in `ims`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors representing the processed
        and augmented images and label.
    """
    ims = x.copy()
    label = None
    # convert to PIL for easier transforms
    ims = [Image.fromarray(im) for im in ims]
    if y is not None:
        label = Image.fromarray(y.copy().astype(np.float32).squeeze())
    if crop:
        # During evaluation we want to handle cropping manually
        ims, label = crop_image_and_label(ims, label, im_size)
    if augment:
        # During evaluation we don't want to apply augs
        ims, label = random_augs(
            ims,
            label,
            label_no_data_value=label_no_data_value,
            chip_no_data_value=chip_no_data_value,
            max_pixel_value=max_pixel_value,
        )
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
        im_size=img_size,
        temporal_size=temporal_size,
        augment=False,
        crop=False,
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
    water_mask: bool = False,
) -> np.ndarray:
    """Load and process raster data from a file.

    Args:
        fname (str): Filename to load data from.
        is_label (bool): Whether the file is a label file.
        bands (List[int]): Index of bands to select from array.
        no_data_value (int | None): NODATA value in image raster.
        mask_cloud (bool): Perform cloud masking.
        water_mask (bool): Perform water masking.

    Returns:
        np.ndarray: Numpy array representing the processed data.
    """
    if isinstance(fname, dict):
        # @TODO This is used during sliding window inference so masking and processing needs to
        # match what is done to chips in data component
        data = open_mf_tiff_dataset(fname)
        data = data.fillna(no_data_value)
        data = data.band_data.values
    else:
        with rasterio.open(fname) as src:
            data = src.read()
    if (not is_label) and bands:
        data = data[bands, ...]
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


def mask_label_with_chip(
    chips_path: str,
    labels_path: str,
    chip_no_data_value: int = 0,
    label_no_data_value: int = -1,
) -> bool:
    """Masks the label raster using a corresponding chip raster.

    Args:
        chips_path (str): Chip file path.
        labels_path (str): Label file path.
        chip_no_data_value (int): Value of no_data pixels in chip.
        label_no_data_value (int): Value of no_data pixels in label.

    Returns:
        bool: Whether all values in the label array are NaN.
    """
    with rasterio.open(chips_path) as src:
        num_steps = max(1, src.count // 6)
        stacked = src.read([(6 * i) + 1 for i in range(num_steps)])
        stacked = np.where(stacked == chip_no_data_value, 0, 1).all(0).astype(int)

    with rasterio.open(labels_path) as src:
        label = src.read(1)
        label = np.where(label == label_no_data_value, np.nan, label)
        label = np.where(stacked == 0, np.nan, label)

    all_nan = np.all(np.isnan(label))
    return all_nan


def get_valid_filepaths(
    fname: str, input_root: str, no_data_value: int = -9999, ignore_index: int = -1
) -> List[Tuple[str, Optional[str]]]:
    """Get valid file paths from a CSV file.

    Performs a QA check on chips and labels and selects the ones that are good. Good chips have
    non-null data while good labels have non-null values in the corresponding pixel location in the
    chip.

    Args:
        fname (str): Filename of the CSV file.
        input_root (str): Root directory for input images and labels.

    Returns:
        List[Tuple[str, Optional[str]]]: A list of tuples, each containing file paths for input
        image and label image. The label image may be None if not present.
    """
    file_paths: List[Tuple[str, Optional[str]]] = []

    data = pd.read_csv(fname)
    label_present = "Label" in data.columns

    for _, row in data.iterrows():
        im_path = os.path.join(input_root, row["Input"])
        mask_path = os.path.join(input_root, row["Label"]) if label_present else None

        if os.path.exists(im_path):
            try:
                with rasterio.open(im_path) as src:
                    _ = src.crs
                if mask_path is not None:
                    all_nan = mask_label_with_chip(
                        im_path,
                        mask_path,
                        chip_no_data_value=no_data_value,
                        label_no_data_value=ignore_index,
                    )
                    if not all_nan:
                        file_paths.append((im_path, mask_path))
                else:
                    logging.warning(f"{mask_path} is not a valid seg_map")
                    file_paths.append((im_path, None))
            except Exception as e:
                logging.error(e)
                continue
    print(f"Dropped a total of {len(data) - len(file_paths)} rows")
    return file_paths


class InstaGeoDataset(torch.utils.data.Dataset):
    """InstaGeo PyTorch Dataset for Loading and Handling HLS Data."""

    def __init__(
        self,
        filename: str,
        input_root: str,
        preprocess_func: Callable,
        chip_no_data_value: int,
        label_no_data_value: int,
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
            chip_no_data_value (int | None): NODATA value in image raster.
            label_no_data_value (int | None): NODATA value in label raster.
            reduce_to_zero (bool): Reduces the label index to start from Zero.
            replace_label (Tuple): Tuple of value to replace and the replacement value.
            constant_multiplier (float): Constant multiplier for image.
            include_filenames (bool): Flag that determines whether to return filenames.

        """
        self.input_root = input_root
        self.preprocess_func = preprocess_func
        self.bands = bands
        self.file_paths = get_valid_filepaths(
            filename, input_root, chip_no_data_value, label_no_data_value
        )
        self.no_data_value = chip_no_data_value
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
