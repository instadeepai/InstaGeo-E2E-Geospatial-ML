import os
import random
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import rioxarray
import torch
from PIL import Image
from torchvision import transforms


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
    label: Image.Image,
    mean: List[float],
    std: List[float],
    temporal_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize the images and label and convert them to PyTorch tensors.

    Args:
        ims (List[Image.Image]): List of PIL Image objects representing the images.
        label (Image.Image): A PIL Image object representing the label.
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
    ims_tensor = ims_tensor.reshape([temporal_size, -1, h, w])  # C*T,H,W -> T,C,H,W
    ims_tensor = torch.stack([norm(im) for im in ims_tensor]).permute(
        [1, 0, 2, 3]
    )  # T,C,H,W -> C,T,H,W
    label = transforms.ToTensor()(label).squeeze()
    return ims_tensor, label


# @TODO Create a version for inference
def process_and_augment(
    x: np.ndarray,
    y: np.ndarray,
    mean: List[float],
    std: List[float],
    temporal_size: int = 1,
    im_size: int = 224,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process and augment the given images and labels.

    Args:
        x (np.ndarray): Numpy array representing the images.
        y (np.ndarray): Numpy array representing the label.
        mean (List[float]): The mean of each channel in the image
        std (List[float]): The standard deviation of each channel in the image
        temporal_size: The number of temporal steps

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors representing the processed
        and augmented images and label.
    """
    ims, label = x.copy(), y.copy()

    # convert to PIL for easier transforms
    ims, label = [Image.fromarray(im) for im in ims], Image.fromarray(label.squeeze())
    ims, label = random_crop_and_flip(ims, label, im_size)
    ims, label = normalize_and_convert_to_tensor(ims, label, mean, std, temporal_size)
    return ims, label


def get_arr_flood(
    fname: str, is_label: bool = True, bands: List[int] | None = None
) -> np.ndarray:
    """Load and process flood data from a file.

    Args:
        fname (str): Filename to load data from.
        is_label (bool): Whether the file is a label file.
        bands (List[int]): Index of bands to select from array.

    Returns:
        np.ndarray: Numpy array representing the processed data.
    """
    data = rioxarray.open_rasterio(fname)
    data = data.to_numpy()
    if (not is_label) and bands:
        data = data[bands, ...]
    return data


def process_data(
    im_fname: str, mask_fname: str, bands: List[int] | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Process image and mask data from filenames.

    Args:
        im_fname (str): Filename for the image data.
        mask_fname (str): Filename for the mask data.
        bands (List[int]): Indices of bands to select from array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of numpy arrays representing the processed
        image and mask data.
    """
    arr_x = get_arr_flood(im_fname, is_label=False, bands=bands)
    # replace raster NODATA value with 0.0001
    arr_x = np.where(arr_x == -9999, 0.0001, arr_x).astype(np.float32)
    arr_y = get_arr_flood(mask_fname)
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
            file_paths.append((im_path, mask_path))
    return file_paths


class InstaGeoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filename: str,
        input_root: str,
        preprocess_func: Callable,
        bands: List[int] | None = None,
    ):
        """Dataset Class for loading and preprocessing Sentinel11Floods dataset.

        Args:
            filename (str): Filename of the CSV file containing data paths.
            input_root (str): Root directory for input images and labels.
            preprocess_func (Callable): Function to preprocess the data.
            bands (List[int]): Indices of bands to select from array.
        """

        self.input_root = input_root
        self.preprocess_func = preprocess_func
        self.bands = bands
        self.file_paths = load_data_from_csv(filename, input_root)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves a sample from dataset.

        Args:
            i (int): Sample index to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors representing the
            processed images and label.
        """
        im_fname, mask_fname = self.file_paths[i]
        arr_x, arr_y = process_data(im_fname, mask_fname, self.bands)
        return self.preprocess_func(arr_x, arr_y)

    def __len__(self) -> int:
        """Return length of dataset"""
        return len(self.file_paths)
