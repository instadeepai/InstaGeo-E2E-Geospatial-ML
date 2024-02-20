"""Geo Utils Module."""

from typing import Dict

import rasterio
import xarray as xr
from rasterio.crs import CRS


def open_mf_tiff_dataset(band_files: Dict[str, str]) -> tuple[xr.Dataset, CRS]:
    """Open multiple TIFF files as an xarray Dataset.

    Args:
        band_files (Dict[str, str]): A dictionary mapping band names to file paths.

    Returns:
        (xr.Dataset, CRS): A tuple of xarray Dataset combining data from all the
            provided TIFF files and its CRS
    """
    band_paths = list(band_files.values())
    mf_dataset = xr.open_mfdataset(
        band_paths,
        concat_dim="band",
        combine="nested",
    )
    with rasterio.open(band_paths[0]) as src:
        crs = src.crs
    return mf_dataset, crs
