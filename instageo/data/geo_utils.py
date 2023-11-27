from typing import Dict, Tuple

import geopandas as gpd
import pandas as pd
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import Point


def read_csv_gdf(
    filename: str, src_crs: int = 4326, dst_crs: int | None = None
) -> gpd.GeoDataFrame:
    """Reads a CSV file into a GeoDataFrame.

    Args:
        filename (str): The path to the CSV file.
        src_crs (int, optional): The EPSG code of the source coordinate reference system.
            Defaults to 4326 (WGS 84).
        dst_crs (int, optional): The EPSG code of the destination coordinate reference
            system. If specified, reprojects the data to this CRS.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame with the data from the CSV file.

    Assumes:
        - The CSV file contains columns 'x' and 'y' for longitude and latitude,
            respectively.
    """

    # Read the CSV file
    df = pd.read_csv(filename)

    # Convert the DataFrame to a GeoDataFrame with Point geometries
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.x, df.y)])

    # Set the CRS of the GeoDataFrame
    gdf.set_crs(epsg=src_crs, inplace=True)

    # Reproject to the destination CRS if specified
    if dst_crs:
        gdf = gdf.to_crs(epsg=dst_crs)

    return gdf


def open_tiff_as_dataarray(file_path: str, band_name: str) -> xr.Dataset:
    """Open a TIFF file and return it as an xarray Dataset with specified band name.

    Args:
        file_path (str): The file path to the TIFF file.
        band_name (str): The name to assign to the band data in the Dataset.

    Returns:
        xr.Dataset: The opened TIFF file as an xarray Dataset.
    """
    ds = xr.open_dataset(file_path)
    ds = ds.squeeze()
    ds = ds.rename_vars({"band_data": band_name})
    transform, crs = get_transform_crs(file_path)
    ds.attrs["crs"] = crs
    ds.attrs["transform"] = transform
    return ds


def open_mf_tiff_dataset(band_files: Dict[str, str]) -> xr.Dataset:
    """Open multiple TIFF files as an xarray Dataset.

    Args:
        band_files (Dict[str, str]): A dictionary mapping band names to file paths.

    Returns:
        xr.Dataset: An xarray Dataset combining data from all the provided TIFF files.
    """
    data_arrays = [
        open_tiff_as_dataarray(path, band) for band, path in band_files.items()
    ]
    mf_dataset = xr.merge(data_arrays)
    return mf_dataset


def get_transform_crs(tif_file: str) -> Tuple[Affine, CRS]:
    """Get Transform and CRS.

    Retrieve the affine transform and CRS (Coordinate Reference System) from a TIFF
    file.

    Args:
        tif_file (str): The file path to the TIFF file.

    Returns:
        Affine: The affine transform of the TIFF file.
        CRS: The Coordinate Reference System of the TIFF file.
    """
    with rasterio.open(tif_file) as src:
        # Read the affine transform and CRS
        transform = src.transform
        crs = src.crs
    return transform, crs
