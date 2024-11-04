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

"""Geo Utils Module."""

import bisect
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


def create_segmentation_map(
    chip: Any,
    df: pd.DataFrame,
    no_data_value: int,
) -> np.ndarray:
    """Create a segmentation map for the chip using the DataFrame.

    Args:
        chip (Any): The chip (subset of the original data) for which the segmentation
            map is being created.
        df (pd.DataFrame): DataFrame containing the data to be used in the segmentation
            map.
        no_data_value (int): Value to be used for pixels with no data.

    Returns:
        np.ndarray: The created segmentation map as a NumPy array.
    """
    seg_map = chip.isel(band=0).assign(
        {
            "band_data": (
                ("y", "x"),
                no_data_value * np.ones((chip.sizes["x"], chip.sizes["y"])),
            )
        }
    )
    df = df[
        (chip["x"].min().item() <= df["geometry"].x)
        & (df["geometry"].x <= chip["x"].max().item())
        & (chip["y"].min().item() <= df["geometry"].y)
        & (df["geometry"].y <= chip["y"].max().item())
    ]
    # Use a tolerance of 30 meters
    for _, row in df.iterrows():
        nearest_index = seg_map.sel(
            x=row["geometry"].x, y=row["geometry"].y, method="nearest", tolerance=30
        )
        seg_map.loc[
            dict(x=nearest_index["x"].values.item(), y=nearest_index["y"].values.item())
        ] = row["label"]
    return seg_map.band_data.squeeze()


def get_chip_coords(
    df: gpd.GeoDataFrame, tile: xr.DataArray, chip_size: int
) -> list[tuple[int, int]]:
    """Get Chip Coordinates.

    Given a list of x,y coordinates tuples of a point and an xarray dataarray, this
    function returns the corresponding x,y indices of the grid where each point will fall
    when the DataArray is gridded such that each grid has size `chip_size`
    indices where it will fall.

    Args:
        gdf (gpd.GeoDataFrame): GeoPandas dataframe containing the point.
        tile (xr.DataArray): Tile DataArray.
        chip_size (int): Size of each chip.

    Returns:
        List of chip indices.
    """
    coords = []
    for _, row in df.iterrows():
        x = bisect.bisect_left(tile["x"].values, row["geometry"].x)
        y = bisect.bisect_left(tile["y"].values[::-1], row["geometry"].y)
        y = tile.sizes["y"] - y - 1
        x = int(x // chip_size)
        y = int(y // chip_size)
        coords.append((x, y))
    return coords
