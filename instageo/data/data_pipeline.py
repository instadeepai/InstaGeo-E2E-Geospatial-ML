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

"""InstaGeo Data pipeline Module."""

import bisect
import os
from typing import Any, Callable

import geopandas as gpd
import mgrs
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point


def create_and_save_chips_with_seg_maps(
    mf_reader: Callable,
    tile_dict: dict[str, dict[str, str]],
    tile_id: str,
    df: pd.DataFrame,
    chip_size: int,
    output_directory: str,
    no_data_value: int,
    src_crs: int,
    mask_cloud: bool,
    water_mask: bool,
) -> tuple[list[str], list[str | None]]:
    """Chip Creator.

    Create chips and corresponding segmentation maps from a satellite image tile, read in as Xarray
    dataset, and save them to an output directory.

    Args:
        mf_reader (callable[dict[str, dict[str, str]], bool, bool]): A multi-file reader that
            accepts a dictionary of satellite image tile paths and reads it into an Xarray dataset.
            Optionally performs water and cloud masking based on the boolean flags passed.
        tile_dict (Dict): A dict mapping band names to tile filepath.
        tile_id (str): MGRS ID of the tiles in `tile_dict`.
        df (pd.DataFrame): DataFrame containing the labelled data for creating segmentation maps.
        chip_size (int): Size of each chip.
        output_directory (str): Directory where the chips and segmentation maps will be saved.
        no_data_value (int): Value to use for no data areas in the segmentation maps.
        src_crs (int): CRS of points in `df`
        mask_cloud (bool): Perform cloud masking if True.
        water_mask (bool): Perform water masking if True.

    Returns:
        A tuple containing the lists of created chips and segmentation maps.
    """
    ds, crs = mf_reader(tile_dict, mask_cloud, water_mask)
    df = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.x, df.y)])
    df.set_crs(epsg=src_crs, inplace=True)
    df = df.to_crs(crs=crs)

    df = df[
        (ds["x"].min().item() <= df["geometry"].x)
        & (df["geometry"].x <= ds["x"].max().item())
        & (ds["y"].min().item() <= df["geometry"].y)
        & (df["geometry"].y <= ds["y"].max().item())
    ]
    os.makedirs(output_directory, exist_ok=True)
    date_id = df.iloc[0]["date"].strftime("%Y%m%d")
    chips = []
    seg_maps: list[str | None] = []
    n_chips_x = ds.sizes["x"] // chip_size
    n_chips_y = ds.sizes["y"] // chip_size
    chip_coords = list(set(get_chip_coords(df, ds, chip_size)))
    for x, y in chip_coords:
        if (x >= n_chips_x) or (y >= n_chips_y):
            continue
        chip_id = f"{date_id}_{tile_id}_{x}_{y}"
        chip_name = f"chip_{chip_id}.tif"
        seg_map_name = f"seg_map_{chip_id}.tif"

        chip_filename = os.path.join(output_directory, "chips", chip_name)
        seg_map_filename = os.path.join(output_directory, "seg_maps", seg_map_name)
        if os.path.exists(chip_filename) or os.path.exists(seg_map_filename):
            continue

        chip = ds.isel(
            x=slice(x * chip_size, (x + 1) * chip_size),
            y=slice(y * chip_size, (y + 1) * chip_size),
        )
        if chip.count().values == 0:
            continue
        seg_map = create_segmentation_map(chip, df, no_data_value)
        if seg_map.where(seg_map != -1).count().values == 0:
            continue
        seg_maps.append(seg_map_name)
        seg_map.rio.to_raster(seg_map_filename)
        chip = chip.fillna(no_data_value)
        chips.append(chip_name)
        chip.band_data.rio.to_raster(chip_filename)
    return chips, seg_maps


def get_tile_info(
    data: pd.DataFrame, num_steps: int = 3, temporal_step: int = 10
) -> tuple[pd.DataFrame, list[tuple[str, list[str]]]]:
    """Get Tile Info.

    Retrieves a summary of all tiles required for a given dataset. The summary contains
    the desired start and end date for each tile. Also retrieves a list of queries
    that can be used to retieve the tiles for each observation in `data`.

    Args:
        data (pd.DataFrame): A dataframe containing observation records.
        num_steps (int): Number of temporal time steps
        temporal_step (int): Size of each temporal step.

    Returns:
        A `tile_info` dataframe and a list of `tile_queries`
    """
    data = data[["mgrs_tile_id", "input_features_date", "x", "y"]].reset_index(
        drop=True
    )
    tile_queries = []
    tile_info = []
    for _, (tile_id, date, lon, lat) in data.iterrows():
        history = []
        for i in range(num_steps):
            curr_date = date - pd.Timedelta(days=temporal_step * i)
            history.append(curr_date.strftime("%Y-%m-%d"))
            tile_info.append([tile_id, curr_date.strftime("%Y-%m-%d"), lon, lat])
        tile_queries.append((tile_id, history))
    tile_info = (
        pd.DataFrame(tile_info, columns=["tile_id", "date", "lon", "lat"])
        .groupby("tile_id")
        .agg(
            min_date=("date", "min"),
            max_date=("date", "max"),
            lon_min=("lon", "min"),
            lon_max=("lon", "max"),
            lat_min=("lat", "min"),
            lat_max=("lat", "max"),
        )
    ).reset_index()
    return tile_info, tile_queries


def get_tiles(data: pd.DataFrame, min_count: int = 100) -> pd.DataFrame:
    """Retrieve Tile IDs for Geospatial Observations from Satellite Data.

    This function associates each geospatial observation with a tile ID based on its
    geographic location, accommodating datasets with varying density across locations. By
    focusing on more densely populated areas, it enables more efficient resource usage and
    refined data analysis.

    The function assigns a tile ID to each observation, counts the occurrences within
    each tile, and retains only those tiles with a specified minimum count (`min_count`) of
    observations.

    Args:
        data: DataFrame containing geospatial observations with location coordinates.
        min_count: Minimum count of observations required per tile to retain.

    Returns:
        A subset of observations within tiles that meet or exceed the specified `min_count`.
    """
    mgrs_object = mgrs.MGRS()
    get_mgrs_tile_id = lambda row: mgrs_object.toMGRS(
        row["y"], row["x"], MGRSPrecision=0
    )
    data["mgrs_tile_id"] = data.apply(get_mgrs_tile_id, axis=1)
    tile_counts = data.groupby("mgrs_tile_id").size().sort_values(ascending=False)
    data = pd.merge(
        data, tile_counts.reset_index(name="counts"), how="left", on="mgrs_tile_id"
    )
    sub_data = data[data["counts"] >= min_count]
    assert not sub_data.empty, "No observation records left"
    return sub_data


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
