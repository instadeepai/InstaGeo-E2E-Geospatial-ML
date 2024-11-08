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

"""Utility Functions for Reading and Processing Harmonized Landsat Sentinel-2 Dataset."""


import re
from datetime import datetime, timedelta

import earthaccess
import pandas as pd
import rasterio
import xarray as xr
from rasterio.crs import CRS


def parse_date_from_entry(hls_tile_name: str) -> datetime | None:
    """Extracts the date from a HLS Tile Name.

    Args:
        hls_tile_name (str): Name of HLS tile.

    Returns:
        Parsed date or None.
    """
    match = re.search(r"\.(\d{7})T", hls_tile_name)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, "%Y%j")
    else:
        return None


def find_closest_tile(
    tile_queries: dict[str, tuple[str, list[str]]],
    tile_database: dict[str, list[str]],
    temporal_tolerance: int = 5,
) -> pd.DataFrame:
    """Find Closes HLS Tile.

    HLS dataset gets updated every 2 or 3 days and each tile is marked by the time of
    observation. This makes it difficult to derterministically find tiles for a given
    observation time. Rather we try to find a tile with observation time closest to our
    desired time.

    To do this, we create a database of tiles within a specific timeframe then we search
    for our desired tile within the database.

    Args:
        tile_queries (dict[str, tuple[str, list[str]]]): A dict with tile_query as key
            and a tuple of tile_id and a list  of dates on which the tile needs to be
            retrieved as value.
        tile_database (dict[str, list[str]]): A database mapping HLS tile_id to a list of
            available tiles within a pre-defined period of time
        temporal_tolerance: Number of days that can be tolerated for matching a closest
            tile in tile_databse.

    Returns:
        DataFrame containing the tile queries to the tile found.
    """
    query_results = {}
    for query_str, (tile_id, dates) in tile_queries.items():
        result = []
        if tile_id in tile_database:
            for date_str in dates:
                date = pd.to_datetime(date_str)
                year, day_of_year = date.year, date.day_of_year
                query_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
                closest_entry = None
                for entry in tile_database[tile_id]:
                    entry_date = parse_date_from_entry(entry)
                    if not entry_date:
                        continue
                    diff = abs((entry_date - query_date).days)
                    if (diff <= temporal_tolerance) and (diff >= 0):
                        closest_entry = entry
                        break
                result.append(closest_entry)
        query_results[query_str] = result
    query_results = pd.DataFrame(
        {"tile_queries": query_results.keys(), "hls_tiles": query_results.values()}
    )
    return query_results


def decode_fmask_value(value: xr.Dataset, position: int) -> xr.Dataset:
    """Decodes HLS v2.0 Fmask.

    The decoding strategy is described in Appendix A of the user manual
    (https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf).

    Arguments:
        value: Input xarray Dataset created from Fmask.tif
        position: Bit position to decode.

    Returns:
        Xarray dataset containing decoded bits.
    """
    quotient = value // (2**position)
    return quotient - ((quotient // 2) * 2)


def retrieve_hls_metadata(tile_info_df: pd.DataFrame) -> dict[str, list[str]]:
    """Retrieve HLS Tiles Metadata.

    Given a tile_id, start_date and end_date, this function fetches all the HLS granules
    available for this tile_id in this time window.

    Args:
        tile_info_df (pd.DataFrame): A dataframe containing tile_id, start_date and
            end_date in each row.

    Returns:
        A dictionary mapping tile_id to a list of available HLS granules.
    """
    granules_dict = {}
    for _, (
        tile_id,
        start_date,
        end_date,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    ) in tile_info_df.iterrows():
        results = earthaccess.search_data(
            short_name=["HLSL30", "HLSS30"],
            bounding_box=(lon_min, lat_min, lon_max, lat_max),
            temporal=(f"{start_date}T00:00:00", f"{end_date}T23:59:59"),
        )
        granules = pd.json_normalize(results)
        granules = granules[granules["meta.native-id"].str.contains(tile_id)]
        granules = list(granules["meta.native-id"])
        granules_dict[tile_id] = granules
    return granules_dict


def open_mf_tiff_dataset(
    band_files: dict[str, dict[str, str]], mask_cloud: bool, water_mask: bool
) -> tuple[xr.Dataset, CRS]:
    """Open multiple TIFF files as an xarray Dataset.

    Args:
        band_files (Dict[str, Dict[str, str]]): A dictionary mapping band names to file paths.
        mask_cloud (bool): Perform cloud masking.
        water_mask (bool): Perform water masking.

    Returns:
        (xr.Dataset, CRS): A tuple of xarray Dataset combining data from all the
            provided TIFF files and its CRS
    """
    band_paths = list(band_files["tiles"].values())
    bands_dataset = xr.open_mfdataset(
        band_paths,
        concat_dim="band",
        combine="nested",
    )
    mask_paths = list(band_files["fmasks"].values())
    mask_dataset = xr.open_mfdataset(
        mask_paths,
        concat_dim="band",
        combine="nested",
    )
    if water_mask:
        mask_water = decode_fmask_value(mask_dataset, 5)
        mask_water = mask_water.band_data.values.any(axis=0).astype(int)
        bands_dataset = bands_dataset.where(mask_water == 0)
    if mask_cloud:
        cloud_mask = decode_fmask_value(mask_dataset, 1)
        cloud_mask = cloud_mask.band_data.values.any(axis=0).astype(int)
        bands_dataset = bands_dataset.where(cloud_mask == 0)
    with rasterio.open(band_paths[0]) as src:
        crs = src.crs
    return bands_dataset, crs
