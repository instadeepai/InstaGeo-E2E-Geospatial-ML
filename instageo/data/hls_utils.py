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


import os
import re
from datetime import datetime, timedelta

import earthaccess
import pandas as pd
import rasterio
import xarray as xr
from absl import logging
from rasterio.crs import CRS

from instageo.data.data_pipeline import get_tile_info


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


def add_hls_granules(
    data: pd.DataFrame,
    num_steps: int = 3,
    temporal_step: int = 10,
    temporal_tolerance: int = 5,
) -> pd.DataFrame:
    """Add HLS Granules.

    Data contains tile_id and a series of date for which the tile is desired. This
    function takes the tile_id and the dates and finds the HLS tiles closest to the
    desired date with a tolearance of `temporal_tolerance`.

    Args:
        data (pd.DataFrame): A dataframe containing observations that fall within a
            dense tile.
        num_steps (int): Number of temporal steps into the past to fetch.
        temporal_step (int): Step size (in days) for creating temporal steps.
        temporal_tolerance (int): Tolerance (in days) for finding closest HLS tile.

    Returns:
        A dataframe containing a list of HLS granules. Each granule is a directory
        containing all the bands.
    """
    tiles_info, tile_queries = get_tile_info(
        data, num_steps=num_steps, temporal_step=temporal_step
    )
    tile_queries_str = [
        f"{tile_id}_{'_'.join(dates)}" for tile_id, dates in tile_queries
    ]
    data["tile_queries"] = tile_queries_str
    tile_database = retrieve_hls_metadata(tiles_info)
    tile_queries_dict = {k: v for k, v in zip(tile_queries_str, tile_queries)}
    query_result = find_closest_tile(
        tile_queries=tile_queries_dict,
        tile_database=tile_database,
        temporal_tolerance=temporal_tolerance,
    )
    data = pd.merge(data, query_result, how="left", on="tile_queries")
    return data


def create_hls_dataset(
    data_with_tiles: pd.DataFrame, outdir: str
) -> tuple[dict[str, dict[str, dict[str, str]]], set[str]]:
    """Creates HLS Dataset.

    A HLS dataset is a list of dictionary mapping band names to corresponding GeoTiff
    filepath. It is required for creating chips.

    Args:
        data_with_tiles (pd.DataFrame): A dataframe containing observations that fall
            within a dense tile. It also has `hls_tiles` column that contains a temporal
            series of HLS granules.
        outdir (str): Output directory where tiles could be downloaded to.

    Returns:
        A tuple containing HLS dataset and a list of tiles that needs to be downloaded.
    """
    data_with_tiles = data_with_tiles.drop_duplicates(subset=["hls_tiles"])
    data_with_tiles = data_with_tiles[
        data_with_tiles["hls_tiles"].apply(
            lambda granule_lst: all("HLS" in str(item) for item in granule_lst)
        )
    ]
    assert not data_with_tiles.empty, "No observation record with valid HLS tiles"
    hls_dataset = {}
    granules_to_download = []
    s30_bands = ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"]
    l30_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "Fmask"]
    for hls_tiles, obsv_date in zip(
        data_with_tiles["hls_tiles"], data_with_tiles["date"]
    ):
        band_id, band_path = [], []
        mask_id, mask_path = [], []
        for idx, tile in enumerate(hls_tiles):
            tile = tile.strip(".")
            if "HLS.S30" in tile:
                for band in s30_bands:
                    if band == "Fmask":
                        mask_id.append(f"{band}_{idx}")
                        mask_path.append(
                            os.path.join(outdir, "hls_tiles", f"{tile}.{band}.tif")
                        )
                    else:
                        band_id.append(f"{band}_{idx}")
                        band_path.append(
                            os.path.join(outdir, "hls_tiles", f"{tile}.{band}.tif")
                        )
                    granules_to_download.append(
                        f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/{tile}/{tile}.{band}.tif"  # noqa
                    )
            else:
                for band in l30_bands:
                    if band == "Fmask":
                        mask_id.append(f"{band}_{idx}")
                        mask_path.append(
                            os.path.join(outdir, "hls_tiles", f"{tile}.{band}.tif")
                        )
                    else:
                        band_id.append(f"{band}_{idx}")
                        band_path.append(
                            os.path.join(outdir, "hls_tiles", f"{tile}.{band}.tif")
                        )
                    granules_to_download.append(
                        f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/{tile}/{tile}.{band}.tif"  # noqa
                    )

        hls_dataset[f'{obsv_date.strftime("%Y-%m-%d")}_{tile.split(".")[2]}'] = {
            "tiles": {k: v for k, v in zip(band_id, band_path)},
            "fmasks": {k: v for k, v in zip(mask_id, mask_path)},
        }
    return hls_dataset, set(granules_to_download)


def parallel_download(urls: set[str], outdir: str, max_retries: int = 3) -> None:
    """Parallel Download.

    Wraps `download_tile` with multiprocessing.Pool for downloading multiple tiles in
    parallel.

    Args:
        urls: Tile urls to download.
        outdir: Directory to save downloaded tiles.
        max_retries: Number of times to retry downloading all tiles.

    Returns:
        None
    """
    num_cpus = os.cpu_count()
    earthaccess.login(persist=True)
    retries = 0
    complete = False
    while retries <= max_retries:
        temp_urls = [
            url
            for url in urls
            if not os.path.exists(os.path.join(outdir, url.split("/")[-1]))
        ]
        if not temp_urls:
            complete = True
            break
        earthaccess.download(temp_urls, local_path=outdir, threads=num_cpus)
        for filename in os.listdir(outdir):
            file_path = os.path.join(outdir, filename)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                if file_size < 1024:
                    os.remove(file_path)
        retries += 1
    if complete:
        logging.info("Successfully downloaded all granules")
    else:
        logging.warning(
            f"Couldn't download the following granules after {max_retries} retries:\n{urls}"  # noqa
        )
