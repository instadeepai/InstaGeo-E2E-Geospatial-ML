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

import bisect
import os
import re
from datetime import datetime, timedelta
from multiprocessing import cpu_count
from typing import Any

import earthaccess
import pandas as pd
from absl import logging
from shapely.geometry import box

from instageo.data.geo_utils import get_tile_info

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
    tile_database: dict[str, tuple[list[str], list[list[str]]]],
    temporal_tolerance: int = 5,
) -> pd.DataFrame:
    """Find Closes HLS Tile.

    HLS dataset gets updated every 2 or 3 days and each tile is marked by the time of
    observation. This makes it difficult to deterministically find tiles for a given
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
    # parse dates only once at the beginning for every tile_id
    parsed_tiles_entries: Any = {}
    select_parsed_date = lambda item: item[2]
    for tile_id in tile_database:
        parsed_tiles_entries[tile_id] = list(
            filter(
                select_parsed_date,
                [
                    (entry, data_links, parse_date_from_entry(entry))
                    for entry, data_links in zip(*tile_database[tile_id])
                ],
            )
        )
    del tile_database

    query_results: Any = {}
    for query_str, (tile_id, dates) in tile_queries.items():
        result = []
        result_data_links = []
        if tile_id in parsed_tiles_entries:
            for date_str in dates:
                date = pd.to_datetime(date_str)
                year, day_of_year = date.year, date.day_of_year
                query_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
                closest_entry = None
                closest_entry_data_links = []
                min_diff = timedelta.max.days

                index = bisect.bisect_left(
                    parsed_tiles_entries[tile_id], query_date, key=select_parsed_date
                )

                if index > 0:
                    entry, data_links, before_date = parsed_tiles_entries[tile_id][
                        index - 1
                    ]
                    diff = abs((before_date - query_date).days)
                    if diff < min_diff:
                        closest_entry = entry
                        closest_entry_data_links = data_links
                        min_diff = diff

                if index < len(parsed_tiles_entries[tile_id]):
                    entry, data_links, after_date = parsed_tiles_entries[tile_id][index]
                    diff = abs((after_date - query_date).days)
                    if diff < min_diff:
                        closest_entry = entry
                        closest_entry_data_links = data_links
                        min_diff = diff

                result.append(closest_entry if min_diff <= temporal_tolerance else None)
                result_data_links.append(
                    closest_entry_data_links if min_diff <= temporal_tolerance else None
                )

        query_results[query_str] = result, result_data_links

    query_results = pd.DataFrame.from_dict(
        query_results, orient="index", columns=["hls_tiles", "data_links"]
    )
    query_results.index.name = "tile_queries"
    return query_results


def retrieve_hls_metadata(
    tile_info_df: pd.DataFrame,
) -> dict[str, tuple[list[str], list[list[str]]]]:
    """Retrieve HLS Tiles Metadata.

    Given a tile_id, start_date and end_date, this function fetches all the HLS granules
    available for this tile_id in this time window.

    Args:
        tile_info_df (pd.DataFrame): A dataframe containing tile_id, start_date and
            end_date in each row.

    Returns:
        A dictionary mapping tile_id to a list of available HLS granules.
    """
    granules_dict: Any = {}
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
            bounding_box=(make_valid_bbox(lon_min, lat_min, lon_max, lat_max)),
            temporal=(f"{start_date}T00:00:00", f"{end_date}T23:59:59"),
        )
        granules = pd.json_normalize(
            [result | {"data_links": result.data_links()} for result in results]
        )
        assert not granules.empty, "No granules found"
        granules = granules[granules["meta.native-id"].str.contains(tile_id)]
        granules, data_links = list(granules["meta.native-id"]), list(
            granules["data_links"]
        )
        granules_dict[tile_id] = granules, data_links
    return granules_dict


def get_hls_tiles(data: pd.DataFrame, min_count: int = 100) -> pd.DataFrame:
    """Retrieve Harmonized Landsat Sentinel (HLS) Tile IDs for Geospatial Observations.

    Observations are usually described by geolocation scattered across the globe. They are
    dense as well as sparse in various locations. In order to optimize resource usage, we
    subset the observations in dense locations.

    The function assigns an HLS tile ID to each observation, counts the occurrences within
    each tile, and retains only those tiles with a specified minimum count (`min_count`) of
    observations.

    Args:
        data: DataFrame containing geospatial observations with location coordinates.
        min_count: Minimum count of observations required per HLS tile to retain.

    Returns:
        Subset of observations where there are at least `min_count` observations per tile

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


def get_hls_tile_info(
    data: pd.DataFrame,
    num_steps: int = 3,
    temporal_step: int = 10,
    temporal_tolerance: int = 5,
) -> tuple[pd.DataFrame, list[tuple[str, list[str]]]]:
    """Get HLS Tile Info.

    Retrieves a summary of all tiles required for a given dataset. The summary contains
    the desired start and end date for each HLS tile. Also retrieves a list of queries
    that can be used to retrieve the tiles for each observation in `data`.

    Args:
        data (pd.DataFrame): A dataframe containing observation records.
        num_steps (int): Number of temporal time steps
        temporal_step (int): Size of each temporal step.
        temporal_tolerance (int): Number of days used as offset for the
        start and end dates to search for each HLS tile.

    Returns:
        A `tile_info` dataframe and a list of `tile_queries`
    """
    data = data[["mgrs_tile_id", "input_features_date", "x", "y"]].reset_index(
        drop=True
    )
    tile_queries = []
    tile_info: Any = []
    for _, (tile_id, date, lon, lat) in data.iterrows():
        history = []
        for i in range(num_steps):
            curr_date = date - pd.Timedelta(days=temporal_step * i)
            history.append(curr_date.strftime("%Y-%m-%d"))
            tile_info.append([tile_id, curr_date, lon, lat])
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
    tile_info["min_date"] -= pd.Timedelta(days=temporal_tolerance)
    tile_info["max_date"] += pd.Timedelta(days=temporal_tolerance)
    tile_info["min_date"] = tile_info["min_date"].dt.strftime("%Y-%m-%d")
    tile_info["max_date"] = tile_info["max_date"].dt.strftime("%Y-%m-%d")
    return tile_info, tile_queries


def add_hls_granules(
    data: pd.DataFrame,
    num_steps: int = 3,
    temporal_step: int = 10,
    temporal_tolerance: int = 5,
) -> pd.DataFrame:
    """Add HLS Granules.

    Data contains tile_id and a series of date for which the tile is desired. This
    function takes the tile_id and the dates and finds the HLS tiles closest to the
    desired date with a tolerance of `temporal_tolerance`.

    Args:
        band_files (Dict[str, Dict[str, str]]): A dictionary mapping band names to file paths.
        mask_cloud (bool): Perform cloud masking.
        water_mask (bool): Perform water masking.

    Returns:
        (xr.Dataset, CRS): A tuple of xarray Dataset combining data from all the
            provided TIFF files and its CRS
    """
    tiles_info, tile_queries = get_hls_tile_info(
        data,
        num_steps=num_steps,
        temporal_step=temporal_step,
        temporal_tolerance=temporal_tolerance,
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
    desired date with a tolerance of `temporal_tolerance`.

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
