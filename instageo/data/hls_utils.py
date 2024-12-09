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
import mgrs
import pandas as pd
from absl import logging
from shapely.geometry import box


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
    select_parsed_date = lambda item: item[1]
    for tile_id in tile_database:
        parsed_tiles_entries[tile_id] = list(
            filter(
                select_parsed_date,
                [
                    (entry, parse_date_from_entry(entry))
                    for entry in tile_database[tile_id]
                ],
            )
        )
    del tile_database

    query_results = {}
    for query_str, (tile_id, dates) in tile_queries.items():
        result = []
        if tile_id in parsed_tiles_entries:
            for date_str in dates:
                date = pd.to_datetime(date_str)
                year, day_of_year = date.year, date.day_of_year
                query_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
                closest_entry = None
                min_diff = timedelta.max.days

                index = bisect.bisect_left(
                    parsed_tiles_entries[tile_id], query_date, key=select_parsed_date
                )

                if index > 0:
                    entry, before_date = parsed_tiles_entries[tile_id][index - 1]
                    diff = abs((before_date - query_date).days)
                    if diff < min_diff:
                        closest_entry = entry
                        min_diff = diff

                if index < len(parsed_tiles_entries[tile_id]):
                    entry, after_date = parsed_tiles_entries[tile_id][index]
                    diff = abs((after_date - query_date).days)
                    if diff < min_diff:
                        closest_entry = entry
                        min_diff = diff

                if min_diff <= temporal_tolerance:
                    result.append(closest_entry)

        query_results[query_str] = result
    query_results = pd.DataFrame(
        {"tile_queries": query_results.keys(), "hls_tiles": query_results.values()}
    )
    return query_results


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

    def _make_valid_bbox(
        lon_min: float, lat_min: float, lon_max: float, lat_max: float
    ) -> tuple[float, float, float, float]:
        """Create a valid bounding box to search for HLS tiles.

        The purpose of this function is to still be able to extract data through
        earthaccess even given just a single observation in a tile (min_count = 1).
        When the number of observations in a tile is 1, or if we only have aligned
        observations, the lon_min, lat_min, lon_max, lat_max extracted from those
        won't produce a valid bounding box. Thus, we attempt to create a small buffer
        around the observation(s) to produce a valid bounding box.

        Args:
            lon_min (float): Minimum longitude
            lat_min (float): Minimum latitude
            lon_max (float): Maximum longitude
            lat_max (float): Maximum latitude

        Returns:
            A tuple of coordinates to use for a bounding box

        """
        epsilon = 5e-5
        if box(lon_min, lat_min, lon_max, lat_max).is_valid:
            return lon_min, lat_min, lon_max, lat_max
        else:
            return box(lon_min, lat_min, lon_max, lat_max).buffer(epsilon).bounds

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
            bounding_box=(_make_valid_bbox(lon_min, lat_min, lon_max, lat_max)),
            temporal=(f"{start_date}T00:00:00", f"{end_date}T23:59:59"),
        )
        granules = pd.json_normalize(results)
        assert not granules.empty, "No granules found"
        granules = granules[granules["meta.native-id"].str.contains(tile_id)]
        granules = list(granules["meta.native-id"])
        granules_dict[tile_id] = granules
    return granules_dict


def get_hls_tiles(data: pd.DataFrame, min_count: int = 100) -> pd.DataFrame:
    """Get HLS Tile ID for Each Observation.

    Observations are usually described by geolocation scattered across the globe. They are
    dense as well as sparse in various locations. In order to optimize resource usage, we
    subset the observations in dense locations.

    We first add the HLS tile ID for each observationa and count the number of
    observations in each tile. Then we retain the tiles with `min_count` observations.

    Args:
        data: Dataframe containing locust observations
        min_count: minimum count of locust observations per HLS tile.

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
        data (pd.DataFrame): A dataframe containing observations that fall within a
            dense tile.
        num_steps (int): Number of temporal steps into the past to fetch.
        temporal_step (int): Step size (in days) for creating temporal steps.
        temporal_tolerance (int): Tolerance (in days) for finding closest HLS tile.

    Returns:
        A dataframe containing a list of HLS granules. Each granule is a directory
        containing all the bands.
    """
    tiles_info, tile_queries = get_hls_tile_info(
        data,
        num_steps=num_steps,
        temporal_step=temporal_step,
        temporal_tolerance=temporal_tolerance,
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
    num_cpus = cpu_count()
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
