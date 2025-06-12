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
import logging
import os
import re
import time
from datetime import datetime, timedelta
from itertools import chain
from typing import Any, List

import backoff
import dask
import dask.delayed
import earthaccess
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import ratelimit
import rioxarray as rxr
import stackstac
import xarray as xr
from astral import LocationInfo
from astral.sun import sun
from pystac.item import Item
from pystac.item_collection import ItemCollection
from pystac_client import Client
from pystac_client.exceptions import APIError
from rasterio.crs import CRS
from shapely.geometry import box, shape
from shapely.ops import unary_union

from instageo.data import geo_utils, hls_utils
from instageo.data.data_pipeline import (
    BasePointsDataPipeline,
    BaseRasterDataPipeline,
    adjust_dims,
    apply_mask,
    create_segmentation_map,
    get_chip_coords,
    get_tile_info,
    mask_segmentation_map,
)
from instageo.data.settings import (
    DataPipelineSettings,
    GDALOptions,
    HLSAPISettings,
    HLSBandsSettings,
    HLSBlockSizes,
    NoDataValues,
)

# Create instances of the settings classes
NO_DATA_VALUES = NoDataValues()
BLOCKSIZE = HLSBlockSizes()
BANDS = HLSBandsSettings()
API = HLSAPISettings()
DATA_PIPELINE_SETTINGS = DataPipelineSettings()

client = Client.open(API.URL)


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
    temporal_tolerance_minutes: int = 0,
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
        temporal_tolerance (int): Number of days that can be tolerated for matching a closest
            tile in tile_databse.
        temporal_tolerance_minutes (int): Number of minutes to add to the temporal tolerance.


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

    # Convert temporal tolerance to total minutes
    total_tolerance_minutes = (
        temporal_tolerance * 24 * 60
    ) + temporal_tolerance_minutes

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
                    diff = abs((before_date - query_date).total_seconds() / 60)
                    if diff < min_diff:
                        closest_entry = entry
                        closest_entry_data_links = data_links
                        min_diff = diff

                if index < len(parsed_tiles_entries[tile_id]):
                    entry, data_links, after_date = parsed_tiles_entries[tile_id][index]
                    diff = abs((after_date - query_date).total_seconds() / 60)
                    if diff < min_diff:
                        closest_entry = entry
                        closest_entry_data_links = data_links
                        min_diff = diff

                result.append(
                    closest_entry if min_diff <= total_tolerance_minutes else None
                )
                result_data_links.append(
                    closest_entry_data_links
                    if min_diff <= total_tolerance_minutes
                    else None
                )

        query_results[query_str] = result, result_data_links

    query_results = pd.DataFrame.from_dict(
        query_results, orient="index", columns=["hls_tiles", "data_links"]
    )
    query_results.index.name = "tile_queries"
    return query_results


def decode_fmask_value(
    value: xr.Dataset | xr.DataArray, position: int
) -> xr.Dataset | xr.DataArray:
    """Decodes HLS v2.0 Fmask.

    Returns:
        Xarray dataset containing decoded bits.
    """
    quotient = value // (2**position)
    return quotient - ((quotient // 2) * 2)


def retrieve_hls_metadata(
    tile_info_df: pd.DataFrame,
    cloud_coverage: int = 10,
) -> dict[str, tuple[list[str], list[list[str]]]]:
    """Retrieve HLS Tiles Metadata.

    Given a tile_id, start_date and end_date, this function fetches all the HLS granules
    available for this tile_id in this time window.

    Args:
        tile_info_df (pd.DataFrame): A dataframe containing tile_id, start_date and
            end_date in each row.
        cloud_coverage (int): Maximum percentage of cloud cover allowed for a HLS tile.

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
        try:
            results = earthaccess.search_data(
                short_name=["HLSL30", "HLSS30"],
                bounding_box=(
                    geo_utils.make_valid_bbox(lon_min, lat_min, lon_max, lat_max)
                ),
                temporal=(f"{start_date}", f"{end_date}"),
                cloud_cover=(0, max(0.001, cloud_coverage)),
            )
        except RuntimeError:
            logging.info("Sleeping to retry after 30 minutes...")
            time.sleep(30 * 60)
            continue
        granules = pd.json_normalize(
            [result | {"data_links": result.data_links()} for result in results]
        )
        if granules.empty:
            continue
        granules = granules[granules["meta.native-id"].str.contains(tile_id)]
        granules, data_links = list(granules["meta.native-id"]), list(
            granules["data_links"]
        )
        granules_dict[tile_id] = granules, data_links
    return granules_dict


def open_mf_tiff_dataset(
    band_files: dict[str, Any], load_masks: bool
) -> tuple[xr.Dataset, xr.Dataset | None, CRS]:
    """Open multiple TIFF files as an xarray Dataset.

    Args:
        band_files (Dict[str, Dict[str, str]]): A dictionary mapping band names to file paths.
        load_masks (bool): Whether or not to load the masks files.

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
    mask_paths = list(band_files["fmasks"].values())
    mask_dataset = (
        xr.open_mfdataset(
            mask_paths,
            concat_dim="band",
            combine="nested",
        )
        if load_masks
        else None
    )
    with rasterio.open(band_paths[0]) as src:
        crs = src.crs
    return bands_dataset, mask_dataset, crs


@dask.delayed
def load_cog(url: str) -> xr.DataArray:
    """Load a COG file as an xarray DataArray.

    Args:
        url (str): COG url.

    Returns:
        xr.DataArray: An array exposing the data loaded from the COG
    """
    return rxr.open_rasterio(
        url,
        chunks=dict(band=1, x=BLOCKSIZE.X, y=BLOCKSIZE.Y),
        lock=False,
        mask_and_scale=False,  # Scaling will be applied manually
    )


def open_hls_cogs(
    bands_infos: dict[str, Any], load_masks: bool
) -> tuple[xr.DataArray, xr.DataArray | None, str]:
    """Open multiple COGs as an xarray DataArray.

    Args:
        bands_infos (dict[str, Any]): A dictionary containing data links for
        all bands and for all timesteps of interest.
        load_masks (bool): Whether or not to load the masks COGs.

    Returns:
        (xr.DataArray, xr.DataArray | None, str): A tuple of xarray Dataset combining
        data from all the COGs bands, (optionally) the COGs masks and the CRS used
    """
    cogs_urls = bands_infos["data_links"]
    # For each timestep, this will contain a list of links for the different bands
    # with the masks being at the last position

    bands_links = list(chain.from_iterable(urls[:-1] for urls in cogs_urls))
    masks_links = [urls[-1] for urls in cogs_urls]

    all_timesteps_bands = xr.concat(
        dask.compute(*[load_cog(link) for link in bands_links]), dim="band"
    )
    all_timesteps_bands.attrs["scale_factor"] = 1

    # only read masks if necessary
    all_timesteps_masks = (
        xr.concat(dask.compute(*[load_cog(link) for link in masks_links]), dim="band")
        if load_masks
        else None
    )
    return (
        all_timesteps_bands,
        all_timesteps_masks,
        all_timesteps_bands.spatial_ref.crs_wkt,
    )


def add_hls_granules(
    data: pd.DataFrame,
    num_steps: int = 3,
    temporal_step: int = 10,
    temporal_tolerance: int = 5,
    temporal_tolerance_minutes: int = 0,
    cloud_coverage: int = 10,
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
        temporal_tolerance_minutes (int): Number of minutes to add to the temporal
            tolerance.
        cloud_coverage (int): Maximum percentage of cloud cover allowed for a HLS tile.

    Returns:
        A dataframe containing a list of HLS granules. Each granule is a directory
        containing all the bands.
    """
    tiles_info, tile_queries = get_tile_info(
        data,
        num_steps=num_steps,
        temporal_step=temporal_step,
        temporal_tolerance=temporal_tolerance,
        temporal_tolerance_minutes=temporal_tolerance_minutes,
    )
    tile_queries_str = [
        f"{tile_id}_{'_'.join(dates)}" for tile_id, dates in tile_queries
    ]
    data["tile_queries"] = tile_queries_str
    tile_database = retrieve_hls_metadata(tiles_info, cloud_coverage=cloud_coverage)
    tile_queries_dict = {k: v for k, v in zip(tile_queries_str, tile_queries)}
    query_result = find_closest_tile(
        tile_queries=tile_queries_dict,
        tile_database=tile_database,
        temporal_tolerance=temporal_tolerance,
        temporal_tolerance_minutes=temporal_tolerance_minutes,
    )
    data = pd.merge(data, query_result, how="left", on="tile_queries")
    return data


def create_hls_dataset(
    data_with_tiles: pd.DataFrame, outdir: str
) -> tuple[dict[str, dict[str, Any]], set[str]]:
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
    date_format = (
        "%Y-%m-%dT%H:%M:%S" if "time" in data_with_tiles.columns else "%Y-%m-%d"
    )
    data_with_tiles = data_with_tiles[
        data_with_tiles["hls_tiles"].apply(
            lambda granule_lst: all("HLS" in str(item) for item in granule_lst)
        )
    ]
    assert not data_with_tiles.empty, "No observation record with valid HLS tiles"
    hls_dataset = {}
    data_links = []
    s30_bands = ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"]
    l30_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "Fmask"]

    for hls_tiles, download_links, obsv_date in zip(
        data_with_tiles["hls_tiles"],
        data_with_tiles["data_links"],
        data_with_tiles["date"],
    ):
        bands_paths = {}
        masks_paths = {}
        obsv_data_links = []
        tile = None
        for idx, (tile, tile_download_links) in enumerate(
            zip(hls_tiles, download_links)
        ):
            tile = tile.strip(".")
            bands_of_interest = s30_bands if "HLS.S30" in tile else l30_bands
            filtered_downloads_links = [
                next(link for link in tile_download_links if band + ".tif" in link)
                for band in bands_of_interest
            ]
            assert len(set(filtered_downloads_links)) == len(bands_of_interest)
            bands_paths.update(
                {
                    f"{band}_{idx}": os.path.join(
                        outdir, "hls_tiles", f"{tile}.{band}.tif"
                    )
                    for band in bands_of_interest[:-1]
                }
            )
            masks_paths.update(
                {
                    f"{bands_of_interest[-1]}_{idx}": os.path.join(
                        outdir, "hls_tiles", f"{tile}.{bands_of_interest[-1]}.tif"
                    )
                }
            )
            obsv_data_links.append(filtered_downloads_links)
        if tile:
            data_links.extend(obsv_data_links)
            hls_dataset[f'{obsv_date.strftime(date_format)}_{tile.split(".")[2]}'] = {
                "tiles": bands_paths,
                "fmasks": masks_paths,
                "data_links": obsv_data_links,
            }

    return hls_dataset, set(chain.from_iterable(data_links))


def parallel_download(
    dataset: dict[str, Any], outdir: str, max_retries: int = 3
) -> None:
    """Parallel Download.

    Wraps `download_tile` with multiprocessing.Pool for downloading multiple tiles in
    parallel.

    Args:
        dataset: A dataset mapping `key` to STAC Items.
        outdir: Directory to save downloaded tiles.
        max_retries: Number of times to retry downloading all tiles.

    Returns:
        None
    """
    num_cpus = os.cpu_count()
    earthaccess.login(persist=True)
    retries = 0
    complete = False

    urls = set()
    for _, dataset_entry in dataset.items():
        stac_items = dataset_entry["granules"]
        for item in stac_items:
            urls.update([item["assets"][band]["href"] for band in BANDS.ASSET])

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


def is_daytime(item: Item) -> bool:
    """Check if it's daytime.

    Checks if it's daytime using `datetime` and `bbox` properties of a HLS PySTAC item.

    It uses Astral to get the sunrise and sunset times for the item's centroid.

    Args:
        item (Item): HLS PySTAC item

    Returns:
        bool: A boolean that is True if it's daytime
    """
    item_datetime = pd.to_datetime(item.properties["datetime"])
    if not item_datetime:
        return False

    centroid = box(*item.bbox).centroid
    city = LocationInfo(
        "Unknown", "Unknown", "UTC", latitude=centroid.y, longitude=centroid.x
    )
    s = sun(city.observer, date=item_datetime)

    # Check if the item datetime is between sunrise and sunset
    return s["sunrise"] <= item_datetime <= s["sunset"]


def rename_hls_stac_items(item_collection: List[Item]) -> List[Item]:
    """Rename STAC Assets.

    HLSS30 and HLSL30 have different asset names. To make processing easier we map the asset names
    to a common naming convention defined in `BANDS.NAMEPLATE`.

    Arguments:
        item_collection (List[Item]): A list of PySTAC items.

    Returns:
        List of renamed PySTAC items.
    """
    for item in item_collection:
        for original_band, new_band in BANDS.NAMEPLATE[item.collection_id].items():
            item.assets[new_band] = item.assets.pop(original_band)
    return item_collection


def get_raster_tile_info(
    data: gpd.GeoDataFrame,
    num_steps: int = 3,
    temporal_step: int = 10,
    temporal_tolerance: int = 5,
    temporal_tolerance_minutes: int = 0,
) -> tuple[pd.DataFrame, list[tuple[str, list[str]]]]:
    """Get Raster Tile Info.

    Retrieves a summary of all tiles required for a given dataset. The summary contains
    the desired start and end date for each tile. Also retrieves a list of queries
    that can be used to retrieve the tiles for each observation in `data`.

    Args:
        data (pd.DataFrame): A dataframe containing observation records.
        num_steps (int): Number of temporal time steps
        temporal_step (int): Size of each temporal step.
        temporal_tolerance (int): Number of days used as offset for the
        start and end dates to search for each tile.
        temporal_tolerance_minutes (int): Number of minutes to add to the temporal
            tolerance.

    Returns:
        A `tile_info` dataframe and a list of `tile_queries`
    """
    push_max_date_to_end_of_day = "time" not in data.columns
    data = data[["mgrs_tile_id", "input_features_date", "geometry_4326"]].reset_index(
        drop=True
    )
    tile_queries = []
    tile_info: Any = []
    for _, (tile_id, date, polygon) in data.iterrows():
        history = []
        for i in range(num_steps):
            curr_date = pd.to_datetime(date) - pd.Timedelta(days=temporal_step * i)
            history.append(curr_date.strftime("%Y-%m-%dT%H:%M:%S"))
            tile_info.append([tile_id, curr_date, polygon])
        tile_queries.append((tile_id, history))
    tile_info = (
        gpd.GeoDataFrame(
            tile_info,
            columns=["tile_id", "date", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
        .groupby("tile_id")
        .agg(
            {
                "geometry": lambda geom: unary_union(geom),
                "date": lambda x: (x.min(), x.max()),
            }
        )
    ).reset_index()
    tile_info[["min_date", "max_date"]] = tile_info["date"].apply(pd.Series)
    tile_info[["lon_min", "lat_min", "lon_max", "lat_max"]] = tile_info[
        "geometry"
    ].apply(lambda geom: pd.Series(geom.bounds))

    # Convert temporal tolerance to total days including minutes
    total_temporal_tol = temporal_tolerance + (temporal_tolerance_minutes / (24 * 60))
    tile_info["min_date"] -= pd.Timedelta(days=total_temporal_tol)
    tile_info["max_date"] += pd.Timedelta(days=total_temporal_tol)
    tile_info["min_date"] = tile_info["min_date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    if push_max_date_to_end_of_day:
        tile_info["max_date"] = tile_info["max_date"].dt.strftime("%Y-%m-%dT23:59:59")
    else:
        tile_info["max_date"] = tile_info["max_date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    tile_info = tile_info[
        ["tile_id", "min_date", "max_date", "lon_min", "lon_max", "lat_min", "lat_max"]
    ]
    return tile_info, tile_queries


@ratelimit.limits(calls=DATA_PIPELINE_SETTINGS.METADATA_SEARCH_RATELIMIT, period=60)
@backoff.on_exception(
    backoff.expo,
    (APIError, RuntimeError),
    max_tries=5,
    max_time=300,  # 5 minutes max
    jitter=backoff.full_jitter,
)
def retrieve_hls_stac_metadata(
    client: Client,
    tile_info_df: pd.DataFrame,
    cloud_coverage: int = 10,
    daytime_only: bool = False,
) -> dict[str, List[Item]]:
    """Retrieve HLS items Metadata.

    Given a tile_id, start_date and end_date, this function searches all the items
    available for this tile_id in this time window. A pystac_client Client is used
    to query a STAC API.

    Args:
        client (pystac_client.Client): pystac_client client to use to perform the search.
        tile_info_df (pd.DataFrame): A dataframe containing tile_id, start_date and
            end_date in each row.
        cloud_coverage (int): Maximum percentage of cloud cover allowed for a HLS granule.
        daytime_only (bool): Whether to filter out night time granules.

    Returns:
        A dictionary mapping tile_id to a list of available HLS PySTAC items
          representing granules that contain all the polarizations needed.
    """
    items_dict: Any = {}

    for _, (
        tile_id,
        start_date,
        end_date,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    ) in tile_info_df.iterrows():
        try:
            search = client.search(
                collections=API.COLLECTIONS,
                datetime=f"{start_date}/{end_date}",
                bbox=geo_utils.make_valid_bbox(lon_min, lat_min, lon_max, lat_max),
                sortby=[{"field": "datetime"}],
                query={"eo:cloud_cover": {"lte": cloud_coverage}},
            )
            candidate_items = search.item_collection()
        except APIError as e:
            logging.warning(f"API Error for tile {tile_id}: {str(e)}")
            time.sleep(10 * 60)  # Wait 10 minutes before retrying
            continue

        if daytime_only:
            # select daytime observations only
            candidate_items = list(filter(is_daytime, candidate_items))
        if len(candidate_items) == 0:
            logging.warning(f"No items found for {tile_id}")
            continue
        # rename stac items to a common naming convention
        candidate_items = rename_hls_stac_items(candidate_items)

        items_dict[tile_id] = candidate_items
        time.sleep(10)  # Add a small delay between requests
    return items_dict


def dispatch_hls_candidate_items(
    tile_observations: gpd.GeoDataFrame,
    tile_candidate_items: List[Item],
) -> pd.DataFrame | None:
    """Dispatches appropriate HLS PySTAC items to each observation.

    A given observation will have a candidate item if it's geometry falls within the
    geometry of the granule.

    Args:
        tile_observations (pandas.DataFrame): DataFrame containing observations
            of the same tile.
        tile_candidate_items (List[Item]): List of candidate items for a given tile
    Returns:
        A DataFrame with observations and their containing granules (items)
        when possible.
    """
    hls_item_ids = [item.id for item in tile_candidate_items]
    candidate_items_gdf = gpd.GeoDataFrame.from_features(tile_candidate_items, crs=4326)
    candidate_items_gdf["hls_item_id"] = hls_item_ids
    candidate_items_gdf = candidate_items_gdf[["hls_item_id", "geometry"]]

    tile_observations = tile_observations.set_geometry("geometry_4326")

    matches = gpd.sjoin(
        tile_observations,
        candidate_items_gdf,
        predicate="within",
    )
    if matches.empty:
        return None
    else:
        tile_observations["hls_candidate_items"] = (
            matches.groupby(matches.index)
            .agg(
                {
                    "index_right": (
                        lambda indices: [tile_candidate_items[id] for id in indices]
                    )
                }
            )
            .reindex(tile_observations.index, fill_value=[])
        )
        return tile_observations


def find_closest_hls_items(
    obsv: pd.Series, temporal_tolerance: int = 3, temporal_tolerance_minutes: int = 0
) -> List[Item | None]:
    """Finds closest PySTAC items in time with least cloud coverage.

    Retrieves all PySTAC items within `temporal_tolerance` time window for a given observation and
    select the one with least cloud_coverage.

    Args:
        obsv (pandas.Series): Observation data containing the observation date
         and the HLS candidate items from which to pick.
        temporal_tolerance (int): Number of days that can be tolerated for matching
            granules from the candidate items.
        temporal_tolerance_minutes (int): Additional tolerance in minutes for finding
            closest HLS items.

    Returns:
        List[Item | None]: A list containing items if found within the temporal
           tolerance or None values.

    """
    dates = obsv.tile_queries[1]
    items = obsv.hls_candidate_items
    if not items:
        return [None] * len(dates)
    closest_items: list[Item] = []
    for date in dates:
        query_date = pd.to_datetime(date, utc=True)

        # Filter items within the temporal tolerance window
        candidate_items = [
            item
            for item in items
            if abs((item.datetime - query_date).total_seconds() / 60)
            <= (temporal_tolerance * 24 * 60 + temporal_tolerance_minutes)
        ]

        if not candidate_items:
            closest_items.append(None)
            continue

        # Select the item with the least cloud coverage
        selected_item = min(
            candidate_items, key=lambda item: item.properties["eo:cloud_cover"]
        )
        closest_items.append(selected_item)
    return closest_items


def find_best_hls_items(
    data: pd.DataFrame,
    tiles_database: dict[str, List[Item]],
    temporal_tolerance: int = 12,
    temporal_tolerance_minutes: int = 0,
) -> dict[str, pd.DataFrame]:
    """Finds best HLS PySTAC items for all observations when possible.

    For each observation, an attempt is made to retrieve the HLS granules
    that actually contain the observation. The `dispatch_candidate_items` function is
    thus used to ensure that the best HLS PySTAC items are not just
    the closest in terms of datetime.

    Args:
        data (pd.DataFrame): A dataframe containing observations that fall within a
            dense tile.
        src_crs (int): Source CRS of the observations.
        tiles_database(dict[str, List[Item]]): A dictionary mapping a tile ID to a list
          of available HLS PySTAC items.
        temporal_tolerance (int): Tolerance (in days) for finding closest HLS items.
        temporal_tolerance_minutes (int): Additional tolerance in minutes for finding
            closest HLS items.

    Returns:
        A dictionary mapping each MGRS tile ID to a DataFrame containing the observations
        that fall within that tile with their associated best PySTAC items representing
        the HLS granules.

    """
    best_hls_items = {}
    for tile_id in tiles_database:
        tile_obsvs = data[data["mgrs_tile_id"] == tile_id]

        # Before retrieving the temporally closest items, let's filter
        # the items by making sure only the ones with geometry in which
        # the observations fall are kept.
        tile_obsvs_with_hls_items = dispatch_hls_candidate_items(
            tile_obsvs, tiles_database[tile_id]
        )
        if tile_obsvs_with_hls_items is None:
            continue
        # We can then retrieve the item with the least cloud coverage within the temporal tolerance.
        tile_obsvs_with_hls_items["hls_items"] = tile_obsvs_with_hls_items.apply(
            lambda obsv: find_closest_hls_items(
                obsv, temporal_tolerance, temporal_tolerance_minutes
            ),
            axis=1,
        )

        best_hls_items[tile_id] = tile_obsvs_with_hls_items.drop(
            columns=["hls_candidate_items"]
        )
    return best_hls_items


def add_hls_stac_items(
    client: Client,
    data: pd.DataFrame,
    num_steps: int = 3,
    temporal_step: int = 10,
    temporal_tolerance: int = 12,
    temporal_tolerance_minutes: int = 0,
    cloud_coverage: int = 10,
    daytime_only: bool = False,
) -> dict[str, pd.DataFrame]:
    """Searches and adds HLS Granules.

    Data contains tile_id and a series of date for which the tile is desired. This
    function takes the tile_id and the dates and finds the HLS granules with the least cloud
    coverage closest to the desired date with a tolerance of `temporal_tolerance`.

    Args:
        client (pystac_client.Client): pystac_client client to use to perform the search.
        data (pd.DataFrame): A dataframe containing observations that fall within a
            dense tile.
        num_steps (int): Number of temporal steps into the past to fetch.
        temporal_step (int): Step size (in days) for creating temporal steps.
        temporal_tolerance (int): Tolerance (in days) for finding closest HLS items.
        temporal_tolerance_minutes (int): Additional tolerance in minutes for finding
            closest HLS items.
        cloud_coverage (int): Maximum percentage of cloud coverage to be tolerated for a granule.
        daytime_only (bool): Flag to determine whether to filter out night time granules.

    Returns:
        A dictionary mapping each MGRS tile ID to a DataFrame containing the observations
          that fall within that tile with their associated PySTAC items representing granules.
    """
    if "input_features_date" not in data.columns:
        data = data.rename(columns={"date": "input_features_date"})
    tiles_info, tile_queries = get_raster_tile_info(
        data,
        num_steps=num_steps,
        temporal_step=temporal_step,
        temporal_tolerance=temporal_tolerance,
        temporal_tolerance_minutes=temporal_tolerance_minutes,
    )
    data["tile_queries"] = tile_queries
    tiles_database = retrieve_hls_stac_metadata(
        client, tiles_info, cloud_coverage=cloud_coverage, daytime_only=daytime_only
    )
    best_items = find_best_hls_items(
        data, tiles_database, temporal_tolerance, temporal_tolerance_minutes
    )
    return best_items


def is_valid_dataset_entry(obsv: pd.Series) -> bool:
    """Checks HLS granules validity for a given observation.

    The granules will be added to the dataset if they are all non
    null and unique for all timesteps.

    Args:
        obsv (pandas.Series): Observation data for which to assess the validity

    Returns:
        True if the granules are unique and non null.
    """
    if any(granule is None for granule in obsv["hls_granules"]) or (
        len(obsv["hls_granules"]) != len(set(obsv["hls_granules"]))
    ):
        return False
    return True


def create_hls_records_with_items(
    best_items: dict[str, pd.DataFrame],
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Creates the HLS dataset from granules found.

    Args:
        best_items: A dictionary mapping each MGRS tile ID to
        a DataFrame containing the observations that fall within that tile with their
        associated PySTAC items representing granules.

    Returns:
        A tuple containing a geopandas dataframe with `stac_items_str` column and a dictionary
        mapping `stac_items_str` to the to PySTAC items representing the granules.
    """
    records_with_items = []
    hls_dataset = {}
    for tile_id in best_items:
        obsvs = best_items[tile_id]
        obsvs["hls_granules"] = obsvs.apply(
            lambda obsv: [
                item.id if isinstance(item, Item) else None
                for item in obsv["hls_items"]
            ],
            axis=1,
        )
        obsvs = obsvs[obsvs.apply(is_valid_dataset_entry, axis=1)]
        obsvs["stac_items_str"] = obsvs["hls_granules"].apply(lambda x: "_".join(x))
        for _, obsv in obsvs.drop_duplicates(subset=["hls_granules"]).iterrows():
            hls_dataset[obsv["stac_items_str"]] = {
                "granules": [item.to_dict() for item in obsv["hls_items"]]
            }
        obsvs = obsvs.drop(
            ["geometry_4326", "hls_items", "tile_queries", "hls_granules"], axis=1
        )
        records_with_items.append(obsvs)
    filtered_records = pd.concat(records_with_items, ignore_index=True).set_geometry(
        "geometry"
    )
    return filtered_records, hls_dataset


class HLSRasterPipeline(BaseRasterDataPipeline):
    """HLS Raster Data Pipeline."""

    def setup(self) -> None:
        """Setup for environment to be used by Dask workers.

        Configures relevant GDAL options for reading COGs
        """
        earthaccess.login(persist=True)
        env = rasterio.Env(**GDALOptions().model_dump())
        env.__enter__()

    @ratelimit.limits(calls=DATA_PIPELINE_SETTINGS.COG_DOWNLOAD_RATELIMIT, period=60)
    @backoff.on_exception(
        backoff.expo,
        (rasterio.errors.RasterioIOError, Exception),
        max_tries=5,
        max_time=300,  # 5 minutes max
        jitter=backoff.full_jitter,
    )
    def load_data(
        self, tile_dict: dict[str, Any]
    ) -> tuple[xr.DataArray, xr.DataArray, str]:
        """See parent class. Load Granules."""
        dsb, dsm, crs = open_hls_stac_items(
            tile_dict["granules"],
            self.src_crs,
            self.spatial_resolution,
            load_masks=True,
            fill_value=NO_DATA_VALUES.HLS,
        )
        return dsb, dsm, crs

    def process_row(
        self,
        row_dict: dict[str, Any],
        flags_dict: dict[str, Any],
        tile_dict: dict[str, Any],
    ) -> None | tuple[str, str]:
        """See parent class. Process a single row."""
        label_filename = f"{os.path.splitext(row_dict['label_filename'])[0]}_{row_dict['mgrs_tile_id']}"  # noqa
        chip_filename = label_filename.replace("mask", "merged").replace(
            "label", "chip"
        )

        chip_path = os.path.join(self.output_directory, "chips", f"{chip_filename}.tif")
        label_path = os.path.join(
            self.output_directory, "seg_maps", f"{label_filename}.tif"
        )
        if os.path.exists(chip_path) and os.path.exists(label_path):
            logging.info(f"Skipping {chip_path} because it's already created")
            return chip_path, label_path
        try:
            dsb, dsm, _ = self.load_data(tile_dict)
            geometry = shape(row_dict["geometry"])

            # Process chip
            chip = geo_utils.slice_xr_dataset(dsb, geometry, chip_size=self.chip_size)
            seg_map = xr.open_dataarray(
                os.path.join(self.raster_path, row_dict["label_filename"])
            )

            if dsm is not None:
                chip_mask = geo_utils.slice_xr_dataset(
                    dsm, geometry, chip_size=self.chip_size
                )
                chip = apply_mask(
                    chip=chip,
                    mask=chip_mask,
                    no_data_value=0,
                    mask_decoder=hls_utils.decode_fmask_value,
                    data_source="HLS",
                    mask_types=self.mask_types,
                    masking_strategy=self.masking_strategy,
                )

            if (
                chip is not None
                and chip.sizes["x"] == seg_map.sizes["x"]
                and chip.sizes["y"] == seg_map.sizes["y"]
            ):
                # Overrides the chip coordinates to match the segmentation map.
                seg_map, chip = xr.align(
                    seg_map, chip, join="override", exclude=["band"]
                )
                # Clip values to valid HLS range (0-10000)
                chip = chip.clip(min=0, max=10000)

                if self.qa_check:
                    if chip.where(chip != NO_DATA_VALUES.HLS).count().values == 0:
                        logging.warning(f"Skipping {chip_filename} due to cloud")
                        return None
                    seg_map = mask_segmentation_map(
                        chip, seg_map, NO_DATA_VALUES.HLS, self.masking_strategy
                    )
                    if (
                        seg_map.where(seg_map != NO_DATA_VALUES.SEG_MAP).count().values
                        == 0
                    ):
                        logging.warning(f"Skipping {label_filename} due to empty label")
                        return None
                seg_map = seg_map.where(
                    ~np.isnan(seg_map), NO_DATA_VALUES.SEG_MAP
                ).astype(np.uint8 if self.task_type == "seg" else np.float32)
                chip = chip.where(~np.isnan(chip), 0).astype(np.uint16)

                seg_map.squeeze().rio.to_raster(label_path)
                chip.squeeze().rio.to_raster(chip_path)
                return chip_path, label_path
            else:
                logging.warning(f"Skipping {label_filename} due to invalid shapes")
            return None
        except Exception as e:
            logging.error(f"Error processing row {row_dict}: {str(e)}")
            return None


class HLSPointsPipeline(BasePointsDataPipeline):
    """HLS Raster Data Pipeline."""

    def setup(self) -> None:
        """Setup for environment to be used by Dask workers.

        Configures relevant GDAL options for reading COGs
        """
        earthaccess.login(persist=True)
        env = rasterio.Env(**GDALOptions().model_dump())
        env.__enter__()

    @ratelimit.limits(calls=DATA_PIPELINE_SETTINGS.COG_DOWNLOAD_RATELIMIT, period=60)
    @backoff.on_exception(
        backoff.expo,
        (rasterio.errors.RasterioIOError, Exception),
        max_tries=5,
        max_time=300,  # 5 minutes max
        jitter=backoff.full_jitter,
    )
    def load_data(
        self, tile_dict: dict[str, Any]
    ) -> tuple[xr.Dataset, xr.Dataset, str]:
        """See parent class. Load Granules."""
        dsb, dsm, crs = open_hls_stac_items(
            tile_dict["granules"],
            self.src_crs,
            self.spatial_resolution,
            load_masks=True,
        )
        return dsb, dsm, crs

    def process_tile(
        self,
        obsv_records: gpd.GeoDataFrame,
        flags_dict: dict[str, Any],
        tile_dict: dict[str, Any],
        batch_size: int,
    ) -> tuple[list[str], list[str]]:
        """Processes a single tile.

        Arguments:
            obsv_records: Observation records dataframe.
            flags_dict: Dictionary mapping all arguments and values needed to process the row.
            tile_dict: Input to `self.load_data`, which contains the granules to load.
            batch_size: Number of records to process at a time.
        Returns: tuple of the chip and label filenames lists.
        """
        tile_name_splits = tile_dict["granules"][0]["id"].split(".")
        tile_id = f"{tile_name_splits[1]}_{tile_name_splits[2]}_{tile_name_splits[3]}"
        stac_items_str = obsv_records.iloc[0]["stac_items_str"]
        chip_paths = []
        label_paths = []

        try:
            date_id = obsv_records.iloc[0]["date"].strftime("%Y%m%d")
            dsb, dsm, crs = self.load_data(tile_dict)
            n_chips_x = dsb.sizes["x"] // self.chip_size
            n_chips_y = dsb.sizes["y"] // self.chip_size
            chip_coords = get_chip_coords(obsv_records, dsb, self.chip_size)

            # Process chips in smaller batches to avoid overwhelming the API
            for i in range(0, len(chip_coords), batch_size):
                batch_coords = chip_coords[i : i + batch_size]
                chips, masks, seg_maps_temp_filenames, chips_temp_filenames = (
                    [],
                    [],
                    [],
                    [],
                )

                for x, y in batch_coords:
                    # TODO: handle potential partially out of bound chips
                    if (x >= n_chips_x) or (y >= n_chips_y):
                        continue

                    chip_id = f"{date_id}_{tile_id}_{x}_{y}"
                    chip_name = f"chip_{chip_id}.tif"
                    seg_map_name = f"seg_map_{chip_id}.tif"

                    chip_filename = os.path.join(
                        self.output_directory, "chips", chip_name
                    )
                    chips_temp_filenames.append(chip_filename)
                    seg_map_filename = os.path.join(
                        self.output_directory, "seg_maps", seg_map_name
                    )
                    seg_maps_temp_filenames.append(seg_map_filename)
                    if os.path.exists(chip_filename) or os.path.exists(
                        seg_map_filename
                    ):
                        logging.info(
                            f"Skipping {chip_filename} because it's already created"
                        )
                        continue

                    chip = dsb.isel(
                        x=slice(x * self.chip_size, (x + 1) * self.chip_size),
                        y=slice(y * self.chip_size, (y + 1) * self.chip_size),
                    )
                    chips.append(chip)

                    if dsm is not None:
                        chip_mask = dsm.isel(
                            x=slice(x * self.chip_size, (x + 1) * self.chip_size),
                            y=slice(y * self.chip_size, (y + 1) * self.chip_size),
                        )
                        masks.append(chip_mask)
                    else:
                        masks.append(None)

                # Process the batch
                try:
                    # Compute chips and masks locally before processing
                    chips = [chip.compute() for chip in chips]
                    masks = (
                        [mask.compute() for mask in masks] if dsm is not None else masks
                    )

                    for chip, mask, chip_filename, seg_map_filename in zip(
                        chips, masks, chips_temp_filenames, seg_maps_temp_filenames
                    ):
                        if mask is not None:
                            chip = apply_mask(
                                chip=chip,
                                mask=mask,
                                no_data_value=NO_DATA_VALUES.HLS,
                                mask_decoder=decode_fmask_value,
                                data_source="HLS",
                                mask_types=self.mask_types,
                                masking_strategy=self.masking_strategy,
                            )

                        if chip.where(chip != NO_DATA_VALUES.HLS).count().values == 0:
                            logging.warning(f"Skipping {chip_filename} due to cloud")
                            continue

                        seg_map = create_segmentation_map(
                            chip, obsv_records, self.window_size
                        )
                        seg_map = mask_segmentation_map(
                            chip,
                            seg_map,
                            NO_DATA_VALUES.HLS,
                            self.masking_strategy,
                        )

                        if (
                            seg_map.where(seg_map != NO_DATA_VALUES.SEG_MAP)
                            .count()
                            .values
                            == 0
                        ):
                            logging.warning(
                                f"Skipping {seg_map_filename} due to empty label"
                            )
                            continue

                        label_paths.append(seg_map_filename)
                        chip_paths.append(chip_filename)
                        seg_map.rio.to_raster(seg_map_filename)
                        chip.rio.to_raster(chip_filename)

                except Exception as e:
                    logging.error(f"Error processing batch: {str(e)}")
                    continue

                # Add a delay between batches to avoid rate limiting
                time.sleep(5)

        except rasterio.errors.RasterioIOError as e:
            logging.error(
                f"Error {e} when reading dataset containing: {stac_items_str}"
            )
        except Exception as e:
            logging.error(f"Error {e} when processing {stac_items_str}")

        return chip_paths, label_paths


def open_hls_stac_items(
    tile_dict: dict[str, Any],
    epsg: int,
    resolution: float,
    load_masks: bool = False,
    fill_value: int = 0,
) -> tuple[xr.DataArray, xr.DataArray | None, str]:
    """Opens multiple HLS STAC Items as an xarray DataArray from given granules `tile_dict`.

    Args:
        tile_dict (dict[str, Any]): A dictionary containing granules IDs to retrieve
        for all timesteps of interest.
        epsg (int): CRS EPSG code.
        resolution (float): Spatial resolution in the specified CRS.
        load_masks (bool): Whether or not to load the masks COGs.
        fill_value (int): Fill value for the data array.

    Returns:
        (xr.DataArray, xr.DataArray | None, str): A tuple of xarray DataArray combining
        data from all the COGs bands of interest, (optionally) the COGs masks and the
        CRS used.
    """
    # Load the bands for all timesteps and stack them in a data array
    assets_to_load = BANDS.ASSET + ["Fmask"] if load_masks else BANDS.ASSET

    # Convert items to plain dicts before stacking to avoid STAC catalog resolution
    plain_items = [item.to_dict() for item in ItemCollection(tile_dict)]

    stacked_items = stackstac.stack(
        plain_items,
        assets=assets_to_load,
        chunksize=(BLOCKSIZE.X, BLOCKSIZE.Y),
        properties=False,
        rescale=False,
        fill_value=fill_value,
        epsg=epsg,
        resolution=resolution,
    )

    bands = adjust_dims(stacked_items.sel(band=BANDS.ASSET))
    masks = adjust_dims(stacked_items.sel(band=["Fmask"])) if load_masks else None

    bands = bands.astype(np.uint16)
    bands.attrs["scale_factor"] = 1

    return bands, masks, bands.crs
