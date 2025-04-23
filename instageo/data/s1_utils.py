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

"""Utility Functions for generating Sentinel-1 chips."""

import bisect
from collections import Counter
from datetime import timedelta
from typing import Any, List

import geopandas as gpd
import pandas as pd
import planetary_computer
import stackstac
import xarray as xr
from absl import logging
from pystac.item import Item
from pystac.item_collection import ItemCollection
from pystac_client import Client

from instageo.data.data_pipeline import NO_DATA_VALUES, adjust_dims, get_tile_info
from instageo.data.geo_utils import make_valid_bbox

BLOCKSIZE = 512
COLLECTION = "sentinel-1-rtc"
POLARIZATIONS = ["vv", "vh"]


def retrieve_s1_metadata(
    client: Client,
    tile_info_df: pd.DataFrame,
) -> dict[str, List[Item]]:
    """Retrieve S1 items Metadata.

    Given a tile_id, start_date and end_date, this function searches all the items
    available for this tile_id in this time window. A pystac_client Client is used
    to query a STAC API.

    Args:
        client (pystac_client.Client): pystac_client client to use to perform the search.
        tile_info_df (pd.DataFrame): A dataframe containing tile_id, start_date and
            end_date in each row.

    Returns:
        A dictionary mapping tile_id to a list of available Sentinel-1 PySTAC items
          representing granules that contain all the polarizations needed.
    """
    items_dict: Any = {}
    contains_polarizations = lambda item: all(
        pol in item.assets for pol in POLARIZATIONS
    )

    for _, (
        tile_id,
        start_date,
        end_date,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    ) in tile_info_df.iterrows():
        search = client.search(
            collections=[COLLECTION],
            datetime=f"{start_date}/{end_date}",
            bbox=make_valid_bbox(lon_min, lat_min, lon_max, lat_max),
            sortby=[{"field": "datetime", "direction": "asc"}],
        )
        candidate_items = search.item_collection()
        candidate_items = list(filter(contains_polarizations, candidate_items))
        if len(candidate_items) == 0:
            logging.warning(f"No items found for {tile_id}")
            continue
        items_dict[tile_id] = candidate_items
    return items_dict


def dispatch_candidate_items(
    tile_observations: pd.DataFrame,
    src_crs: int,
    tile_candidate_items: List[Item],
) -> pd.DataFrame:
    """Dispatches appropriate Sentinel-1 PySTAC items to each observation.

    A given observation will have a candidate item if it falls within the
    geometry of the granule.

    Args:
        tile_observations (pandas.DataFrame): DataFrame containing observations
            of the same tile.
        src_crs (int): Source CRS of the observations.
        tile_candidate_items (List[Item]): List of candidate items for a given tile
    Returns:
        A DataFrame with observations and their containing granules (items)
        when possible.
    """
    s1_item_ids = [item.id for item in tile_candidate_items]
    candidate_items_gdf = gpd.GeoDataFrame.from_features(tile_candidate_items, crs=4326)
    candidate_items_gdf["s1_item_id"] = s1_item_ids
    candidate_items_gdf = candidate_items_gdf[["s1_item_id", "geometry"]]

    tile_observations = gpd.GeoDataFrame(
        tile_observations,
        geometry=gpd.points_from_xy(
            x=tile_observations.x,
            y=tile_observations.y,
        ),
        crs=src_crs,
    ).to_crs(4326)

    matches = gpd.sjoin(
        tile_observations,
        candidate_items_gdf,
        predicate="within",
    )
    tile_observations["s1_candidate_items"] = (
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
    return tile_observations.drop(columns=["geometry"])


def find_best_s1_items(
    data: pd.DataFrame,
    src_crs: int,
    tiles_database: dict[str, List[Item]],
    temporal_tolerance: int = 12,
) -> dict[str, pd.DataFrame]:
    """Finds best Sentinel-1 PySTAC items for all observations when possible.

    For each observation, an attempt is made to retrieve the Sentinel-1 granules
    that actually contain the observation. The `dispatch_candidate_items` function is
    thus used to ensure that the best Sentinel-1 PySTAC items are not just
    the closest in terms of datetime.

    Args:
        data (pd.DataFrame): A dataframe containing observations that fall within a
            dense tile.
        src_crs (int): Source CRS of the observations.
        tiles_database(dict[str, List[Item]]): A dictionary mapping a tile ID to a list
          of available Sentinel-1 PySTAC items.
        temporal_tolerance (int): Tolerance (in days) for finding closest Sentinel-1 items.

    Returns:
        A dictionary mapping each MGRS tile ID to a DataFrame containing the observations
        that fall within that tile with their associated best PySTAC items representing
        the Sentinel-1 granules.

    """
    best_s1_items = {}
    for tile_id in tiles_database:
        tile_obsvs = data[data["mgrs_tile_id"] == tile_id]

        # Before retrieving the temporally closest items, let's filter
        # the items by making sure only the ones with geometry in which
        # the observations fall are kept.
        tile_obsvs_with_s1_items = dispatch_candidate_items(
            tile_obsvs, src_crs, tiles_database[tile_id]
        )

        # We can then retrieve the closest item in time.
        tile_obsvs_with_s1_items["s1_items"] = tile_obsvs_with_s1_items.apply(
            lambda obsv: find_closest_items(obsv, temporal_tolerance), axis=1
        )

        best_s1_items[tile_id] = tile_obsvs_with_s1_items.drop(
            columns=["s1_candidate_items"]
        )
    return best_s1_items


def find_closest_items(
    obsv: pd.Series, temporal_tolerance: int = 12
) -> List[Item | None]:
    """Finds temporally closest Sentinel-1 PySTAC items for a given observation.

    Args:
        obsv (pandas.Series): Observation data containing the observation date
         and the Sentinel-1 candidate items from which to pick.
        temporal_tolerance (int): Number of days that can be tolerated for matching
            granules from the candidate items.

    Returns:
        List[Item | None]: A list containing items if found within the temporal
           tolerance or None values.

    """
    dates = obsv.tile_queries[1]
    items = obsv.s1_candidate_items
    if not items:
        return [None] * len(dates)
    closest_items = []
    for date in dates:
        query_date = pd.to_datetime(date, utc=True)
        closest_item = None
        min_diff = timedelta.max.days

        index = bisect.bisect_left(items, query_date, key=lambda item: item.datetime)

        # Let's select when possible the closest item anterior to the observation date.
        if index > 0:
            diff = abs((items[index - 1].datetime - query_date).days)
            if diff < min_diff:
                closest_item = items[index - 1]
                min_diff = diff

        # Let's select when possible the closest item posterior or corresponding to the
        #  observation date. That item is kept as closest, if it is closer to the
        # observation date, when compared to the potential item selected in the previous step.
        if index < len(items):
            diff = abs((items[index].datetime - query_date).days)
            if diff < min_diff:
                closest_item = items[index]
                min_diff = diff

        # The closest item is added if it is within the temporal tolerance
        closest_items.append(closest_item if min_diff <= temporal_tolerance else None)
    return closest_items


def add_s1_items(
    client: Client,
    data: pd.DataFrame,
    src_crs: int,
    num_steps: int = 3,
    temporal_step: int = 10,
    temporal_tolerance: int = 12,
) -> dict[str, pd.DataFrame]:
    """Searches and adds Sentinel-1 Granules.

    Data contains tile_id and a series of date for which the tile is desired. This
    function takes the tile_id and the dates and finds the Sentinel-1 granules closest to the
    desired date with a tolerance of `temporal_tolerance`.

    Args:
        client (pystac_client.Client): pystac_client client to use to perform the search.
        data (pd.DataFrame): A dataframe containing observations that fall within a
            dense tile.
        src_crs (int): Source CRS of the observations.
        num_steps (int): Number of temporal steps into the past to fetch.
        temporal_step (int): Step size (in days) for creating temporal steps.
        temporal_tolerance (int): Tolerance (in days) for finding closest Sentinel-1 items.


    Returns:
        A dictionary mapping each MGRS tile ID to a DataFrame containing the observations
          that fall within that tile with their associated PySTAC items representing granules.
    """
    tiles_info, tile_queries = get_tile_info(
        data,
        num_steps=num_steps,
        temporal_step=temporal_step,
        temporal_tolerance=temporal_tolerance,
    )
    data["tile_queries"] = tile_queries
    tiles_database = retrieve_s1_metadata(client, tiles_info)
    best_items = find_best_s1_items(data, src_crs, tiles_database, temporal_tolerance)
    return best_items


def create_s1_dataset(
    best_items: dict[str, pd.DataFrame]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Creates the Sentinel-1 dataset from granules found.

    Args:
        best_items (dict[str, pd.DataFrame]) : A dictionary mapping each MGRS tile ID to
        a DataFrame containing the observations that fall within that tile with their
        associated PySTAC items representing granules.

    Returns:
        ([dict[str, Any], dict[str, Any]): A tuple containing a dictionary mapping
        the observations to the granules ID and a dictionary mapping the observations
        to PySTAC items representing the granules.

    """
    s1_dataset = {}
    s1_dataset_with_items = {}
    for tile_id in best_items:
        tile_obsv_ids_counter: Any = Counter()
        obsvs = best_items[tile_id]
        obsvs["s1_granules"] = obsvs.apply(
            lambda obsv: [
                item.id if isinstance(item, Item) else None for item in obsv["s1_items"]
            ],
            axis=1,
        )
        obsvs = obsvs.drop_duplicates(subset=["s1_granules"])
        obsvs = obsvs[obsvs.apply(is_valid_dataset_entry, axis=1)]

        for _, obsv in obsvs.iterrows():
            tile_obsv_id = obsv.date.strftime("%Y-%m-%d") + f"_{tile_id}"
            s1_dataset[f"{tile_obsv_id}_{tile_obsv_ids_counter[tile_obsv_id]}"] = {
                "granules": [item.to_dict() for item in obsv["s1_items"]]
            }
            s1_dataset_with_items[
                f"{tile_obsv_id}_{tile_obsv_ids_counter[tile_obsv_id]}"
            ] = {"items": obsv["s1_items"]}
            tile_obsv_ids_counter[tile_obsv_id] += 1
    return s1_dataset, s1_dataset_with_items


def is_valid_dataset_entry(obsv: pd.Series) -> bool:
    """Checks S1 granules validity for a given observation.

    The granules will be added to the dataset if they are all non
    null and unique for all timesteps.

    Args:
        obsv (pandas.Series): Observation data for which to assess the validity

    Returns:
        True if the granules are unique and non null.
    """
    if any(granule is None for granule in obsv["s1_granules"]) or (
        len(obsv["s1_granules"]) != len(set(obsv["s1_granules"]))
    ):
        return False
    return True


def load_pystac_items_from_dataset(s1_dataset: dict[str, Any]) -> dict[str, Any]:
    """Loads PySTAC items from serialized items data provided in dataset.

    Args:
        s1_dataset (dict[str, Any]): Dictionary containing the dataset.

    Returns:
        A dictionary mapping the granules ID to PySTAC items.
    """
    s1_dataset_with_items = {}
    for entry in s1_dataset:
        s1_dataset_with_items[entry] = {
            "items": [
                Item.from_dict(item_dict) for item_dict in s1_dataset[entry]["granules"]
            ]
        }
    return s1_dataset_with_items


def open_s1_cogs(
    tile_dict: dict[str, Any], load_masks: bool = False
) -> tuple[xr.DataArray, None, str]:
    """Opens multiple S1 COGs as an xarray DataArray from given PySTAC items.

    Args:
        tile_dict (dict[str, Any]): A dictionary containing PySTAC items to load
        for all timesteps of interest.
        load_masks (bool): Whether or not to load the masks COGs.

    Returns:
        (xr.DataArray, None, str): A tuple of xarray DataArray combining
        data from all the COGs bands of interest, None (as a placeholder for the masks)
        and the CRS used.
    """
    stacked_items = stackstac.stack(
        planetary_computer.sign(ItemCollection(tile_dict["items"])),
        assets=POLARIZATIONS,
        epsg=4326,
        chunksize=BLOCKSIZE,
        properties=False,
        rescale=False,
        fill_value=NO_DATA_VALUES.get("S1"),
    )
    bands = adjust_dims(stacked_items.sel(band=POLARIZATIONS)).astype("float32")
    return bands, None, bands.crs


def decode_mask() -> None:
    """Dummy function for decoding masks."""
    pass
