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
from itertools import chain
from typing import Any

import dask
import dask.delayed
import earthaccess
import pandas as pd
import rasterio
import rioxarray as rxr
import xarray as xr
from absl import logging
from rasterio.crs import CRS

from instageo.data.data_pipeline import get_tile_info, make_valid_bbox

# Block sizes for the internal tiling of HLS COGs
BLOCKSIZE_X = 256
BLOCKSIZE_Y = 256


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


def decode_fmask_value(
    value: xr.Dataset | xr.DataArray, position: int
) -> xr.Dataset | xr.DataArray:
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
        cloud_coverage (int): Minimum percentage of cloud cover acceptable for a HLS tile.

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
            cloud_cover=cloud_coverage,
        )
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
        chunks=dict(band=1, x=BLOCKSIZE_X, y=BLOCKSIZE_Y),
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
        cloud_coverage (int): Minimum percentage of cloud cover acceptable for a HLS tile.

    Returns:
        A dataframe containing a list of HLS granules. Each granule is a directory
        containing all the bands.
    """
    tiles_info, tile_queries = get_tile_info(
        data,
        num_steps=num_steps,
        temporal_step=temporal_step,
        temporal_tolerance=temporal_tolerance,
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
            hls_dataset[f'{obsv_date.strftime("%Y-%m-%d")}_{tile.split(".")[2]}'] = {
                "tiles": bands_paths,
                "fmasks": masks_paths,
                "data_links": obsv_data_links,
            }

    return hls_dataset, set(chain.from_iterable(data_links))


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
