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

"""Utility Functions for Reading and Processing Sentinel-2 Dataset."""

import glob
import multiprocessing
import os
import subprocess
import time
import zipfile
from datetime import datetime
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import planetary_computer
import rasterio
import requests  # type: ignore
import stackstac
import xarray as xr
from absl import logging
from pystac.item import Item
from pystac.item_collection import ItemCollection
from pystac_client import Client, ItemSearch
from rasterio.crs import CRS
from tqdm import tqdm

from instageo.data.data_pipeline import (
    NO_DATA_VALUES,
    adjust_dims,
    get_tile_info,
    make_valid_bbox,
)

BLOCKSIZE = 512
COLLECTION = ["sentinel-2-l2a"]
S2_HLS_COMMON_BANDS_ASSET = ["B02", "B03", "B04", "B8A", "B11", "B12"]
SCL_ASSET = ["SCL"]


class S2AuthState:
    """Authentication class.

    S2AuthState manages the authentication state for accessing Sentinel-2 resources.
    This class handles token acquisition and refreshing, ensuring that the authentication
    tokens remain valid for API requests or downloads.

    Attributes:
        client_id (str | None): The client ID used for authentication.
        username (str | None): The username for authentication.
        password (str | None): The password for authentication.
        access_token (Optional[str]): The current access token used for API requests.
            Initially None and updated after authentication.
        refresh_token (Optional[str]): The refresh token used for obtaining a new access token.
            Initially None and updated after authentication.
        token_expiry_time (Optional[float]): The timestamp (in seconds since epoch) indicating
            when the current access token will expire. Initially None.

    """

    def __init__(
        self, client_id: str | None, username: str | None, password: str | None
    ):
        """Initializes the S2AuthState instance with user credentials.

        Args:
            client_id (str | None): The client ID for authentication.
            username (str | None): The username for authentication.
            password (str | None): The password for authentication.
        """
        self.client_id = client_id
        self.username = username
        self.password = password
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expiry_time: Optional[float] = None

    def authenticate(self) -> None:
        """Authenticates the user and initializes the access and refresh tokens.

        Raises:
            ValueError: If authentication fails.
        """
        access_token, refresh_token, expires_in = self._get_access_and_refresh_token()
        if access_token is None or refresh_token is None or expires_in is None:
            raise ValueError("Failed to authenticate and obtain tokens.")

        self.access_token = access_token
        self.refresh_token = refresh_token
        self.token_expiry_time = time.time() + expires_in

    def _get_access_and_refresh_token(
        self,
    ) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """Access and refresh token creator.

        Obtains an access token and a refresh token by authenticating with the provided
        client ID, username, and password.

        Args:
            client_id (str | None): The client ID for authentication.
            username (str | None): The username for authentication.
            password (str | None): The password for authentication.

        Returns:
            Tuple[Optional[str], Optional[str], Optional[int]]: A tuple containing the access token,
            refresh token, and the token expiration time in seconds. Returns (None, None, None)
            if the authentication fails.
        """
        url = (
            "https://identity.dataspace.copernicus.eu/auth/realms/"
            "CDSE/protocol/openid-connect/token"
        )
        payload = {
            "client_id": self.client_id,
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(url, data=payload, headers=headers)

        if response.status_code == 200:
            token_data = response.json()
            return (
                token_data.get("access_token"),
                token_data.get("refresh_token"),
                token_data.get("expires_in"),
            )
        else:
            logging.info(f"Failed to get access token: {response.text}")
            return None, None, None

    def _refresh_access_token(self) -> Tuple[Optional[str], Optional[int]]:
        """Access token refresher.

        Refreshes an access token using the provided client ID and refresh token.

        Args:
            client_id (str): The client ID for authentication.
            refresh_token (str): The refresh token used to request a new access token.

        Returns:
            Tuple[Optional[str], Optional[int]]: A tuple containing the new access token
            and its expiration time in seconds, or (None, None) if the refresh attempt fails.
        """
        url = (
            "https://identity.dataspace.copernicus.eu/auth/realms/"
            "CDSE/protocol/openid-connect/token"
        )
        payload = {
            "client_id": self.client_id,
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(url, data=payload, headers=headers)

        if response.status_code == 200:
            token_data = response.json()
            return token_data.get("access_token"), token_data.get("expires_in")
        else:
            logging.info(f"Failed to refresh access token: {response.text}")
            return None, None

    def refresh_access_token_if_needed(self) -> str:
        """Refreshes the access token if it has expired.

        Returns:
            str: The current (refreshed) access token.

        Raises:
            ValueError: If token refresh fails.
        """
        current_time = time.time()
        if self.access_token is None or self.refresh_token is None:
            logging.info("Authentication state is invalid, reauthenticating...")
            self.authenticate()
            if self.access_token is None:
                raise ValueError("Failed to refresh or reauthenticate access token.")
            return self.access_token

        if self.token_expiry_time is None or current_time >= self.token_expiry_time:
            logging.info("Access token expired or not initialized, refreshing...")
            access_token, expires_in = self._refresh_access_token()
            if access_token is None or expires_in is None:
                logging.info("Refresh token expired or invalid, reauthenticating...")
                self.authenticate()
            else:
                self.access_token = access_token
                self.token_expiry_time = current_time + expires_in

        return self.access_token


def download_with_wget(access_token: str, download_url: str, output_file: str) -> None:
    """Wrapper to download using wget.

    Downloads a file using the wget command with an authorization header for secure access.

    Args:
        access_token (str): The access token used for authorization in the download request.
        download_url (str): The URL of the file to be downloaded.
        output_file (str): The path where the downloaded file will be saved.

    Returns:
        None
    """
    wget_command = [
        "wget",
        "--header",
        f"Authorization: Bearer {access_token}",
        download_url,
        "-O",
        output_file,
        "--progress=dot:giga",
    ]

    try:
        process = subprocess.Popen(
            wget_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="")
        process.wait()
        if process.returncode == 0:
            logging.info(f"Download completed: {output_file}")
        else:
            logging.info(f"Download failed with exit code {process.returncode}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download: {e}")


def download_worker(
    client_id: str | None,
    username: str | None,
    password: str | None,
    download_info: Tuple[str, str],
    output_directory: str,
) -> None:
    """Worker function for downloading a Sentinel-2 product.

    Each process has its own authentication state to avoid inconsistencies. Token refreshing and
    reauthentication are handled independently in each worker.

    Args:
        client_id (str | None): The client ID for authentication.
        username (str | None): The username for authentication.
        password (str | None): The password for authentication.
        download_info (Tuple[str, str, str]): Download information tuple.
        output_directory (str): Directory to save the downloaded file.

    Returns:
        None
    """
    auth_state = S2AuthState(client_id=client_id, username=username, password=password)
    auth_state.authenticate()
    access_token = auth_state.refresh_access_token_if_needed()

    download_url, tile_id = download_info
    output_file = os.path.join(output_directory, f"{tile_id}.zip")

    download_with_wget(access_token, download_url, output_file)


def parallel_downloads_s2(
    client_id: str | None,
    username: str | None,
    password: str | None,
    download_info_list: List[Tuple[str, str]],
    output_directory: str,
    num_workers: int = 4,
) -> None:
    """Parallel download for Sentinel-2 Tiles.

    Manages parallel downloads of products using multiprocessing, handling token generation
    and refresh.

    Args:
        client_id (str | None): The client ID for authentication.
        username (str | None): The username for authentication.
        password (str | None): The password for authentication.
        download_info_list (List[Tuple[str, str]]): A list of download information tuples.
        output_directory (str): The directory path where the downloaded files will be saved.
        num_workers (int): Number of parallel processes used during download.

    Returns:
        None
    """
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(
            download_worker,
            [
                (
                    client_id,
                    username,
                    password,
                    download_info,
                    output_directory,
                )
                for download_info in download_info_list
            ],
        )


def open_mf_jp2_dataset(
    band_files: dict[str, dict[str, str]],
    load_masks: bool = False,
) -> tuple[xr.Dataset, xr.Dataset | None, CRS]:
    """Open multiple JP2 files as an xarray Dataset.

    Args:
        band_files (Dict[str, Dict[str, str]]): A dictionary mapping band names to file paths.
        mask_cloud (bool): Perform cloud masking.
        water_mask (bool): Perform water masking.
        load_masks (bool): Whether or not to load the masks files.


    Returns:
        (xr.Dataset, CRS): A tuple of xarray Dataset combining data from all the
            provided TIFF files and its CRS
    """
    band_paths = []
    scl_paths = []
    for granule in band_files["granules"]:
        for band in S2_HLS_COMMON_BANDS_ASSET:
            pattern = os.path.join(
                granule, "GRANULE/*", "IMG_DATA/R20m", f"*_{band}_20m.jp2"
            )
            matching_paths = glob.glob(pattern)
            band_paths.append(matching_paths[0])
        pattern = os.path.join(granule, "GRANULE/*", "IMG_DATA/R20m", "*_SCL_20m.jp2")
        matching_paths = glob.glob(pattern)
        scl_paths.append(matching_paths[0])

    bands_dataset = xr.open_mfdataset(
        band_paths,
        concat_dim="band",
        combine="nested",
        mask_and_scale=False,  # Scaling will be applied manually
    )
    bands_dataset.band_data.attrs["scale_factor"] = 1
    scl_data = (
        xr.open_mfdataset(
            scl_paths,
            concat_dim="band",
            combine="nested",
        )
        if load_masks
        else None
    )

    with rasterio.open(band_paths[0]) as src:
        crs = src.crs
    return bands_dataset, scl_data, crs


def create_mask_from_scl(
    scl_data: xr.Dataset | xr.DataArray, class_ids: list[int]
) -> xr.Dataset | xr.DataArray:
    """Creates masks based on SCL data .

    Arguments:
        scl_data: SCL input xarray Dataset or DataArray.
        class_ids: Class ids to use to produce the mask.

    Returns:
        Xarray dataset or dataarray containing the produced mask.
    """
    return scl_data.isin(class_ids).astype(np.uint8)


def process_s2_metadata(metadata: dict, tile_id: str) -> pd.DataFrame:
    """Processes Sentinel-2 Metadata.

    Given Sentinel-2 metadata retrieved from the Copernicus OpenSearch API, this function extracts
    the relevant fields for each granule found.

    Args:
        metadata (dict): Dict containing various metadata about the retrieved Sentinel-2 granules.
        tile_id (str): ID of the tile used to query the Copernicus OpenSearch API.

    Returns:
        A pandas dataframe containing the granules and extracted metadata.
    """
    granules = [
        {
            "uuid": granule["id"],
            "title": granule["properties"]["title"],
            "tile_id": granule["properties"]["title"].split("_")[5],
            "date": granule["properties"]["startDate"],
            "url": granule["properties"]["services"]["download"]["url"],
            "size": granule["properties"]["services"]["download"]["size"],
            "cloud_cover": granule["properties"]["cloudCover"],
            "thumbnail": granule["properties"]["thumbnail"],
        }
        for granule in metadata["features"]
    ]
    if granules:
        granules_df = pd.DataFrame(granules)
        granules_df = granules_df[granules_df["tile_id"].str.contains(tile_id)]
    else:
        granules_df = None
    return granules_df


def retrieve_s2_metadata(
    tile_info_df: pd.DataFrame, cloud_coverage: int = 10
) -> dict[str, list[str]]:
    """Retrieve Sentinel-2 Tiles Metadata.

    Given a tile_id, start_date and end_date, this function fetches all the Sentinel-2 granules
    available for this tile_id in this time window.

    Args:
        tile_info_df (pd.DataFrame): A dataframe containing tile_id, start_date and
            end_date in each row.
        cloud_coverage (int): Minimum percentage of cloud cover acceptable for a Sentinel-2 tile.

    Returns:
        A dictionary mapping tile_id to a list of available Sentinel-2 granules.
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
    ) in tqdm(
        tile_info_df.iterrows(),
        desc="Retrieving Sentinel-2 Metadata",
        total=len(tile_info_df),
    ):
        lon_min, lat_min, lon_max, lat_max = make_valid_bbox(
            lon_min, lat_min, lon_max, lat_max
        )
        url = (
            "https://catalogue.dataspace.copernicus.eu/"
            f"resto/api/collections/Sentinel2/search.json"
            f"?productType=S2MSI2A&cloudCover=[0,{cloud_coverage}]"
            f"&startDate={start_date}T00:00:00Z"
            f"&completionDate={end_date}T23:59:59Z"
            f"&maxRecords=500"
            f"&box={lon_min},{lat_min},{lon_max},{lat_max}"
        )
        response = requests.get(url)
        if response.status_code == 200 and response.json():
            granules_metadata = response.json()
            granules_dict[tile_id] = process_s2_metadata(granules_metadata, tile_id)
    return granules_dict


def find_best_tile(
    tile_queries: dict[str, tuple[str, list[str]]],
    tile_database: dict[str, pd.DataFrame],
    temporal_tolerance: int = 5,
) -> pd.DataFrame:
    """Find Best Sentinel-2 Tile.

    Sentinel-2 dataset gets updated every 5 days and each tile is marked by the time of
    observation and size. This makes it difficult to deterministically find tiles for a given
    observation time. Rather we try to find a tile with observation time closest to our
    desired time and maximum size which indicates a higher number of valid pixels.

    To do this, we create a database of tiles within a specific timeframe then we search
    for our desired tile within the database.

    Args:
        tile_queries (dict[str, tuple[str, list[str]]]): A dict with tile_query as key
            and a tuple of tile_id and a list  of dates on which the tile needs to be
            retrieved as value.
        tile_database (dict[str, list[str]]): A database mapping Sentinel-2 tile_id to a list of
            available granules within a pre-defined period of time.
        temporal_tolerance: Number of days that can be tolerated for matching a closest
            tile in tile_databse.

    Returns:
        DataFrame containing the tile queries to the tile found.
    """
    query_results = []
    for query_str, (tile_id, dates) in tile_queries.items():
        if (tile_id not in tile_database) or (tile_database[tile_id] is None):
            query_results.append(
                {
                    "tile_queries": query_str,
                    "s2_tiles": [None] * len(dates),
                    "thumbnails": [None] * len(dates),
                    "urls": [None] * len(dates),
                }
            )
            continue

        # Load tile data and query dates
        tile_entries = tile_database[tile_id]
        query_dates = pd.to_datetime(dates)

        tile_entries["date"] = pd.to_datetime(tile_entries["date"]).dt.tz_localize(None)
        s2_tiles, thumbnails, urls = [], [], []

        for query_date in query_dates:
            # Filter tiles within temporal tolerance
            filtered_tiles = tile_entries[
                (
                    tile_entries["date"]
                    >= query_date - pd.Timedelta(days=temporal_tolerance)
                )
                & (
                    tile_entries["date"]
                    <= query_date + pd.Timedelta(days=temporal_tolerance)
                )
            ]

            if not filtered_tiles.empty:
                # Sort tiles by size (descending) and temporal difference (ascending)
                best_tile = (
                    filtered_tiles.assign(
                        temporal_diff=(filtered_tiles["date"] - query_date).abs()
                    )
                    .sort_values(by=["size", "temporal_diff"], ascending=[False, True])
                    .iloc[0]
                )

                s2_tiles.append(best_tile["title"])
                thumbnails.append(best_tile["thumbnail"])
                urls.append(best_tile["url"])
            else:
                s2_tiles.append(None)
                thumbnails.append(None)
                urls.append(None)

        query_results.append(
            {
                "tile_queries": query_str,
                "s2_tiles": s2_tiles,
                "thumbnails": thumbnails,
                "urls": urls,
            }
        )

    return pd.DataFrame(query_results)


def extract_and_delete_zip_files(parent_dir: str) -> None:
    """Extract and Delete Zip Files.

    Extracts all ZIP files in the subdirectories of the specified parent directory to their
    respective parent directories and deletes the ZIP files after extraction.

    Args:
        parent_dir (str): The path to the parent directory containing subdirectories with ZIP files.

    Returns:
        None
    """
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                extract_to = root

                try:
                    # Extract the ZIP file
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_to)
                    logging.info(f"Extracted: {zip_path} to {extract_to}")

                    # Delete the ZIP file after extraction
                    os.remove(zip_path)
                    logging.info(f"Deleted: {zip_path}")

                except Exception as e:
                    logging.error(f"Error processing {zip_path}: {e}")


def download_tile_data(
    granules_to_download: pd.DataFrame,
    output_directory: str,
    client_id: str | None,
    username: str | None,
    password: str | None,
    max_retries: int = 3,
) -> None:
    """Download Sentinel-2 Tiles .

    Processes the provided tile database to filter products by month, creates necessary
    directories, and initiates parallel downloads if valid products are found.

    Args:
        tile_database (Dict[str, Any]): A dictionary where keys are tile names and values
        are data about each tile.
        output_directory (str): Path to the directory where tiles should be stored.
        client_id (str | None): The client ID for authentication during the download process.
        username (str | None): The username for authentication.
        password (str | None): The password for authentication.

    Returns:
        None
    """
    retries = 0
    complete = False
    while retries <= max_retries:
        download_info_list = [
            (row["urls"], row["tiles"])
            for _, row in granules_to_download.iterrows()
            if not (
                os.path.exists(os.path.join(output_directory, f"{row['tiles']}.zip"))
                or os.path.isfile(
                    os.path.join(output_directory, row["tiles"], "manifest.safe")
                )
            )
        ]
        if not download_info_list:
            complete = True
            break
        parallel_downloads_s2(
            client_id,
            username,
            password,
            download_info_list,
            output_directory,
            num_workers=4,
        )
        for filename in os.listdir(output_directory):
            file_path = os.path.join(output_directory, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        bad_file = zip_ref.testzip()
                        if bad_file:
                            logging.info(f"Deleting {file_path}: Bad ZIP file")
                            os.remove(file_path)
                except zipfile.BadZipFile:
                    logging.info(f"Deleting {file_path}: Corrupted ZIP file")
                    os.remove(file_path)
        retries += 1
    if complete:
        logging.info("Successfully downloaded all granules")
    else:
        logging.warning(
            f"Couldn't download the following granules after {max_retries} retries:\n{download_info_list}"  # noqa
        )


def add_s2_granules(
    data: pd.DataFrame,
    num_steps: int = 3,
    temporal_step: int = 10,
    temporal_tolerance: int = 5,
    cloud_coverage: int = 10,
) -> pd.DataFrame:
    """Add Sentinel-2 Granules.

    Data contains tile_id and a series of date for which the tile is desired. This
    function takes the tile_id and the dates and finds the Sentinel-2 granules closest to the
    desired date with a tolerance of `temporal_tolerance`.

    Args:
        data (pd.DataFrame): A dataframe containing observations that fall within a
            dense tile.
        num_steps (int): Number of temporal steps into the past to fetch.
        temporal_step (int): Step size (in days) for creating temporal steps.
        temporal_tolerance (int): Tolerance (in days) for finding closest Sentinel-2 granule.
        cloud_coverage (int): Minimum percentage of cloud cover acceptable for a Sentinel-2 tile.


    Returns:
        A dataframe containing a list of Sentinel-2 granules. Each granule is a directory
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
    tile_database = retrieve_s2_metadata(tiles_info, cloud_coverage)
    tile_queries_dict = {k: v for k, v in zip(tile_queries_str, tile_queries)}
    query_result = find_best_tile(
        tile_queries=tile_queries_dict,
        tile_database=tile_database,
        temporal_tolerance=temporal_tolerance,
    )
    data = pd.merge(data, query_result, how="left", on="tile_queries")
    return data


def create_s2_dataset(
    data_with_tiles: pd.DataFrame, outdir: str
) -> tuple[dict[str, dict[str, list[str]]], pd.DataFrame]:
    """Creates Sentinel-2 Dataset.

    A Sentinel-2 dataset is a JSON mapping unique granule ids in the dataset to their respective
    granule paths. This is required for creating chips for the entire dataset.

    Args:
        data_with_tiles (pd.DataFrame): A dataframe containing observations that fall
            within a dense tile. It also has `s2_tiles` column that contains a temporal
            series of Sentinel-2 granules.
        outdir (str): Output directory where tiles could be downloaded to.

    Returns:
        A tuple containing Sentinel-2 dataset and a list of tiles that needs to be downloaded.
    """
    data_with_tiles = data_with_tiles.drop_duplicates(subset=["s2_tiles"])
    data_with_tiles = data_with_tiles[
        data_with_tiles["s2_tiles"].apply(
            lambda granule_lst: all(str(item).startswith("S2") for item in granule_lst)
        )
    ]
    assert (
        not data_with_tiles.empty
    ), "No observation record with valid Sentinel-2 granules"
    s2_dataset = {}
    tiles_to_download = []
    urls = []
    for _, row in data_with_tiles.iterrows():
        s2_dataset[f'{row["date"].strftime("%Y-%m-%d")}_{row["mgrs_tile_id"]}'] = {
            "granules": [
                os.path.join(outdir, "s2_tiles", tile) for tile in row["s2_tiles"]
            ],
        }
        tiles_to_download.extend(row["s2_tiles"])
        urls.extend(row["urls"])
    granules_to_download = pd.DataFrame(
        {"tiles": tiles_to_download, "urls": urls}
    ).drop_duplicates()
    return s2_dataset, granules_to_download


def search_and_open_s2_cogs(
    client: Client,
    bands_to_load: List[str],
    tile_dict: dict[str, Any],
    load_masks: bool = False,
) -> tuple[xr.DataArray, xr.DataArray | None, str]:
    """Searches and opens multiple S2 COGs as an xarray DataArray from given granules IDs.

    Args:
        client (pystac_client.Client): pystac_client client to use to perform the search.
        bands_to_load (List[str]): List of bands to load.
        tile_dict (dict[str, Any]): A dictionary containing granules IDs to retrieve
        for all timesteps of interest.
        load_masks (bool): Whether or not to load the masks COGs.

    Returns:
        (xr.DataArray, xr.DataArray | None, str): A tuple of xarray DataArray combining
        data from all the COGs bands of interest, (optionally) the COGs masks and the
        CRS used.
    """
    search_objs = get_item_search_objs(client, tile_dict)
    results = get_item_collection(search_objs)

    # Load the bands for all timesteps and stack them in a data array
    assets_to_load = bands_to_load + SCL_ASSET if load_masks else bands_to_load
    stacked_items = stackstac.stack(
        planetary_computer.sign(ItemCollection(results)),
        assets=assets_to_load,
        chunksize=BLOCKSIZE,
        properties=False,
        rescale=False,
        fill_value=NO_DATA_VALUES.get("S2"),
    )

    bands = adjust_dims(stacked_items.sel(band=bands_to_load))
    masks = adjust_dims(stacked_items.sel(band=SCL_ASSET)) if load_masks else None

    bands = bands.astype(np.uint16)
    bands.attrs["scale_factor"] = 1

    return bands, masks, bands.crs


def get_item_search_objs(client: Client, tile_dict: dict[str, Any]) -> List[ItemSearch]:
    """Creates a list of ItemSearch objects by utilizing the granules IDs provided.

    Args:
        client (pystac_client.Client): pystac_client client to use to perform the search.
        tile_dict (dict[str, Any]): A dictionary containing granules IDs to retrieve
        for all timesteps of interest.

    Returns:
       List[ItemSearch]: A list of search objects.
    """
    s2_ids = list(map(lambda path: path.split("/")[-1], tile_dict["granules"]))
    dates = [
        datetime.strptime(id.split("_")[2].split("T")[0], "%Y%m%d").strftime("%Y-%m-%d")
        for id in s2_ids
    ]
    tile = s2_ids[0].split("_")[5][1:]
    search_objs = [
        client.search(
            collections=[COLLECTION],
            datetime=date,
            query={
                "s2:mgrs_tile": {"eq": tile},
            },
        )
        for date in dates
    ]

    return search_objs


def get_item_collection(search_objs: List[ItemSearch]) -> List[Item]:
    """Iterates through search objects to return a list of STAC items.

    Args:
        search_objs (ItemSearch): Search objects to "browse" to retrieve
        the items collection.

    Returns:
       List[Item]: A list of items from the search objects.
    """
    return [list(search_obj.item_collection())[0] for search_obj in search_objs]
