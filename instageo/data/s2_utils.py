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

import multiprocessing
import os
import shutil
import subprocess
import time
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import requests  # type: ignore
import rioxarray
import xarray as xr
from rasterio.crs import CRS


def get_access_and_refresh_token(
    client_id: str, username: str, password: str
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Access and refresh token creator.

    Obtains an access token and a refresh token by authenticating with the provided
    client ID, username, and password.

    Args:
        client_id (str): The client ID for authentication.
        username (str): The username for authentication.
        password (str): The password for authentication.

    Returns:
        Tuple[Optional[str], Optional[str], Optional[int]]: A tuple containing the access token,
        refresh token, and the token expiration time in seconds. Returns (None, None, None)
        if the authentication fails.
    """
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    payload = {
        "client_id": client_id,
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(url, data=payload, headers=headers)

    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data["access_token"]
        refresh_token = token_data["refresh_token"]
        expires_in = token_data["expires_in"]
        return access_token, refresh_token, expires_in
    else:
        print("Failed to get access token:", response.text)
        return None, None, None


def refresh_access_token(
    client_id: str, refresh_token: str
) -> Tuple[Optional[str], Optional[int]]:
    """Access token refresher.

    Refreshes an access token using the provided client ID and refresh token.

    Args:
        client_id (str): The client ID for authentication.
        refresh_token (str): The refresh token used to request a new access token.

    Returns:
        Tuple[Optional[str], Optional[int]]: A tuple containing the new access token
        and its expiration time in seconds, or (None, None) if the refresh attempt fails.
    """
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    payload = {
        "client_id": client_id,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(url, data=payload, headers=headers)

    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data["access_token"]
        expires_in = token_data["expires_in"]
        return access_token, expires_in
    else:
        print("Failed to refresh access token:", response.text)
        return None, None


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
            print(f"Download completed: {output_file}")
        else:
            print(f"Download failed with exit code {process.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download: {e}")


def check_and_refresh_token(
    client_id: str, refresh_token: str, access_token: str, token_expiry_time: float
) -> Tuple[str, float]:
    """Token refresh.

    Checks if the current access token has expired and refreshes it if necessary.

    Args:
        client_id (str): The client ID used for authentication.
        refresh_token (str): The refresh token used to obtain a new access token.
        access_token (str): The current access token for authentication.
        token_expiry_time (float): The timestamp in seconds when the access token
        will expire.

    Returns:
        Tuple[str, int]: The updated access token and its new expiry time.
    """
    current_time = time.time()
    if current_time >= token_expiry_time:
        print("Access token expired, refreshing...")
        new_access_token, expires_in = refresh_access_token(client_id, refresh_token)

        if new_access_token is None or expires_in is None:
            raise ValueError("Failed to refresh access token or expiry time.")

        access_token = new_access_token
        token_expiry_time = current_time + expires_in
    return access_token, token_expiry_time


def filter_products_by_month(
    tile_products: List[Dict[str, str]], temporal_step: int, num_steps: int
) -> Optional[List[Dict[str, str]]]:
    """Filter S2 products.

    Filters products to ensure that they include acquisitions from a specified number of
    different months if the temporal step condition is met.

    Args:
        tile_products (List[Dict[str, str]]): A list of product dictionaries, each containing an
        'acquisition_date' key with the date formatted as a string ('%Y-%m-%d').
        temporal_step (int): The temporal step in days used to decide whether to apply the
        filtering condition.
        num_steps (int): The required number of distinct months that products should span.

    Returns:
        Optional[List[Dict[str, str]]]: The list of filtered products if they span at least
        `num_steps` distinct months; otherwise, returns None.
    """
    # Only proceed with filtering if the temporal step is 30 or more
    if temporal_step < 30:
        return tile_products

    filtered_products = []
    months_seen = set()

    for product in tile_products:
        acquisition_date = datetime.strptime(product["acquisition_date"], "%Y-%m-%d")
        month_key = (acquisition_date.year, acquisition_date.month)
        months_seen.add(month_key)
        filtered_products.append(product)

    # Ensure there are products from at least "num_steps" distinct months
    return filtered_products if len(months_seen) >= num_steps else None


def download_product(
    client_id: str,
    refresh_token: str,
    access_token: str,
    download_info: Tuple[str, str, str],
    token_expiry_time: float,
    output_directory: str,
) -> None:
    """Download Sentinel-2 product.

    Downloads a product using the provided download information and handles token refreshing.

    Args:
        client_id (str): The client ID for authentication.
        refresh_token (str): The refresh token for renewing access.
        access_token (str): The current access token for download authentication.
        download_info (Tuple[str, str, str]): A tuple containing the download URL, full tile ID,
        and tile name.
        token_expiry_time (float): The expiry time for the current access token.
        output_directory (str): The directory path where the downloaded files will be saved.

    Returns:
        None
    """
    download_url, full_tile_id, tile_name = download_info
    output_file = os.path.join(output_directory, tile_name, f"{full_tile_id}.zip")
    access_token, token_expiry_time = check_and_refresh_token(
        client_id, refresh_token, access_token, token_expiry_time
    )
    download_with_wget(access_token, download_url, output_file)


def parallel_downloads_s2(
    client_id: str,
    username: str,
    password: str,
    download_info_list: List[Tuple[str, str, str]],
    output_directory: str,
) -> None:
    """Parallel download for Sentinel-2 Tiles.

    Manages parallel downloads of products using multiprocessing, handling token generation
    and refresh.

    Args:
        client_id (str): The client ID for authentication.
        username (str): The username for authentication.
        password (str): The password for authentication.
        download_info_list (List[Tuple[str, str, str]]): A list of download information tuples.
        output_directory (str): The directory path where the downloaded files will be saved.

    Returns:
        None
    """
    access_token, refresh_token, expires_in = get_access_and_refresh_token(
        client_id, username, password
    )

    if expires_in is None:
        raise ValueError("Failed to obtain a valid token expiry time.")

    token_expiry_time = time.time() + expires_in

    with multiprocessing.Pool(processes=4) as pool:
        pool.starmap(
            download_product,
            [
                (
                    client_id,
                    refresh_token,
                    access_token,
                    download_info,
                    token_expiry_time,
                    output_directory,
                )
                for download_info in download_info_list
            ],
        )


def unzip_file(zip_file: str, output_dir: str) -> None:
    """Unzip S2 products.

    Extracts the contents of a zip file to the specified output directory.

    Args:
        zip_file (str): The path to the zip file to be extracted.
        output_dir (str): The directory where the contents of the zip file will be extracted.

    Returns:
        None: This function doesn't return any value. It prints messages indicating the status.

    Raises:
        zipfile.BadZipFile: If the provided file is not a valid zip file.
        FileNotFoundError: If the zip file is not found at the given path.
    """
    try:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(output_dir)
            print(f"Extracted: {zip_file} to {output_dir}")
        os.remove(zip_file)
        print(f"Deleted zip file: {zip_file}")
    except zipfile.BadZipFile:
        print(f"Error: {zip_file} is not a valid zip file")
    except FileNotFoundError:
        print(f"File not found: {zip_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_band_files(tile_name: str, full_tile_id: str, output_directory: str) -> None:
    """Handle band files.

    Navigates through the folders to find the bands in the IMG_DATA/R20m folder,
    moves them to the tile folder, and deletes unnecessary files.

    Args:
        tile_name (str): The tile name.
        full_tile_id (str): The base folder where the unzipped contents are stored.
        output_directory (str): The root directory where the tile folders are located.

    Returns:
    None: This function doesn't return any value. It performs file operations and prints messages.
    """
    bands_needed = ["B02", "B03", "B04", "B8A", "B11", "B12", "SCL"]
    tile_folder = os.path.join(output_directory, tile_name)
    print(f"Main directory: {tile_folder}")

    base_dir = os.path.join(tile_folder, full_tile_id, "GRANULE")
    print(f"Base directory: {base_dir}")

    if os.path.isdir(base_dir):
        granule_folders = os.listdir(base_dir)
        if granule_folders:
            granule_folder = granule_folders[0]
            img_data_dir = os.path.join(base_dir, granule_folder, "IMG_DATA", "R20m")

            if os.path.isdir(img_data_dir):
                for file_name in os.listdir(img_data_dir):
                    for band in bands_needed:
                        if band in file_name:
                            full_file_path = os.path.join(img_data_dir, file_name)
                            target_file_path = os.path.join(tile_folder, file_name)

                            shutil.move(full_file_path, target_file_path)
                            print(f"Moved {file_name} to {tile_folder}")

                shutil.rmtree(os.path.join(tile_folder, full_tile_id))
                print(f"Deleted unneeded directories for {full_tile_id}")
            else:
                print(f"R20m folder not found in {granule_folder}")
        else:
            print(f"No subfolder found in GRANULE for {full_tile_id}")
    else:
        print(f"GRANULE folder not found in {full_tile_id}")


def count_valid_pixels(scl_band_path: str) -> int:
    """Valid pixels count.

    Count valid pixels in the SCL band, ignoring pixels with class 0 (No Data).

    Args:
        scl_band_path (str): The file path to the SCL band.

    Returns:
    int: The number of valid pixels.
    """
    with rasterio.open(scl_band_path) as src:
        scl_data = src.read(1)  # Read the SCL band data as a 2D numpy array
        valid_pixels = np.count_nonzero(scl_data != 0)
        print("valid_pixels for ", scl_band_path, valid_pixels)
    return valid_pixels


def find_scl_file(
    output_path: str, tile_name: str, acquisition_date: datetime
) -> Optional[str]:
    """SCL file finder.

    Find the SCL file matching the acquisition date.

    Args:
        output_path (str): Path to the tile's folder.
        tile_name (str): Name of the tile.
        acquisition_date (datetime): Acquisition date for the product.

    Returns:
    Optional[str]: Path to the matching SCL file, or None if not found.
    """
    acquisition_date_str = acquisition_date.strftime("%Y%m%d")

    for file_name in os.listdir(output_path):
        if file_name.endswith("_SCL_20m.jp2") and tile_name in file_name:
            if acquisition_date_str in file_name:
                return os.path.join(output_path, file_name)

    return None


def open_mf_jp2_dataset(
    band_folder: str, num_steps: int, mask_cloud: bool, water_mask: bool
) -> tuple[xr.Dataset | None, CRS | None]:
    """Handle JP2 data.

    Open multiple JP2 files as an xarray Dataset and optionally apply filtering for water
      and clouds.

    Args:
        band_folder (str): Path to the folder where the bands and SCL band are stored.
        num_steps (int): Number of timestamps or steps, indicating how many sets of bands
        are expected.
        mask_cloud (bool): Whether to apply cloud filtering (classes 8 and 9).
        water_mask (bool): Whether to apply water filtering (class 6).

    Returns:
        (xr.Dataset | None, CRS | None): A tuple of xarray Dataset combining data from all the
            provided JP2 files and its CRS, or (None, None) if the folder is invalid.
    """
    if not os.path.exists(band_folder):
        print(f"Folder '{band_folder}' does not exist. Skipping...")
        return None, None

    band_crs = None

    band_files = [
        os.path.join(band_folder, f)
        for f in os.listdir(band_folder)
        if f.endswith(".jp2") and "SCL" not in f
    ]
    scl_band_files = [
        os.path.join(band_folder, f)
        for f in os.listdir(band_folder)
        if "SCL" in f and f.endswith(".jp2")
    ]

    if len(band_files) % 6 != 0:
        print(f"Unexpected number of band files in '{band_folder}': {len(band_files)}")
        return None, band_crs

    expected_band_count = num_steps * 6  # Assuming 6 bands per timestamp
    if len(band_files) < expected_band_count:
        print(
            f"Not enough band files in '{band_folder}'. Expected at least {expected_band_count}, "
            f"found {len(band_files)}."
        )
        return None, band_crs

    if len(scl_band_files) != num_steps:
        print(f"Skipping folder '{band_folder}' - missing SCL bands for each timestamp")
        return None, band_crs

    band_files.sort()
    bands_list = []

    for band_file in band_files:
        if os.path.getsize(band_file) > 0:
            band_data = rioxarray.open_rasterio(band_file)
            bands_list.append(band_data)
            if band_crs is None:
                band_crs = band_data.rio.crs
        else:
            print(f"Skipping empty band file: {band_file}")

    if not bands_list:
        print(f"Skipping folder '{band_folder}' - no valid band files found")
        return None, band_crs

    bands_dataset = xr.concat(bands_list, dim="band")

    scl_band_files.sort()
    scl_data = []

    for scl_path in scl_band_files:
        with rasterio.open(scl_path) as scl_src:
            scl_data.append(scl_src.read(1))

    scl_data = np.stack(scl_data, axis=0)

    if water_mask:
        water_mask_array = np.isin(scl_data, [6]).astype(np.uint8)  # Class 6 for water
        bands_dataset = bands_dataset.where(water_mask_array == 0)

    if mask_cloud:
        cloud_mask_array = np.isin(scl_data, [8, 9]).astype(
            np.uint8
        )  # Classes 8 and 9 for clouds
        bands_dataset = bands_dataset.where(cloud_mask_array == 0)

    return bands_dataset, band_crs
