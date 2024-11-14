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

# import zipfile
import multiprocessing
import os
import subprocess
import time
from datetime import datetime

# import shutil
# import numpy as np
# import rasterio
from typing import Dict, List, Optional, Tuple

import requests  # type: ignore


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
    """Parallel download for Sentinel-2 Tiles .

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
