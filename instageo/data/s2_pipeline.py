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

"""HLS pipeline Module."""

import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import pandas as pd
import rasterio
import requests  # type: ignore
from absl import logging
from shapely.geometry import Point

from instageo.data.geo_utils import create_segmentation_map, get_chip_coords
from instageo.data.s2_utils import (
    count_valid_pixels,
    filter_products_by_month,
    find_scl_file,
    get_band_files,
    open_mf_jp2_dataset,
    parallel_downloads_s2,
    unzip_file,
)


def retrieve_sentinel2_metadata(
    tile_df: pd.DataFrame,
    cloud_coverage: float,
    temporal_tolerance: int,
    history_dates: list[tuple[str, list[str]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieve Sentinel-2 Tiles Metadata.

    Given a tile_id, start_date, and a time window, this function fetches all
    Sentinel-2 granules available for this tile_id in this time window, but only
    keeps tiles that match the `history_dates` or `tile_df["mgrs_tile"]`.

    Args:
        tile_df (pd.DataFrame): A dataframe containing tile_id, start_date, and geographical
        boundaries.
        cloud_coverage (float): Maximum acceptable cloud coverage for each granule.
        temporal_tolerance (int): Number of days before and after each historical date for
        the time window.
        history_dates (List[Tuple[str, List[str]]]): List of time windows for each
        tile, with acquisition dates.

    Returns:
        A dictionary mapping tile_id to a list of available Sentinel-2 granules.
    """
    granules_dict: Dict[str, List[Dict[str, Any]]] = {}
    unique_full_tile_ids = set()

    valid_tile_ids = set(tile_df["tile_id"].unique())
    valid_tile_ids.update([tile for tile, _ in history_dates])

    for _, row in tile_df.iterrows():
        lon_min, lon_max, lat_min, lat_max = (
            row["lon_min"],
            row["lon_max"],
            row["lat_min"],
            row["lat_max"],
        )

        for tile_id, date_list in history_dates:
            for date_str in date_list:
                center_date = pd.to_datetime(date_str)
                start_date_window = (
                    center_date - timedelta(days=temporal_tolerance)
                ).strftime("%Y-%m-%d")
                end_date_window = (
                    center_date + timedelta(days=temporal_tolerance)
                ).strftime("%Y-%m-%d")

                url = (
                    "https://catalogue.dataspace.copernicus.eu/"
                    f"resto/api/collections/Sentinel2/search.json"
                    f"?productType=S2MSI2A&cloudCover=[0,{cloud_coverage}]"
                    f"&startDate={start_date_window}T00:00:00Z"
                    f"&completionDate={end_date_window}T23:59:59Z"
                    f"&maxRecords=500"
                    f"&box={lon_min},{lat_min},{lon_max},{lat_max}"
                )
                response = requests.get(url)
                if response.status_code != 200:
                    continue

                data = response.json()

                if "features" not in data or not data["features"]:
                    continue

                for feature in data["features"]:
                    full_tile_id = feature["properties"]["title"]
                    if full_tile_id in unique_full_tile_ids:
                        continue

                    extracted_tile_id = re.search(r"T(\d{2}[A-Z]{3})", full_tile_id)
                    tile_id_extracted = (
                        extracted_tile_id.group(1) if extracted_tile_id else None
                    )

                    acquisition_date = re.search(r"(\d{8})", full_tile_id)

                    tile_acquisition_date: str | None = (
                        datetime.strptime(acquisition_date.group(1), "%Y%m%d").strftime(
                            "%Y-%m-%d"
                        )
                        if acquisition_date
                        else None
                    )

                    # Check if tile ID is valid before adding
                    if tile_id_extracted and tile_id_extracted in valid_tile_ids:
                        granule_info = {
                            "full_tile_id": full_tile_id,
                            "tile_id": tile_id_extracted,
                            "cloudCover": feature["properties"]["cloudCover"],
                            "download_link": feature["properties"]["services"][
                                "download"
                            ]["url"],
                            "thumbnail": feature["properties"]["thumbnail"],
                            "acquisition_date": tile_acquisition_date,
                        }

                        granules_dict.setdefault(tile_id_extracted, []).append(
                            granule_info
                        )
                        unique_full_tile_ids.add(full_tile_id)

    return granules_dict


def download_tile_data(
    tile_database: Dict[str, Any],
    output_directory: str,
    client_id: str,
    username: str,
    password: str,
    temporal_step: int,
    num_steps: int,
) -> List[Tuple[str, str, str]]:
    """Download Sentinel-2 Tiles .

    Processes the provided tile database to filter products by month, creates necessary
    directories, and initiates parallel downloads if valid products are found.

    Args:
        tile_database (Dict[str, Any]): A dictionary where keys are tile names and values
        are data about each tile.
        output_directory (str): Path to the directory where tiles should be stored.
        client_id (str): The client ID for authentication during the download process.
        username (str): The username for authentication.
        password (str): The password for authentication.
        temporal_step (int): The temporal step for filtering products.
        num_steps (int): The number of steps for filtering products.

    Returns:
        List[Tuple[str, str, str]]: A list of download information tuples containing
        (download_link, full_tile_id, tile_name).
    """
    download_info_list: List[Tuple[str, str, str]] = []

    for tile_name, tile_data in tile_database.items():
        os.makedirs(os.path.join(output_directory, tile_name), exist_ok=True)

        filtered_products = filter_products_by_month(
            tile_data, temporal_step, num_steps
        )

        if filtered_products:
            for entry in filtered_products:
                download_link = entry["download_link"]
                full_tile_id = entry["full_tile_id"]
                download_info_list.append((download_link, full_tile_id, tile_name))

    if download_info_list:
        parallel_downloads_s2(
            client_id, username, password, download_info_list, output_directory
        )
    else:
        print("No valid products found for download.")

    return download_info_list


def unzip_all(
    download_info_list: List[Tuple[str, str, str]], output_directory: str
) -> None:
    """Unzip files.

    Iterates over a list of download information and extracts each zip file
    if it exists, using the unzip_file function.

    Args:
        download_info_list (List[Tuple[str, str, str]]): A list of tuples containing
        download link, full tile ID, and tile name for each zip file to be processed.
        Each tuple follows the format (download_link, full_tile_id, tile_name).
        output_directory (str): The directory where the zip files and extracted contents
        are located.

    Returns:
        None: This function doesn't return any value. It prints messages about the
        extraction process.
    """
    for download_link, full_tile_id, tile_name in download_info_list:
        zip_file = os.path.join(output_directory, tile_name, f"{full_tile_id}.zip")
        output_dir = os.path.join(output_directory, tile_name)

        if os.path.exists(zip_file):
            unzip_file(zip_file, output_dir)
        else:
            print(f"Zip file not found: {zip_file}")


def process_tile_bands(
    tile_database: Dict[str, List[Dict[str, Any]]], output_directory: str
) -> None:
    """Processes each tile in the provided tile database to retrieve specified band files.

    Args:
        tile_database (Dict[str, List[Dict[str, Any]]]): A dictionary where keys are tile names
        and values are lists of products with data about each tile.
        output_directory (str): The path to the directory where tile data will be processed
        and stored.

    Returns:
        None
    """
    for tile_name, tile_data in tile_database.items():
        for product in tile_data:
            full_tile_id = product["full_tile_id"]
            get_band_files(tile_name, full_tile_id, output_directory)


def filter_best_product_in_folder(
    tile_name: str,
    tile_products: List[Dict[str, str]],
    output_directory: str,
    history_dates: List[Tuple[str, List[str]]],
    temporal_tolerance: int,
) -> None:
    """Filter best product.

    Filters the best product (based on highest valid pixel count in the SCL band)
    in each time window for a tile, and removes other products' bands.

    Args:
        tile_name (str): Name of the tile.
        tile_products (list): List of product dictionaries for the tile.
        output_directory (str): The folder where all the tile folders are stored.
        history_dates (List[Tuple[str, List[str]]]): List of time windows for each
        tile, with acquisition dates.
        temporal_tolerance (int): Number of days before and after each historical date for
        the time window.

    Returns:
        None
    """
    folder_path = os.path.join(output_directory, tile_name)

    for tile, dates in history_dates:
        if tile == tile_name:
            sorted_dates = sorted(
                dates, key=lambda date: datetime.strptime(date, "%Y-%m-%d")
            )

            for current_date in sorted_dates:
                center_date = datetime.strptime(current_date, "%Y-%m-%d")
                start_date = center_date - timedelta(days=temporal_tolerance)
                end_date = center_date + timedelta(days=temporal_tolerance)

                print(
                    f"Processing date: {current_date} (window: {start_date} to {end_date})"
                )

                window_products = []
                for product in tile_products:
                    acquisition_date = datetime.strptime(
                        product["acquisition_date"], "%Y-%m-%d"
                    )

                    if start_date <= acquisition_date <= end_date:
                        scl_band_path = find_scl_file(
                            folder_path, tile_name, acquisition_date
                        )
                        if scl_band_path:
                            try:
                                valid_pixel_count = count_valid_pixels(scl_band_path)
                                if valid_pixel_count > 0:
                                    window_products.append((product, valid_pixel_count))
                            except Exception as e:
                                print(
                                    f"ERROR: Could not count valid pixels for {scl_band_path}: {e}"
                                )
                        else:
                            print(
                                f"WARNING: No SCL file found for {product['acquisition_date']}"
                            )

                if window_products:
                    window_products.sort(key=lambda x: x[1], reverse=True)
                    best_product = window_products[0][0]
                    best_product_acquisition_date = datetime.strptime(
                        best_product["acquisition_date"], "%Y-%m-%d"
                    ).strftime("%Y%m%d")

                    print(
                        f"Best product found for {current_date}: {best_product_acquisition_date}"
                    )

                    for folder_item in os.listdir(folder_path):
                        item_path = os.path.join(folder_path, folder_item)

                        if tile_name in folder_item:
                            file_date_str = folder_item.split("_")[1][:8]
                            file_date = datetime.strptime(file_date_str, "%Y%m%d")
                            if start_date <= file_date <= end_date:
                                if (
                                    best_product_acquisition_date not in folder_item
                                    and os.path.isfile(item_path)
                                ):
                                    os.remove(item_path)
                                    print(f"{item_path} removed")
                else:
                    print(
                        f"No valid product found for date {current_date} in time window."
                    )


def create_and_save_chips_with_seg_maps_s2(
    granules_dict: dict,
    sub_data: pd.DataFrame,
    chip_size: int,
    output_directory: str,
    no_data_value: int,
    src_crs: int,
    mask_cloud: bool,
    water_mask: bool,
    temporal_tolerance: int,
    history_dates: list[tuple[str, list[str]]],
) -> tuple[list[str], list[str | None]]:
    """Chip Creator with Loop Inside.

    Processes multiple `band_folder`s directly, handling chips and segmentation map creation.

    Args:
        granules_dict (dict): Dictionary containing granule IDs and metadata.
        sub_data (pd.DataFrame): DataFrame containing the data for segmentation maps.
        chip_size (int): Size of each chip.
        output_directory (str): Directory where the chips and segmentation maps will be saved.
        no_data_value (int): Value to use for no data areas in the segmentation maps.
        src_crs (int): CRS of points in `sub_data`.
        mask_cloud (bool): Perform cloud masking if True.
        water_mask (bool): Perform water masking if True.
        temporal_tolerance (int): Number of days before and after each historical date for
        the time window.
        history_dates (list[tuple[str, list[str]]]): List of time windows for each
        tile, with acquisition dates.

    Returns:
        A tuple containing the lists of created chips and segmentation maps.
    """
    all_chips = []
    all_seg_maps: list[str | None] = []

    for tile_id in granules_dict.keys():
        band_folder = os.path.join(output_directory, str(tile_id))
        print(f"Processing folder: {band_folder}")

        try:
            datasets, crs = open_mf_jp2_dataset(
                band_folder, history_dates, mask_cloud, water_mask, temporal_tolerance
            )

            if datasets is None or crs is None:
                print(f"No datasets or CRS for folder: {band_folder}")
                continue

            df = gpd.GeoDataFrame(
                sub_data,
                geometry=[Point(xy) for xy in zip(sub_data.x, sub_data.y)],
            )
            df.set_crs(epsg=src_crs, inplace=True)
            df = df.to_crs(crs=crs)

            for group_id, dataset in zip([g[0] for g in history_dates], datasets):
                if dataset is None:
                    print(f"Skipping group ID '{group_id}' - Dataset is None.")
                    continue

                df_filtered = df[
                    (dataset["x"].min().item() <= df["geometry"].x)
                    & (df["geometry"].x <= dataset["x"].max().item())
                    & (dataset["y"].min().item() <= df["geometry"].y)
                    & (df["geometry"].y <= dataset["y"].max().item())
                ]

                if df_filtered.empty:
                    print(
                        f"No valid points for group ID '{group_id}' in dataset bounds."
                    )
                    continue

                chips = []
                seg_maps = []
                date_id = group_id

                n_chips_x = dataset.sizes["x"] // chip_size
                n_chips_y = dataset.sizes["y"] // chip_size
                chip_coords = list(
                    set(get_chip_coords(df_filtered, dataset, chip_size))
                )

                for x, y in chip_coords:
                    if (x >= n_chips_x) or (y >= n_chips_y):
                        continue

                    chip_id = f"{date_id}_{tile_id}_{x}_{y}"
                    chip_name = f"chip_{chip_id}.tif"
                    seg_map_name = f"seg_map_{chip_id}.tif"

                    chip_filename = os.path.join(output_directory, "chips", chip_name)
                    seg_map_filename = os.path.join(
                        output_directory, "seg_maps", seg_map_name
                    )

                    if os.path.exists(chip_filename) or os.path.exists(
                        seg_map_filename
                    ):
                        continue

                    chip = dataset.isel(
                        x=slice(x * chip_size, (x + 1) * chip_size),
                        y=slice(y * chip_size, (y + 1) * chip_size),
                    )
                    if chip.count().values == 0:
                        continue

                    seg_map = create_segmentation_map(chip, df_filtered, no_data_value)
                    if seg_map.where(seg_map != -1).count().values == 0:
                        continue

                    chip.fillna(no_data_value).rio.to_raster(chip_filename)
                    seg_map.rio.to_raster(seg_map_filename)

                    chips.append(chip_name)
                    seg_maps.append(seg_map_name)

                all_chips.extend(chips)
                all_seg_maps.extend(seg_maps)

        except rasterio.errors.RasterioIOError as e:
            logging.error(f"Error {e} when reading dataset containing: {band_folder}")
        except IndexError as e:
            logging.error(f"Error {e} when processing {tile_id}")

    return all_chips, all_seg_maps
