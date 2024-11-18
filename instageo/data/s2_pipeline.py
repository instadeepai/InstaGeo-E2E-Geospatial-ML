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
import requests  # type: ignore
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
    history_dates: dict,
) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieve Sentinel-2 Tiles Metadata.

    Given a tile_id, start_date, and a time window, this function fetches all
    Sentinel-2 granules available for this tile_id in this time window.

    Args:
        tile_df (pd.DataFrame): A dataframe containing tile_id, start_date, and geographical
        boundaries.
        cloud_coverage (float): Maximum acceptable cloud coverage for each granule.
        temporal_tolerance (int): Number of days before and after each historical date for
        the time window.
        history_dates (dict): A dictionary of history dates for each tile.


    Returns:
        A dictionary mapping tile_id to a list of available Sentinel-2 granules.
    """
    granules_dict: Dict[str, List[Dict[str, Any]]] = {}
    unique_full_tile_ids = set()

    for _, row in tile_df.iterrows():
        lon_min, lon_max, lat_min, lat_max = (
            row["lon_min"],
            row["lon_max"],
            row["lat_min"],
            row["lat_max"],
        )

        for _, date_list in history_dates:
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

                    granule_info = {
                        "full_tile_id": full_tile_id,
                        "tile_id": tile_id_extracted,
                        "cloudCover": feature["properties"]["cloudCover"],
                        "download_link": feature["properties"]["services"]["download"][
                            "url"
                        ],
                        "thumbnail": feature["properties"]["thumbnail"],
                        "acquisition_date": tile_acquisition_date,
                    }

                    if tile_id_extracted:
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
    time_windows: List[Tuple[str, List[str]]],
) -> None:
    """Filter best product.

    Filters the best product (based on highest valid pixel count in the SCL band)
    in each time window for a tile, and removes other products' bands.

    Args:
        tile_name (str): Name of the tile.
        tile_products (list): List of product dictionaries for the tile.
        output_directory (str): The folder where all the tile folders are stored.
        time_windows (List[Tuple[str, List[str]]]): List of time windows for each
        tile, with acquisition dates.

    Returns:
    None
    """
    folder_path = os.path.join(output_directory, tile_name)

    tile_time_window = [dates for name, dates in time_windows if name == tile_name]

    if tile_time_window:
        sorted_dates = sorted(
            tile_time_window[0], key=lambda date: datetime.strptime(date, "%Y-%m-%d")
        )

        for i in range(len(sorted_dates) - 1):
            start_date = datetime.strptime(sorted_dates[i], "%Y-%m-%d")
            end_date = datetime.strptime(sorted_dates[i + 1], "%Y-%m-%d")

            best_product = None
            best_valid_pixel_count = -1

            for product in tile_products:
                acquisition_date = datetime.strptime(
                    product["acquisition_date"], "%Y-%m-%d"
                )

                if start_date <= acquisition_date <= end_date:
                    scl_band_path = find_scl_file(
                        folder_path, tile_name, acquisition_date
                    )

                    if scl_band_path:
                        valid_pixel_count = count_valid_pixels(scl_band_path)

                        if valid_pixel_count > best_valid_pixel_count:
                            best_product = product
                            best_valid_pixel_count = valid_pixel_count
                    else:
                        print(f"SCL band not found for {product['full_tile_id']}")

            if best_product:
                print(
                    f"Best product for {tile_name} between {start_date.date()} and "
                    f"{end_date.date()} is {best_product['full_tile_id']} with "
                    f"{best_valid_pixel_count} valid pixels."
                )

                best_product_acquisition_date = datetime.strptime(
                    best_product["acquisition_date"], "%Y-%m-%d"
                ).strftime("%Y%m%d")

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
                                print(f"Removed file: {item_path}")
            else:
                print(
                    f"No valid product found for {tile_name} between {start_date.date()} "
                    f"and {end_date.date()}."
                )


def create_and_save_chips_with_seg_maps_s2(
    band_folder: str,
    df: pd.DataFrame,
    chip_size: int,
    output_directory: str,
    no_data_value: int,
    src_crs: int,
    num_steps: int,
    mask_cloud: bool,
    water_mask: bool,
) -> tuple[list[str], list[str | None]]:
    """Chip Creator.

    Create chips and corresponding segmentation maps from a HLS tile and save them to
    an output directory.

    Args:
        band_folder (str): S2 tile filepath.
        df (pd.DataFrame): DataFrame containing the data for segmentation maps.
        chip_size (int): Size of each chip.
        output_directory (str): Directory where the chips and segmentation maps will be
            saved.
        no_data_value (int): Value to use for no data areas in the segmentation maps.
        src_crs (int): CRS of points in `df`
        mask_cloud (bool): Perform cloud masking if True.
        water_mask (bool): Perform water masking if True.

    Returns:
        A tuple containing the lists of created chips and segmentation maps.
    """
    ds, crs = open_mf_jp2_dataset(band_folder, num_steps, mask_cloud, water_mask)
    if ds is None:
        raise ValueError("The band folder could not be loaded.")

    df = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.x, df.y)])
    df.set_crs(epsg=src_crs, inplace=True)
    df = df.to_crs(crs=crs)

    df = df[
        (ds["x"].min().item() <= df["geometry"].x)
        & (df["geometry"].x <= ds["x"].max().item())
        & (ds["y"].min().item() <= df["geometry"].y)
        & (df["geometry"].y <= ds["y"].max().item())
    ]
    os.makedirs(output_directory, exist_ok=True)

    # tile_name_splits = hls_tile_dict["tiles"]["B02_0"].split(".")
    # tile_id = f"{tile_name_splits[1]}_{tile_name_splits[2]}_{tile_name_splits[3]}"

    tile_id = os.path.basename(band_folder)

    date_id = df.iloc[0]["date"].strftime("%Y%m%d")
    chips = []
    seg_maps: list[str | None] = []
    n_chips_x = ds.sizes["x"] // chip_size
    n_chips_y = ds.sizes["y"] // chip_size
    chip_coords = list(set(get_chip_coords(df, ds, chip_size)))
    for x, y in chip_coords:
        if (x >= n_chips_x) or (y >= n_chips_y):
            continue
        chip_id = f"{date_id}_{tile_id}_{x}_{y}"
        chip_name = f"chip_{chip_id}.tif"
        seg_map_name = f"seg_map_{chip_id}.tif"

        chip_filename = os.path.join(output_directory, "chips", chip_name)
        seg_map_filename = os.path.join(output_directory, "seg_maps", seg_map_name)
        if os.path.exists(chip_filename) or os.path.exists(seg_map_filename):
            continue

        chip = ds.isel(
            x=slice(x * chip_size, (x + 1) * chip_size),
            y=slice(y * chip_size, (y + 1) * chip_size),
        )
        if chip.count().values == 0:
            continue
        seg_map = create_segmentation_map(chip, df, no_data_value)
        if seg_map.where(seg_map != -1).count().values == 0:
            continue
        seg_maps.append(seg_map_name)
        seg_map.rio.to_raster(seg_map_filename)
        chip = chip.fillna(no_data_value)
        chips.append(chip_name)
        chip.band_data.rio.to_raster(chip_filename)
    return chips, seg_maps
