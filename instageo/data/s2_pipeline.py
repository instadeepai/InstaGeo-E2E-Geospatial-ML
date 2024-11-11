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

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import requests  # type: ignore

from instageo.data.geo_utils import get_tile_info


def retrieve_sentinel2_metadata(
    tile_df: pd.DataFrame,
    cloud_coverage: float,
    num_steps: int,
    temporal_step: int,
    temporal_tolerance: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieve Sentinel-2 Tiles Metadata.

    Given a tile_id, start_date, and a time window, this function fetches all
    Sentinel-2 granules available for this tile_id in this time window.

    Args:
        tile_df (pd.DataFrame): A dataframe containing tile_id, start_date, and geographical
        boundaries.
        cloud_coverage (float): Maximum acceptable cloud coverage for each granule.
        num_steps (int): Number of historical steps to calculate from the start date.
        temporal_step (int): Number of days between each historical date.
        temporal_tolerance (int): Number of days before and after each historical date for the time
        window.

    Returns:
        A dictionary mapping tile_id to a list of available Sentinel-2 granules.
    """
    tile_df, history_dates = get_tile_info(tile_df, num_steps, temporal_step)
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
                    # logging.error(f"Failed to retrieve data: {response.status_code}")
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
                    # acquisition_date = (
                    #     datetime.strptime(acquisition_date.group(1), "%Y%m%d").strftime(
                    #         "%Y-%m-%d"
                    #     )
                    #     if acquisition_date
                    #     else None
                    # )

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
