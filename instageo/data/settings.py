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

"""Utility Pydantic settings."""

import logging
import os
from typing import List

import earthaccess
from pydantic_settings import BaseSettings

logging.basicConfig(level=logging.INFO)


def get_access_token() -> str:
    """Configures EarthData credentials based on AIChor environment variables."""
    author_email = os.getenv("VCS_AUTHOR_EMAIL")
    if author_email:
        author_username, _ = author_email.split("@")
        sanitized_username = author_username.replace("-", "_").replace(".", "_").upper()

        username_key = f"{sanitized_username}__EARTHDATA_USERNAME"
        password_key = f"{sanitized_username}__EARTHDATA_PASSWORD"

        if username_key in os.environ and password_key in os.environ:
            logging.info(f"EarthData credentials for user {sanitized_username} found.")
            os.environ["EARTHDATA_USERNAME"] = os.environ[username_key]
            os.environ["EARTHDATA_PASSWORD"] = os.environ[password_key]
        else:
            logging.warning(
                f"EarthData credentials for user {sanitized_username} not found."
            )
    return earthaccess.get_edl_token().get("access_token")


class GDALOptions(BaseSettings):
    """GDAL Settings used in the context of Dask client."""

    CPL_VSIL_CURL_ALLOWED_EXTENSIONS: str = ".tif"
    GDAL_HTTP_AUTH: str = "BEARER"
    GDAL_HTTP_BEARER: str = get_access_token()
    GDAL_DISABLE_READDIR_ON_OPEN: str = "EMPTY_DIR"
    GDAL_HTTP_MAX_RETRY: str = "10"
    GDAL_HTTP_RETRY_DELAY: str = "0.5"
    GDAL_CACHEMAX: int = 1024  # 1 GB
    GDAL_SWATH_SIZE: int = 16777216  # 16 MB
    CPL_VSIL_CURL_CACHE_SIZE: int = 67108864  # 64 MB
    GDAL_HTTP_COOKIEFILE: str = "/tmp/cookies.txt"


class NoDataValues(BaseSettings):
    """Settings for no-data values to use for HLS, S2, S1 and segmentation maps."""

    HLS: int = -9999
    S2: int = 0
    S1: int = -1
    SEG_MAP: int = -1


class S2Bands(BaseSettings):
    """Settings for S2 band values."""

    VALUES: List[str] = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        # Cirrus band - "B10" is excluded from the L2A product as it doesn't contain any
        # bottom of the atmosphere information.
        "B11",
        "B12",
    ]
