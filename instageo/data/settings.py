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
from typing import Dict, List

import earthaccess
from pydantic_settings import BaseSettings

logging.basicConfig(level=logging.INFO)


def get_access_token() -> str:
    """Configures EarthData credentials based on AIChor environment variables."""
    # Check if we're in test mode
    if os.getenv("TESTING", "false").lower() == "true":
        logging.info("Running in test mode, skipping EarthData authentication")
        return ""

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
            logging.warning(f"EarthData credentials for user {sanitized_username} not found.")

    try:
        edl_token = earthaccess.get_edl_token()
        if edl_token is None:
            logging.warning("EarthData EDL token is None, using empty string")
            return ""
        return edl_token.get("access_token", "")
    except Exception as e:
        logging.warning(f"Failed to get EarthData access token: {e}")
        return ""


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
    """Settings for no-data values to use for HLS, S2 and segmentation maps."""

    HLS: int = 0
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


class HLSBlockSizes(BaseSettings):
    """Settings for block sizes used in COG tiling."""

    X: int = 256
    Y: int = 256


class HLSBandsSettings(BaseSettings):
    """Settings for HLS bands configuration."""

    ASSET: List[str] = ["blue", "green", "red", "nir narrow", "swir 1", "swir 2"]

    NAMEPLATE: Dict[str, Dict[str, str]] = {
        "HLSL30_2.0": {
            "B01": "coastal aerosol",
            "B02": "blue",
            "B03": "green",
            "B04": "red",
            "B05": "nir narrow",
            "B06": "swir 1",
            "B07": "swir 2",
            "B09": "cirrus",
            "B10": "thermal infrared 1",
            "B11": "thermal",
        },
        "HLSS30_2.0": {
            "B01": "coastal aerosol",
            "B02": "blue",
            "B03": "green",
            "B04": "red",
            "B05": "red-edge 1",
            "B06": "red-edge 2",
            "B07": "red-edge 3",
            "B08": "nir broad",
            "B8A": "nir narrow",
            "B09": "water vapor",
            "B10": "cirrus",
            "B11": "swir 1",
            "B12": "swir 2",
        },
    }


class HLSAPISettings(BaseSettings):
    """Settings for HLS API configuration."""

    URL: str = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
    COLLECTIONS: List[str] = ["HLSL30_2.0", "HLSS30_2.0"]


class S2BlockSizes(BaseSettings):
    """Settings for block sizes used in COG tiling."""

    X: int = 256
    Y: int = 256


class S2APISettings(BaseSettings):
    """Settings for Sentinel-2 API configuration."""

    URL: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    COLLECTIONS: List[str] = ["sentinel-2-l2a"]


class S2BandsSettings(BaseSettings):
    """Settings for Sentinel-2 bands configuration."""

    ASSET: List[str] = ["blue", "green", "red", "nir narrow", "swir 1", "swir 2"]

    NAMEPLATE: Dict[str, Dict[str, str]] = {
        "sentinel-2-l2a": {
            "B01": "coastal aerosol",  # 443 nm
            "B02": "blue",  # 490 nm
            "B03": "green",  # 560 nm
            "B04": "red",  # 665 nm
            "B05": "red-edge 1",  # 705 nm
            "B06": "red-edge 2",  # 740 nm
            "B07": "red-edge 3",  # 783 nm
            "B08": "nir broad",  # 842 nm
            "B8A": "nir narrow",  # 865 nm
            "B09": "water vapor",  # 945 nm
            "B10": "cirrus",  # 1375 nm
            "B11": "swir 1",  # 1610 nm
            "B12": "swir 2",  # 2190 nm
        }
    }


class DataPipelineSettings(BaseSettings):
    """Settings for data pipeline configuration."""

    BATCH_SIZE: int = 16  # Number of records to process at a time
    METADATA_SEARCH_RATELIMIT: int = 10  # Number of metadata searches per minute
    COG_DOWNLOAD_RATELIMIT: int = 30  # Number of COG downloads per minute
