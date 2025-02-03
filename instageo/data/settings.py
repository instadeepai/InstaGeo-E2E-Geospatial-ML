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

import earthaccess
from pydantic_settings import BaseSettings


class GDALOptions(BaseSettings):
    """GDAL Settings used in the context of Dask client."""

    CPL_VSIL_CURL_ALLOWED_EXTENSIONS: str = ".tif"
    GDAL_HTTP_AUTH: str = "BEARER"
    GDAL_HTTP_BEARER: str = earthaccess.get_edl_token().get("access_token")
    GDAL_DISABLE_READDIR_ON_OPEN: str = "EMPTY_DIR"
    GDAL_HTTP_MAX_RETRY: str = "10"
    GDAL_HTTP_RETRY_DELAY: str = "0.5"
    GDAL_CACHEMAX: int = 1024  # 1 GB
    GDAL_SWATH_SIZE: int = 16777216  # 16 MB
    CPL_VSIL_CURL_CACHE_SIZE: int = 67108864  # 64 MB


class NoDataValues(BaseSettings):
    """Settings for no-data values to use for HLS, S2 and segmentation maps."""

    HLS: int = -9999
    S2: int = 0
    SEG_MAP: int = -1
