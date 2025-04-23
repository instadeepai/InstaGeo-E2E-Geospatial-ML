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
"""Geospatial Utility Functions."""
from typing import Callable

import mgrs
import rasterio
import xarray as xr
from pyproj import Transformer
from shapely.geometry import Polygon, box


def get_polygon_tile_ids(polygon: Polygon) -> set[str]:
    """Get MGRS tile IDs for Polygon edges.

    Args:
        polygon (Polygon): A shapely polygon.

    Returns:
        set[str]: A set of MGRS tile IDs the Polygon overlaps.
    """
    lon_min, lat_min, lon_max, lat_max = polygon.bounds

    mgrs_object = mgrs.MGRS()
    get_mgrs_tile_id: Callable[[float, float], str] = lambda x, y: mgrs_object.toMGRS(
        y, x, MGRSPrecision=0
    )
    tile_id_tl = get_mgrs_tile_id(lon_min, lat_min)
    tile_id_lr = get_mgrs_tile_id(lon_max, lat_max)
    tile_id_tr = get_mgrs_tile_id(lon_min, lat_max)
    tile_id_ll = get_mgrs_tile_id(lon_max, lat_min)

    tile_ids = {tile_id_tl, tile_id_lr, tile_id_tr, tile_id_ll}

    return tile_ids


def make_valid_bbox(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float
) -> tuple[float, float, float, float]:
    """Create a valid bounding box to search for tiles.

    The purpose of this function is to still be able to extract data through
    earthaccess even given just a single observation in a tile (min_count = 1).
    When the number of observations in a tile is 1, or if we only have aligned
    observations, the lon_min, lat_min, lon_max, lat_max extracted from those
    won't produce a valid bounding box. Thus, we attempt to create a small buffer
    around the observation(s) to produce a valid bounding box.

    Args:
        lon_min (float): Minimum longitude
        lat_min (float): Minimum latitude
        lon_max (float): Maximum longitude
        lat_max (float): Maximum latitude

    Returns:
        A tuple of coordinates to use for a bounding box

    """
    epsilon = 1e-3
    if box(lon_min, lat_min, lon_max, lat_max).is_valid:
        return lon_min, lat_min, lon_max, lat_max
    else:
        return box(lon_min, lat_min, lon_max, lat_max).buffer(epsilon).bounds


def slice_xr_dataset(
    dataset: xr.Dataset | xr.DataArray,
    geometry: Polygon,
    geometry_crs: int | None = None,
    chip_size: int | None = None,
) -> xr.Dataset | xr.DataArray:
    """Crops an Xarray dataset to geometry bounds.

    The `chip_size` is used to ensure an exact size is returned because the size can change due
    to reprojection.

    Args:
        dataset (xarray.Dataset): Xarray dataset
        geometry (shapely.Polygon): A shapely polygon
        geometry_crs (int, optional): CRS EPSG code. Defaults to None.
        chip_size (int, optional): Size of the chip. Defaults to None.

    Returns:
        xr.Dataset: cropped dataset that lies within the bound of the Polygon
    """
    try:
        minx, miny, maxx, maxy = geometry.bounds
        if geometry_crs is not None:
            transformer = Transformer.from_crs(
                geometry_crs, dataset.rio.crs, always_xy=True
            )
            minx, miny = transformer.transform(minx, miny)
            maxx, maxy = transformer.transform(maxx, maxy)

        affine_transform = rasterio.transform.AffineTransformer(dataset.rio.transform())
        row_min, col_min = affine_transform.rowcol(minx, miny)
        row_max, col_max = affine_transform.rowcol(maxx, maxy)

        row_min, row_max = sorted([row_min, row_max])
        col_min, col_max = sorted([col_min, col_max])

        # re-projection adds a few pixels to the geometry bounds resulting in a chip_size slightly
        # larger than the original size of the geometry
        clipped = dataset.isel(
            x=slice(col_min, col_min + chip_size if chip_size else col_max),
            y=slice(row_min, row_min + chip_size if chip_size else row_max),
        )

        return clipped
    except rasterio.RasterioIOError:
        print("No data found in bounds. Skipping this geometry.")
        return None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None
