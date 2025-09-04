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
from typing import Callable, Tuple

import geopandas as gpd
import mgrs
import numpy as np
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

    # Ensure coordinates are in correct order (min <= max)
    actual_lon_min = min(lon_min, lon_max)
    actual_lon_max = max(lon_min, lon_max)
    actual_lat_min = min(lat_min, lat_max)
    actual_lat_max = max(lat_min, lat_max)

    # Try to create a box with the corrected coordinates
    bbox = box(actual_lon_min, actual_lat_min, actual_lon_max, actual_lat_max)

    if bbox.is_valid and bbox.area > 0:
        return actual_lon_min, actual_lat_min, actual_lon_max, actual_lat_max
    else:
        # If the box is invalid or has zero area, add a buffer
        return bbox.buffer(epsilon).bounds


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
        if clipped.data.size == 0:
            return None
        return clipped
    except rasterio.RasterioIOError:
        print("No data found in bounds. Skipping this geometry.")
        return None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def get_extent(  # type: ignore[no-untyped-call]
    dataset: xr.Dataset,
) -> Tuple[float, float, float, float]:
    """Return the spatial extent (lon_min, lat_min, lon_max, lat_max) of an xarray Dataset.

    Args:
        dataset: xr.Dataset

    Returns:
        tuple of float: The minimum longitude, minimum latitude, maximum longitude,
        and maximum latitude.
    """
    lat_min = dataset["y"].min().item()
    lat_max = dataset["y"].max().item()
    lon_min = dataset["x"].min().item()
    lon_max = dataset["x"].max().item()
    return lon_min, lat_min, lon_max, lat_max


def create_grid_polygons(
    bbox_list: list[list[float]],
    date: str,
    chip_size: int,
    spatial_resolution: int,
    crs: int,
) -> gpd.GeoDataFrame:
    """Create a grid of polygons from a list of bounding boxes.

    Args:
        bbox_list: list of [lon_min, lat_min, lon_max, lat_max]
        date: date parameter
        chip_size: size of each chip in pixels
        spatial_resolution: spatial resolution in crs units per pixel
        crs: crs of the bounding boxes
    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the polygons, labels, and date
    """
    obsv_records = []

    for bbox in bbox_list:
        lon_min, lat_min, lon_max, lat_max = bbox

        lons = np.arange(lon_min, lon_max, spatial_resolution)
        lats = np.arange(lat_min, lat_max, spatial_resolution)

        # Create xarray Dataset with empty/minimal data
        ds = xr.Dataset(
            {
                "data": (
                    ["y", "x"],
                    np.zeros((len(lats), len(lons))),
                ),  # Empty data array
            },
            coords={
                "x": lons,
                "y": lats,
            },
            attrs={"crs": f"EPSG:{crs}"},
        )
        # Calculate the number of chips in each dimension
        n_chips_x = ds.sizes["x"] // chip_size
        n_chips_y = ds.sizes["y"] // chip_size
        chip_count = 0
        for x in range(n_chips_x):
            for y in range(n_chips_y):
                label_chip_filename = f"label_x{x}_y{y}_{date}.tif"
                # Extract the chip
                label_chip = ds.isel(
                    x=slice(x * chip_size, (x + 1) * chip_size),
                    y=slice(y * chip_size, (y + 1) * chip_size),
                )

                label_geometry = box(*get_extent(label_chip))
                obsv_records.append([label_chip_filename, date, label_geometry])
                chip_count += 1

    obsv_records_gdf = gpd.GeoDataFrame(
        obsv_records,
        columns=["label_filename", "date", "geometry"],
        geometry="geometry",
        crs=f"EPSG:{crs}",
    )
    obsv_records_gdf["geometry_4326"] = obsv_records_gdf["geometry"].to_crs("EPSG:4326")
    obsv_records_gdf["mgrs_tile_id"] = obsv_records_gdf["geometry_4326"].map(
        get_polygon_tile_ids
    )
    obsv_records_gdf = obsv_records_gdf.explode("mgrs_tile_id", ignore_index=True)
    return obsv_records_gdf
