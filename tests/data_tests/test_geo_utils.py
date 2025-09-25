import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Polygon, box

from instageo.data.geo_utils import (
    create_grid_polygons,
    get_extent,
    get_polygon_tile_ids,
    make_valid_bbox,
    slice_xr_dataset,
)


def test_get_extent():
    # Create a sample xarray dataset
    data = np.random.rand(4, 4)
    lats = np.linspace(0, 3, 4)
    lons = np.linspace(0, 3, 4)
    ds = xr.DataArray(data, dims=["y", "x"], coords={"y": lats, "x": lons})

    # Test the function
    lon_min, lat_min, lon_max, lat_max = get_extent(ds)

    assert lat_min == 0
    assert lat_max == 3
    assert lon_min == 0
    assert lon_max == 3


def test_make_valid_bbox():
    # Test with a valid bbox
    valid_bbox = make_valid_bbox(0, 0, 1, 1)
    assert valid_bbox == (0, 0, 1, 1)

    # Test with an invalid bbox (min > max)
    invalid_bbox = make_valid_bbox(1, 1, 0, 0)
    # The function should return a valid bbox with a small buffer
    assert len(invalid_bbox) == 4
    # The coordinates should be in the correct order (min, min, max, max)
    assert invalid_bbox[0] < invalid_bbox[2]  # min_lon < max_lon
    assert invalid_bbox[1] < invalid_bbox[3]  # min_lat < max_lat

    # Test with a point (zero area) - should trigger buffer case
    point_bbox = make_valid_bbox(1, 1, 1, 1)
    assert len(point_bbox) == 4
    assert point_bbox[0] < point_bbox[2]  # Should have area after buffering
    assert point_bbox[1] < point_bbox[3]


def test_get_polygon_tile_ids():
    # Test with a simple polygon
    polygon = box(-1, -1, 1, 1)  # Simple square polygon
    tile_ids = get_polygon_tile_ids(polygon)

    # Should return a set of MGRS tile IDs
    assert isinstance(tile_ids, set)
    assert len(tile_ids) >= 1  # At least one tile ID

    # All tile IDs should be strings
    for tile_id in tile_ids:
        assert isinstance(tile_id, str)
        assert len(tile_id) > 0  # Non-empty string

    # Test with a smaller polygon
    small_polygon = box(0, 0, 0.1, 0.1)
    small_tile_ids = get_polygon_tile_ids(small_polygon)
    assert isinstance(small_tile_ids, set)
    assert len(small_tile_ids) >= 1


def test_slice_xr_dataset():
    # Create a sample xarray dataset
    data = np.random.rand(4, 4)
    lats = np.linspace(0, 3, 4)
    lons = np.linspace(0, 3, 4)
    ds = xr.Dataset(data_vars={"data": (["y", "x"], data)}, coords={"y": lats, "x": lons})

    # Set up the rio attributes
    ds.rio.set_crs("EPSG:4326", inplace=True)
    ds.rio.transform((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))

    # Create a polygon for slicing
    lon_min, lat_min, lon_max, lat_max = (0.5, 0.5, 2.5, 2.5)
    polygon = Polygon(
        [
            (lon_min, lat_min),
            (lon_max, lat_min),
            (lon_max, lat_max),
            (lon_min, lat_max),
        ]
    )

    # Test slicing without chip_size
    sliced = slice_xr_dataset(ds, polygon)
    print(sliced)
    assert sliced is not None
    assert sliced.data.shape[0] <= ds.data.shape[0]
    assert sliced.data.shape[1] <= ds.data.shape[1]

    # Test slicing with chip_size
    sliced = slice_xr_dataset(ds, polygon, chip_size=2)
    assert sliced is not None
    assert sliced.data.shape[0] == 2
    assert sliced.data.shape[1] == 2

    # Test with invalid geometry
    lon_min, lat_min, lon_max, lat_max = (10, 10, 11, 11)
    non_intersecting_polygon = Polygon(
        [
            (lon_min, lat_min),
            (lon_max, lat_min),
            (lon_max, lat_max),
            (lon_min, lat_max),
        ]
    )
    sliced = slice_xr_dataset(ds, non_intersecting_polygon)
    assert sliced is None


def test_create_grid_polygons():
    # Test with simple bbox list
    bbox_list = [[-1, -1, 1, 1]]  # Single bbox
    date = "2023-01-01"
    chip_size = 2
    spatial_resolution = 1
    crs = 4326

    result = create_grid_polygons(bbox_list, date, chip_size, spatial_resolution, crs)

    # Should return a GeoDataFrame
    assert isinstance(result, gpd.GeoDataFrame)

    # Should have expected columns
    expected_columns = [
        "label_filename",
        "date",
        "geometry",
        "geometry_4326",
        "mgrs_tile_id",
    ]
    for col in expected_columns:
        assert col in result.columns

    # Should have at least one row
    assert len(result) >= 1

    # All geometries should be valid
    assert result["geometry"].is_valid.all()

    # All dates should match input
    assert (result["date"] == date).all()

    # Test with empty bbox list
    empty_result = create_grid_polygons([], date, chip_size, spatial_resolution, crs)
    assert isinstance(empty_result, gpd.GeoDataFrame)
    assert len(empty_result) == 0
