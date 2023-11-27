import geopandas as gpd
import pytest
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import Affine

from instageo.data.geo_utils import (
    get_transform_crs,
    open_mf_tiff_dataset,
    open_tiff_as_dataarray,
    read_csv_gdf,
)


def test_read_csv_gdf_4326():
    test_file = "tests/data/sample.csv"
    gdf = read_csv_gdf(test_file)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs.to_epsg() == 4326


def test_read_csv_gdf_32613():
    test_file = "tests/data/sample.csv"
    gdf = read_csv_gdf(test_file, dst_crs=32613)
    assert gdf.crs.to_epsg() == 32613


def test_read_csv_gdf_invalid_file():
    with pytest.raises(FileNotFoundError):
        read_csv_gdf("non_existent_file.csv")


def test_open_tiff_as_dataarray():
    file_path = "tests/data/sample.tif"
    band_name = "test_band"

    result = open_tiff_as_dataarray(file_path, band_name)

    assert isinstance(result, xr.Dataset)
    assert band_name in result
    assert "crs" in result.attrs
    assert "transform" in result.attrs


def test_open_mf_tiff_dataset():
    band_files = {
        "band1": "tests/data/sample.tif",
        "band2": "tests/data/sample.tif",
    }

    result = open_mf_tiff_dataset(band_files)

    assert isinstance(result, xr.Dataset)
    assert all(band in result for band in band_files)


def test_get_transform_crs():
    file_path = "tests/data/sample.tif"
    transform, crs = get_transform_crs(file_path)
    assert isinstance(transform, Affine)
    assert isinstance(crs, CRS)
