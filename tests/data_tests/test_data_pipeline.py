import os
import shutil

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point

from instageo.data.geo_utils import apply_mask, decode_fmask_value, open_mf_tiff_dataset


def test_open_mf_tiff_dataset():
    band_files = {
        "tiles": {
            "band1": "tests/data/sample.tif",
            "band2": "tests/data/sample.tif",
        },
        "fmasks": {
            "band1": "tests/data/fmask.tif",
            "band2": "tests/data/fmask.tif",
        },
    }

    result, _, crs = open_mf_tiff_dataset(band_files, load_masks=False)
    assert isinstance(result, xr.Dataset)
    assert isinstance(crs, CRS)
    assert crs == 32613
    assert result["band_data"].shape == (2, 224, 224)


def test_open_mf_tiff_dataset_cloud_mask():
    band_files = {
        "tiles": {
            "band1": "tests/data/sample.tif",
            "band2": "tests/data/sample.tif",
        },
        "fmasks": {
            "band1": "tests/data/fmask.tif",
            "band2": "tests/data/fmask.tif",
        },
    }
    result_no_mask, _, crs = open_mf_tiff_dataset(band_files, load_masks=False)
    num_points = result_no_mask.band_data.count().values.item()
    result_with_mask, mask_ds, crs = open_mf_tiff_dataset(band_files, load_masks=True)
    result_with_mask = apply_mask(
        result_with_mask,
        mask_ds.band_data,
        -1,
        masking_strategy="any",
    )
    fmask = xr.open_dataset("tests/data/fmask.tif")
    cloud_mask = decode_fmask_value(fmask, 1)
    num_clouds = cloud_mask.where(cloud_mask == 1).band_data.count().values.item()
    assert (
        result_with_mask.band_data.where(result_with_mask.band_data != -1)
        .count()
        .values.item()
        == num_points - 2 * num_clouds
    )

    assert isinstance(result_with_mask, xr.Dataset)
    assert isinstance(crs, CRS)
    assert crs == 32613
    assert result_with_mask["band_data"].shape == (2, 224, 224)


@pytest.fixture
def setup_and_teardown_output_dir():
    output_dir = "/tmp/test_hls"
    os.makedirs(output_dir, exist_ok=True)
    yield
    shutil.rmtree(output_dir)


def test_get_tiles(observation_data):
    hls_tiles = get_tiles(data=observation_data, min_count=1)
    assert list(hls_tiles["mgrs_tile_id"]) == [
        "38PMB",
        "38PMB",
        "38PPB",
        "39QTT",
        "30RYS",
        "38QMC",
        "39QUT",
        "38PMB",
        "39QUT",
        "38PMB",
    ]


def test_get_chip_coords():
    df = pd.read_csv("tests/data/sample_4326.csv")
    df = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.x, df.y)])
    df.set_crs(epsg=4326, inplace=True)
    df = df.to_crs(crs=32613)

    ds = xr.open_dataset("tests/data/HLS.S30.T38PMB.2022145T072619.v2.0.B02.tif")
    chip_coords = get_chip_coords(df, ds, 64)
    assert chip_coords == [
        (2, 0),
        (0, 3),
        (2, 2),
        (0, 3),
        (2, 0),
        (3, 2),
        (2, 3),
        (0, 3),
        (2, 3),
        (1, 2),
    ]


def test_get_tile_info(observation_data):
    hls_tiles = get_tiles(observation_data, min_count=3)
    tiles_info, tile_queries = get_tile_info(hls_tiles, num_steps=3, temporal_step=5)
    pd.testing.assert_frame_equal(
        tiles_info,
        pd.DataFrame(
            {
                "tile_id": ["38PMB"],
                "min_date": ["2022-05-29"],
                "max_date": ["2022-06-09"],
                "lon_min": [44.451435],
                "lon_max": [44.744167],
                "lat_min": [15.099767],
                "lat_max": [15.287778],
            }
        ),
        check_like=True,
    )
    assert tile_queries == [
        ("38PMB", ["2022-06-08", "2022-06-03", "2022-05-29"]),
        ("38PMB", ["2022-06-08", "2022-06-03", "2022-05-29"]),
        ("38PMB", ["2022-06-08", "2022-06-03", "2022-05-29"]),
        ("38PMB", ["2022-06-09", "2022-06-04", "2022-05-30"]),
    ]
