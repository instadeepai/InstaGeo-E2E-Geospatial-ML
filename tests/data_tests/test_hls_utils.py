import datetime
import os
import shutil

import pandas as pd
import pytest
import xarray as xr
from rasterio.crs import CRS

from instageo.data.hls_utils import (
    decode_fmask_value,
    find_closest_tile,
    open_mf_tiff_dataset,
    parse_date_from_entry,
    retrieve_hls_metadata,
)


@pytest.fixture
def setup_and_teardown_output_dir():
    output_dir = "/tmp/test_hls"
    os.makedirs(output_dir, exist_ok=True)
    yield
    shutil.rmtree(output_dir)


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

    result, crs = open_mf_tiff_dataset(band_files, mask_cloud=False, water_mask=False)
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
    result_no_mask, crs = open_mf_tiff_dataset(
        band_files, mask_cloud=False, water_mask=False
    )
    num_points = result_no_mask.band_data.count().values.item()
    result_with_mask, crs = open_mf_tiff_dataset(
        band_files, mask_cloud=True, water_mask=False
    )
    fmask = xr.open_dataset("tests/data/fmask.tif")
    cloud_mask = decode_fmask_value(fmask, 1)
    num_clouds = cloud_mask.where(cloud_mask == 1).band_data.count().values.item()
    assert (
        result_with_mask.band_data.count().values.item() == num_points - 2 * num_clouds
    )
    assert isinstance(result_with_mask, xr.Dataset)
    assert isinstance(crs, CRS)
    assert crs == 32613
    assert result_with_mask["band_data"].shape == (2, 224, 224)


def test_retrieve_hls_metadata():
    tile_info = pd.DataFrame(
        {
            "tile_id": ["38PMB"],
            "min_date": ["2022-05-29"],
            "max_date": ["2022-06-09"],
            "lon_min": [44.451435],
            "lon_max": [44.744167],
            "lat_min": [15.099767],
            "lat_max": [15.287778],
        }
    )
    tile_database = retrieve_hls_metadata(tile_info)
    assert tile_database == {
        "38PMB": [
            "HLS.S30.T38PMB.2022150T072621.v2.0",
            "HLS.S30.T38PMB.2022152T071619.v2.0",
            "HLS.L30.T38PMB.2022154T072604.v2.0",
            "HLS.L30.T38PMB.2022155T071923.v2.0",
            "HLS.S30.T38PMB.2022155T072619.v2.0",
            "HLS.S30.T38PMB.2022157T071631.v2.0",
            "HLS.S30.T38PMB.2022160T072621.v2.0",
        ]
    }


@pytest.mark.parametrize(
    "value, position, result",
    [
        (100, 0, 0),
        (100, 1, 0),
        (100, 2, 1),
        (100, 3, 0),
        (100, 4, 0),
        (100, 5, 1),
        (100, 6, 1),
        (100, 7, 0),
    ],
)
def test_decode_fmask_value(value, position, result):
    assert decode_fmask_value(value, position) == result


def test_find_closest_tile():
    tile_database = {
        "30RYS": [
            "HLS.L30.T30RYS.2022140T102748.v2.0.",
            "HLS.S30.T30RYS.2022142T103629.v2.0.",
            "HLS.S30.T30RYS.2022144T102611.v2.0.",
            "HLS.L30.T30RYS.2022147T103357.v2.0.",
            "HLS.S30.T30RYS.2022147T103631.v2.0.",
            "HLS.L30.T30RYS.2022148T102722.v2.0.",
            "HLS.S30.T30RYS.2022149T102559.v2.0.",
            "HLS.S30.T30RYS.2022152T103629.v2.0.",
            "HLS.S30.T30RYS.2022154T102611.v2.0.",
            "HLS.L30.T30RYS.2022155T103334.v2.0.",
            "HLS.L30.T30RYS.2022156T102755.v2.0.",
            "HLS.S30.T30RYS.2022157T103631.v2.0.",
            "HLS.S30.T30RYS.2022159T102559.v2.0.",
        ],
        "38PMB": [
            "HLS.L30.T38PMB.2022139T071922.v2.0.",
            "HLS.S30.T38PMB.2022140T072621.v2.0.",
            "HLS.S30.T38PMB.2022142T071619.v2.0.",
            "HLS.S30.T38PMB.2022145T072619.v2.0.",
            "HLS.L30.T38PMB.2022146T072532.v2.0.",
            "HLS.L30.T38PMB.2022147T071947.v2.0.",
            "HLS.S30.T38PMB.2022147T071621.v2.0.",
            "HLS.S30.T38PMB.2022150T072621.v2.0.",
            "HLS.S30.T38PMB.2022152T071619.v2.0.",
            "HLS.L30.T38PMB.2022154T072604.v2.0.",
            "HLS.L30.T38PMB.2022155T071923.v2.0.",
            "HLS.S30.T38PMB.2022155T072619.v2.0.",
            "HLS.S30.T38PMB.2022157T071631.v2.0.",
            "HLS.S30.T38PMB.2022160T072621.v2.0.",
            "HLS.L30.T38PMB.2022162T072536.v2.0.",
            "HLS.S30.T38PMB.2022162T071619.v2.0.",
        ],
        "38PNB": [
            "HLS.S30.T38PNB.2022142T071619.v2.0.",
            "HLS.L30.T38PNB.2022147T071947.v2.0.",
            "HLS.S30.T38PNB.2022147T071621.v2.0.",
            "HLS.L30.T38PNB.2022148T071311.v2.0.",
            "HLS.S30.T38PNB.2022152T071619.v2.0.",
            "HLS.L30.T38PNB.2022155T071923.v2.0.",
            "HLS.L30.T38PNB.2022156T071344.v2.0.",
            "HLS.S30.T38PNB.2022157T071631.v2.0.",
        ],
        "38PPB": [
            "HLS.L30.T38PPB.2022139T071922.v2.0.",
            "HLS.L30.T38PPB.2022140T071337.v2.0.",
            "HLS.S30.T38PPB.2022142T071619.v2.0.",
            "HLS.L30.T38PPB.2022147T071947.v2.0.",
            "HLS.S30.T38PPB.2022147T071621.v2.0.",
            "HLS.L30.T38PPB.2022148T071311.v2.0.",
            "HLS.S30.T38PPB.2022152T071619.v2.0.",
            "HLS.L30.T38PPB.2022155T071923.v2.0.",
            "HLS.L30.T38PPB.2022156T071344.v2.0.",
            "HLS.S30.T38PPB.2022157T071631.v2.0.",
        ],
        "38QMC": [],
        "38QMD": [
            "HLS.S30.T38QMD.2022142T071619.v2.0.",
            "HLS.S30.T38QMD.2022145T072619.v2.0.",
            "HLS.L30.T38QMD.2022146T072508.v2.0.",
            "HLS.L30.T38QMD.2022147T071923.v2.0.",
            "HLS.S30.T38QMD.2022147T071621.v2.0.",
            "HLS.S30.T38QMD.2022150T072621.v2.0.",
            "HLS.S30.T38QMD.2022152T071619.v2.0.",
            "HLS.L30.T38QMD.2022154T072540.v2.0.",
            "HLS.L30.T38QMD.2022155T071859.v2.0.",
            "HLS.S30.T38QMD.2022155T072619.v2.0.",
            "HLS.S30.T38QMD.2022157T071631.v2.0.",
            "HLS.S30.T38QMD.2022160T072621.v2.0.",
            "HLS.L30.T38QMD.2022162T072512.v2.0.",
            "HLS.S30.T38QMD.2022162T071619.v2.0.",
        ],
        "39QTT": [],
        "39QUT": [],
    }
    tile_queries = [
        ("38PMB", ["2022-06-08", "2022-05-29", "2022-05-19"]),
        ("38PPB", ["2022-06-08", "2022-05-29", "2022-05-19"]),
        ("39QTT", ["2022-06-08", "2022-05-29", "2022-05-19"]),
        ("30RYS", ["2022-06-09", "2022-05-30", "2022-05-20"]),
        ("38QMC", ["2022-06-09", "2022-05-30", "2022-05-20"]),
        ("39QUT", ["2022-06-09", "2022-05-30", "2022-05-20"]),
        ("38PMB", ["2022-06-09", "2022-05-30", "2022-05-20"]),
        ("38PNB", ["2022-06-10", "2022-05-31", "2022-05-21"]),
        ("39QTT", ["2022-06-10", "2022-05-31", "2022-05-21"]),
        ("38QMD", ["2022-06-11", "2022-06-01", "2022-05-22"]),
        ("38PMB", ["2022-06-11", "2022-06-01", "2022-05-22"]),
    ]

    tile_queries_str = [
        f"{tile_id}_{'_'.join(dates)}" for tile_id, dates in tile_queries
    ]
    tile_queries = {k: v for k, v in zip(tile_queries_str, tile_queries)}
    query_result = find_closest_tile(tile_queries, tile_database)
    assert list(query_result["hls_tiles"]) == [
        [
            "HLS.L30.T38PMB.2022154T072604.v2.0.",
            "HLS.S30.T38PMB.2022145T072619.v2.0.",
            "HLS.L30.T38PMB.2022139T071922.v2.0.",
        ],
        [
            "HLS.L30.T38PPB.2022155T071923.v2.0.",
            "HLS.L30.T38PPB.2022147T071947.v2.0.",
            "HLS.L30.T38PPB.2022139T071922.v2.0.",
        ],
        [None, None, None],
        [
            "HLS.L30.T30RYS.2022155T103334.v2.0.",
            "HLS.L30.T30RYS.2022147T103357.v2.0.",
            "HLS.L30.T30RYS.2022140T102748.v2.0.",
        ],
        [None, None, None],
        [None, None, None],
        [
            "HLS.L30.T38PMB.2022155T071923.v2.0.",
            "HLS.S30.T38PMB.2022145T072619.v2.0.",
            "HLS.L30.T38PMB.2022139T071922.v2.0.",
        ],
        [
            "HLS.L30.T38PNB.2022156T071344.v2.0.",
            "HLS.L30.T38PNB.2022147T071947.v2.0.",
            "HLS.S30.T38PNB.2022142T071619.v2.0.",
        ],
        [None, None, None],
        [
            "HLS.S30.T38QMD.2022157T071631.v2.0.",
            "HLS.L30.T38QMD.2022147T071923.v2.0.",
            "HLS.S30.T38QMD.2022142T071619.v2.0.",
        ],
        [
            "HLS.S30.T38PMB.2022157T071631.v2.0.",
            "HLS.L30.T38PMB.2022147T071947.v2.0.",
            "HLS.L30.T38PMB.2022139T071922.v2.0.",
        ],
    ]


@pytest.mark.parametrize(
    "tile_id, result",
    [
        ("HLS.L30.T38PMB.2022139T071922.v2.0.", "2022-05-19"),
        ("HLS.L30.T38PMB.202213T071922.v2.0.", None),
    ],
)
def test_parse_date_from_entry(tile_id, result):
    parsed_date = parse_date_from_entry(tile_id)
    if isinstance(parsed_date, datetime.datetime):
        parsed_date = parsed_date.strftime("%Y-%m-%d")
    assert parsed_date == result
