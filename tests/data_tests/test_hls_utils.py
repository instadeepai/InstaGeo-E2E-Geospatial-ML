import datetime
import os
import shutil

import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.crs import CRS

from instageo.data.data_pipeline import apply_mask, get_tiles
from instageo.data.hls_utils import (
    add_hls_granules,
    create_hls_dataset,
    decode_fmask_value,
    find_closest_tile,
    open_mf_tiff_dataset,
    parallel_download,
    parse_date_from_entry,
    retrieve_hls_metadata,
)


@pytest.fixture
def observation_data():
    data = pd.DataFrame(
        {
            "date": {
                0: "2022-06-08",
                1: "2022-06-08",
                2: "2022-06-08",
                3: "2022-06-08",
                4: "2022-06-09",
                5: "2022-06-09",
                6: "2022-06-09",
                7: "2022-06-08",
                8: "2022-06-09",
                9: "2022-06-09",
            },
            "x": {
                0: 44.48,
                1: 44.48865,
                2: 46.437787,
                3: 49.095545,
                4: -0.1305,
                5: 44.6216,
                6: 49.398908,
                7: 44.451435,
                8: 49.435228,
                9: 44.744167,
            },
            "y": {
                0: 15.115617,
                1: 15.099767,
                2: 14.714659,
                3: 16.066929,
                4: 28.028967,
                5: 16.16195,
                6: 16.139727,
                7: 15.209633,
                8: 16.151837,
                9: 15.287778,
            },
            "year": {
                0: 2022,
                1: 2022,
                2: 2022,
                3: 2022,
                4: 2022,
                5: 2022,
                6: 2022,
                7: 2022,
                8: 2022,
                9: 2022,
            },
        }
    )
    data["date"] = pd.to_datetime(data["date"])
    data["input_features_date"] = data["date"]
    return data


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
        decode_fmask_value,
        "HLS",
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


def test_retrieve_hls_metadata():
    tile_info = pd.DataFrame(
        {
            "tile_id": ["38PMB"],
            "min_date": ["2022-05-24"],
            "max_date": ["2022-06-14"],
            "lon_min": [44.451435],
            "lon_max": [44.744167],
            "lat_min": [15.099767],
            "lat_max": [15.287778],
        }
    )
    tile_database = retrieve_hls_metadata(tile_info, cloud_coverage=0.0001)
    assert tile_database["38PMB"][0] == [  # ignore links
        "HLS.S30.T38PMB.2022145T072619.v2.0",
        "HLS.L30.T38PMB.2022146T072532.v2.0",
        "HLS.L30.T38PMB.2022154T072604.v2.0",
        "HLS.L30.T38PMB.2022163T072000.v2.0",
        "HLS.S30.T38PMB.2022165T072619.v2.0",
    ]


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
    # Let's ignore the links when defining the tile_database param
    tile_database = {
        "30RYS": (
            [
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
            [[], [], [], [], [], [], [], [], [], [], [], [], []],
        ),
        "38PMB": (
            [
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
            [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        ),
        "38PNB": (
            [
                "HLS.S30.T38PNB.2022142T071619.v2.0.",
                "HLS.L30.T38PNB.2022147T071947.v2.0.",
                "HLS.S30.T38PNB.2022147T071621.v2.0.",
                "HLS.L30.T38PNB.2022148T071311.v2.0.",
                "HLS.S30.T38PNB.2022152T071619.v2.0.",
                "HLS.L30.T38PNB.2022155T071923.v2.0.",
                "HLS.L30.T38PNB.2022156T071344.v2.0.",
                "HLS.S30.T38PNB.2022157T071631.v2.0.",
            ],
            [[], [], [], [], [], [], [], []],
        ),
        "38PPB": (
            [
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
            [[], [], [], [], [], [], [], [], [], []],
        ),
        "38QMC": [],
        "38QMD": (
            [
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
            [[], [], [], [], [], [], [], [], [], [], [], [], [], []],
        ),
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
    hls_tiles_with_links = list(query_result["hls_tiles"])
    assert list(hls_tiles for hls_tiles in hls_tiles_with_links) == [
        [
            "HLS.S30.T38PMB.2022160T072621.v2.0.",
            "HLS.S30.T38PMB.2022150T072621.v2.0.",
            "HLS.L30.T38PMB.2022139T071922.v2.0.",
        ],
        [
            "HLS.S30.T38PPB.2022157T071631.v2.0.",
            "HLS.L30.T38PPB.2022148T071311.v2.0.",
            "HLS.L30.T38PPB.2022139T071922.v2.0.",
        ],
        [None, None, None],
        [
            "HLS.S30.T30RYS.2022159T102559.v2.0.",
            "HLS.S30.T30RYS.2022149T102559.v2.0.",
            "HLS.L30.T30RYS.2022140T102748.v2.0.",
        ],
        [None, None, None],
        [None, None, None],
        [
            "HLS.S30.T38PMB.2022160T072621.v2.0.",
            "HLS.S30.T38PMB.2022150T072621.v2.0.",
            "HLS.S30.T38PMB.2022140T072621.v2.0.",
        ],
        [
            "HLS.S30.T38PNB.2022157T071631.v2.0.",
            "HLS.S30.T38PNB.2022152T071619.v2.0.",
            "HLS.S30.T38PNB.2022142T071619.v2.0.",
        ],
        [None, None, None],
        [
            "HLS.L30.T38QMD.2022162T072512.v2.0.",
            "HLS.S30.T38QMD.2022152T071619.v2.0.",
            "HLS.S30.T38QMD.2022142T071619.v2.0.",
        ],
        [
            "HLS.L30.T38PMB.2022162T072536.v2.0.",
            "HLS.S30.T38PMB.2022152T071619.v2.0.",
            "HLS.S30.T38PMB.2022142T071619.v2.0.",
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


def test_add_hls_granules(observation_data):
    data = get_tiles(observation_data, min_count=3)
    result = add_hls_granules(data, cloud_coverage=0.0001)
    assert list(result["hls_tiles"]) == [
        [
            "HLS.L30.T38PMB.2022163T072000.v2.0",
            "HLS.L30.T38PMB.2022146T072532.v2.0",
            "HLS.L30.T38PMB.2022139T071922.v2.0",
        ],
        [
            "HLS.L30.T38PMB.2022163T072000.v2.0",
            "HLS.L30.T38PMB.2022146T072532.v2.0",
            "HLS.L30.T38PMB.2022139T071922.v2.0",
        ],
        [
            "HLS.L30.T38PMB.2022163T072000.v2.0",
            "HLS.L30.T38PMB.2022146T072532.v2.0",
            "HLS.L30.T38PMB.2022139T071922.v2.0",
        ],
        [
            "HLS.L30.T38PMB.2022163T072000.v2.0",
            "HLS.L30.T38PMB.2022146T072532.v2.0",
            "HLS.S30.T38PMB.2022140T072621.v2.0",
        ],
    ]


def test_create_hls_dataset(observation_data):
    data = get_tiles(observation_data, min_count=3)
    data_with_tiles = add_hls_granules(
        data, num_steps=3, temporal_step=10, temporal_tolerance=5, cloud_coverage=0.0001
    )
    hls_dataset, tiles_to_download = create_hls_dataset(data_with_tiles, outdir="")
    assert len(tiles_to_download) == 28
    assert len(hls_dataset) == 2
    assert len(hls_dataset["2022-06-08_T38PMB"]["tiles"]) == 18
    assert len(hls_dataset["2022-06-08_T38PMB"]["fmasks"]) == 3
    assert (
        hls_dataset["2022-06-08_T38PMB"]["tiles"]["B02_0"]
        == "hls_tiles/HLS.L30.T38PMB.2022163T072000.v2.0.B02.tif"
    )
    assert (
        hls_dataset["2022-06-08_T38PMB"]["fmasks"]["Fmask_0"]
        == "hls_tiles/HLS.L30.T38PMB.2022163T072000.v2.0.Fmask.tif"
    )
    assert (
        hls_dataset["2022-06-08_T38PMB"]["tiles"]["B02_1"]
        == "hls_tiles/HLS.L30.T38PMB.2022146T072532.v2.0.B02.tif"
    )
    assert (
        hls_dataset["2022-06-08_T38PMB"]["fmasks"]["Fmask_1"]
        == "hls_tiles/HLS.L30.T38PMB.2022146T072532.v2.0.Fmask.tif"
    )
    assert (
        hls_dataset["2022-06-08_T38PMB"]["tiles"]["B02_2"]
        == "hls_tiles/HLS.L30.T38PMB.2022139T071922.v2.0.B02.tif"
    )
    assert (
        hls_dataset["2022-06-08_T38PMB"]["fmasks"]["Fmask_2"]
        == "hls_tiles/HLS.L30.T38PMB.2022139T071922.v2.0.Fmask.tif"
    )


@pytest.mark.auth
def test_download_hls_tile(setup_and_teardown_output_dir):
    outdir = "/tmp/test_hls"
    urls = [
        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B01.tif"  # noqa
    ]
    parallel_download(urls, outdir=outdir)
    out_filename = "/tmp/test_hls/HLS.L30.T38PMB.2022139T071922.v2.0.B01.tif"  # noqa
    assert os.path.exists(out_filename)
    src = rasterio.open(out_filename)
    assert isinstance(src.crs, CRS)


@pytest.mark.auth
def test_download_hls_tile_with_retry(setup_and_teardown_output_dir):
    outdir = "/tmp/test_hls"
    open(
        os.path.join(outdir, "HLS.L30.T38PMB.2022139T071922.v2.0.B02.tif"), "w"
    ).close()
    urls = {
        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B03.tif",  # noqa
        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B02.tif",  # noqa
    }
    parallel_download(urls, outdir=outdir)
    out_filename = os.path.join(outdir, "HLS.L30.T38PMB.2022139T071922.v2.0.B02.tif")
    assert os.path.exists(out_filename)
    src = rasterio.open(out_filename)
    assert isinstance(src.crs, CRS)
