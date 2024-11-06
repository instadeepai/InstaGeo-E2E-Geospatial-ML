import datetime
import os
import shutil

import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS

from instageo.data.geo_utils import get_tiles
from instageo.data.hls_pipeline import create_hls_dataset
from instageo.data.hls_utils import (
    add_hls_granules,
    find_closest_tile,
    get_hls_tile_info,
    parallel_download,
    parse_date_from_entry,
    retrieve_hls_metadata,
)


@pytest.fixture
def setup_and_teardown_output_dir():
    output_dir = "/tmp/test_hls"
    os.makedirs(output_dir, exist_ok=True)
    yield
    shutil.rmtree(output_dir)


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


def test_get_hls_tile_info(observation_data):
    hls_tiles = get_tiles(observation_data, min_count=3)
    tiles_info, tile_queries = get_hls_tile_info(
        hls_tiles, num_steps=3, temporal_step=5
    )
    pd.testing.assert_frame_equal(
        tiles_info,
        pd.DataFrame(
            {
                "tile_id": ["38PMB"],
                "min_date": ["2022-05-24"],
                "max_date": ["2022-06-14"],
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
    tile_database = retrieve_hls_metadata(tile_info)
    print(tile_database)
    assert tile_database["38PMB"][0] == [  # ignore links
        "HLS.S30.T38PMB.2022145T072619.v2.0",
        "HLS.L30.T38PMB.2022146T072532.v2.0",
        "HLS.L30.T38PMB.2022147T071947.v2.0",
        "HLS.S30.T38PMB.2022147T071621.v2.0",
        "HLS.S30.T38PMB.2022150T072621.v2.0",
        "HLS.S30.T38PMB.2022152T071619.v2.0",
        "HLS.L30.T38PMB.2022154T072604.v2.0",
        "HLS.L30.T38PMB.2022155T071923.v2.0",
        "HLS.S30.T38PMB.2022155T072619.v2.0",
        "HLS.S30.T38PMB.2022157T071631.v2.0",
        "HLS.S30.T38PMB.2022160T072621.v2.0",
        "HLS.L30.T38PMB.2022162T072536.v2.0",
        "HLS.S30.T38PMB.2022162T071619.v2.0",
        "HLS.L30.T38PMB.2022163T072000.v2.0",
        "HLS.S30.T38PMB.2022165T072619.v2.0",
    ]


def test_add_hls_granules(observation_data):
    data = get_tiles(observation_data, min_count=3)
    result = add_hls_granules(data)
    assert list(result["hls_tiles"]) == [
        [
            "HLS.S30.T38PMB.2022160T072621.v2.0",
            "HLS.S30.T38PMB.2022150T072621.v2.0",
            "HLS.L30.T38PMB.2022139T071922.v2.0",
        ],
        [
            "HLS.S30.T38PMB.2022160T072621.v2.0",
            "HLS.S30.T38PMB.2022150T072621.v2.0",
            "HLS.L30.T38PMB.2022139T071922.v2.0",
        ],
        [
            "HLS.S30.T38PMB.2022160T072621.v2.0",
            "HLS.S30.T38PMB.2022150T072621.v2.0",
            "HLS.L30.T38PMB.2022139T071922.v2.0",
        ],
        [
            "HLS.S30.T38PMB.2022160T072621.v2.0",
            "HLS.S30.T38PMB.2022150T072621.v2.0",
            "HLS.S30.T38PMB.2022140T072621.v2.0",
        ],
    ]


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


def test_create_hls_dataset(observation_data):
    data = get_tiles(observation_data, min_count=3)
    data_with_tiles = add_hls_granules(
        data, num_steps=3, temporal_step=10, temporal_tolerance=5
    )
    hls_dataset, tiles_to_download = create_hls_dataset(data_with_tiles, outdir="")
    assert len(tiles_to_download) == 28
    assert len(hls_dataset) == 2
    assert len(hls_dataset["2022-06-08_T38PMB"]["tiles"]) == 18
    assert len(hls_dataset["2022-06-08_T38PMB"]["fmasks"]) == 3
    assert (
        hls_dataset["2022-06-08_T38PMB"]["tiles"]["B02_0"]
        == "hls_tiles/HLS.S30.T38PMB.2022160T072621.v2.0.B02.tif"
    )
    assert (
        hls_dataset["2022-06-08_T38PMB"]["fmasks"]["Fmask_0"]
        == "hls_tiles/HLS.S30.T38PMB.2022160T072621.v2.0.Fmask.tif"
    )
    assert (
        hls_dataset["2022-06-08_T38PMB"]["tiles"]["B02_1"]
        == "hls_tiles/HLS.S30.T38PMB.2022150T072621.v2.0.B02.tif"
    )
    assert (
        hls_dataset["2022-06-08_T38PMB"]["fmasks"]["Fmask_1"]
        == "hls_tiles/HLS.S30.T38PMB.2022150T072621.v2.0.Fmask.tif"
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
