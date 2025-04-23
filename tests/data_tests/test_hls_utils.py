import datetime
import os
import shutil
from datetime import timezone
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
import rasterio
import shapely
import xarray as xr
from pystac import Item
from pystac_client import Client
from rasterio.crs import CRS
from shapely.geometry import Point, Polygon

from instageo.data.data_pipeline import apply_mask, get_tiles
from instageo.data.hls_utils import (
    add_hls_granules,
    create_hls_dataset,
    decode_fmask_value,
    dispatch_hls_candidate_items,
    find_best_hls_items,
    find_closest_tile,
    get_raster_tile_info,
    is_daytime,
    is_valid_dataset_entry,
    load_cog,
    open_hls_cogs,
    open_mf_tiff_dataset,
    parallel_download,
    parse_date_from_entry,
    retrieve_hls_metadata,
    retrieve_hls_stac_metadata,
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
        mask_types=["cloud"],
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


def test_load_cog():
    url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/36/Q/WD/2020/7/S2A_36QWD_20200701_0_L2A/TCI.tif"
    data_array = load_cog(url)
    result = data_array.compute()

    assert isinstance(result, xr.DataArray)
    assert result.shape == (3, 10980, 10980)
    assert result.dtype == ("uint8")


@pytest.fixture
def real_bands_infos():
    return {
        "data_links": [
            [
                "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/36/Q/WD/2020/7/S2A_36QWD_20200701_0_L2A/TCI.tif",
                "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/36/Q/WD/2020/7/S2A_36QWD_20200701_0_L2A/TCI.tif",
            ]
        ]
    }


def test_open_hls_cogs_real_data(real_bands_infos):
    bands, masks, crs = open_hls_cogs(real_bands_infos, load_masks=False)

    assert isinstance(bands, xr.DataArray)
    assert bands.attrs["scale_factor"] == 1
    assert crs is not None
    assert bands.shape[0] == 3


@pytest.fixture
def mock_earthaccess_login():
    with patch("instageo.data.hls_utils.earthaccess.login") as mock_login:
        yield mock_login


@pytest.fixture
def mock_earthaccess_download():
    with patch("instageo.data.hls_utils.earthaccess.download") as mock_download:
        yield mock_download


@pytest.fixture
def mock_os_path_exists():
    with patch("instageo.data.hls_utils.os.path.exists") as mock_exists:
        yield mock_exists


@pytest.fixture
def mock_os_operations():
    with patch("instageo.data.hls_utils.os.listdir") as mock_listdir, patch(
        "instageo.data.hls_utils.os.remove"
    ) as mock_remove, patch(
        "instageo.data.hls_utils.os.path.getsize", return_value=2000
    ):
        yield mock_listdir, mock_remove


@pytest.fixture
def mock_logging_warning():
    with patch("instageo.data.hls_utils.logging.warning") as mock_warn:
        yield mock_warn


def test_parallel_download_calls_login(
    mock_earthaccess_login,
    mock_earthaccess_download,
    mock_os_path_exists,
    mock_os_operations,
):
    urls = {"https://example.com/tile1"}
    outdir = "/mock/output/dir"

    mock_os_path_exists.side_effect = lambda x: False  # Simulate files do not exist

    parallel_download(urls, outdir, max_retries=1)

    mock_earthaccess_login.assert_called_once_with(
        persist=True
    )  # Check login is called


def test_parallel_download_success(
    mock_earthaccess_login,
    mock_earthaccess_download,
    mock_os_path_exists,
    mock_os_operations,
):
    urls = {"https://example.com/tile1", "https://example.com/tile2"}
    outdir = "/mock/output/dir"

    # Simulate that files do not exist initially
    existing_files = set()

    def mock_exists(path):
        filename = os.path.basename(path)
        return filename in existing_files

    mock_os_path_exists.side_effect = mock_exists

    def mock_download(url_list, local_path, threads):
        for url in url_list:
            existing_files.add(os.path.basename(url))  # Simulate file being downloaded

    mock_earthaccess_download.side_effect = mock_download

    parallel_download(urls, outdir, max_retries=1)

    mock_earthaccess_download.assert_called_once_with(
        list(urls), local_path=outdir, threads=os.cpu_count()
    )


def test_parallel_download_with_existing_files(
    mock_earthaccess_login,
    mock_earthaccess_download,
    mock_os_path_exists,
    mock_os_operations,
):
    urls = {"https://example.com/tile1", "https://example.com/tile2"}
    outdir = "/mock/output/dir"

    mock_os_path_exists.side_effect = lambda x: True  # Simulate files already exist

    parallel_download(urls, outdir, max_retries=1)

    mock_earthaccess_download.assert_not_called()  # No need to download


@pytest.fixture
def mock_item():
    """Creates a mock PySTAC Item with a fixed time and location for testing."""
    properties = {"datetime": "2024-03-27T13:00:00Z"}

    bbox = [10.1658, 36.8065, 10.1658, 36.8065]  # Coordinates for Tunis
    return Item(
        id="test_item",
        geometry=None,
        bbox=bbox,
        properties=properties,
        href="",
        datetime=datetime.datetime(
            2024, 3, 27, 13, 0, 0, tzinfo=timezone.utc  # daytime
        ),
    )


def test_is_daytime(monkeypatch, mock_item):
    """Tests the is_daytime function with a known daytime scenario in Tunis."""

    class MockBox:
        def __init__(self, *args):
            pass

        @property
        def centroid(self):
            class MockPoint:
                # Tunis coordinates
                x = 10.1658  # longitude
                y = 36.8065  # latitude

            return MockPoint()

    monkeypatch.setattr(shapely.geometry, "box", MockBox)
    expected = True

    assert is_daytime(mock_item) == expected


@pytest.fixture
def sample_data():
    """Creates a sample GeoDataFrame for testing."""
    data = {
        "mgrs_tile_id": ["tile_1", "tile_2"],
        "input_features_date": [pd.Timestamp("2024-01-15"), pd.Timestamp("2024-02-20")],
        "geometry_4326": [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ],
    }
    return gpd.GeoDataFrame(data, geometry="geometry_4326", crs="EPSG:4326")


def test_get_raster_tile_info(sample_data):
    """Test function output for expected behavior."""
    tile_info, tile_queries = get_raster_tile_info(
        sample_data, num_steps=2, temporal_step=5, temporal_tolerance=2
    )

    assert isinstance(tile_info, pd.DataFrame)
    assert set(tile_info.columns) == {
        "tile_id",
        "min_date",
        "max_date",
        "lon_min",
        "lon_max",
        "lat_min",
        "lat_max",
    }
    assert isinstance(tile_queries, list)
    assert all(isinstance(i, tuple) for i in tile_queries)
    assert all(isinstance(i[1], list) for i in tile_queries)
    for date_col in ["min_date", "max_date"]:
        assert tile_info[date_col].apply(lambda x: isinstance(x, str)).all()
    assert len(tile_queries) == len(sample_data)
    expected_tile_queries = [
        ("tile_1", ["2024-01-15", "2024-01-10"]),
        ("tile_2", ["2024-02-20", "2024-02-15"]),
    ]
    assert tile_queries == expected_tile_queries
    expected_tile_info = pd.DataFrame(
        {
            "tile_id": ["tile_1", "tile_2"],
            "min_date": ["2024-01-08", "2024-02-13"],
            "max_date": ["2024-01-17", "2024-02-22"],
            "lon_min": [0.0, 2.0],
            "lon_max": [1.0, 3.0],
            "lat_min": [0.0, 2.0],
            "lat_max": [1.0, 3.0],
        }
    )
    pd.testing.assert_frame_equal(
        tile_info.sort_values("tile_id").reset_index(drop=True),
        expected_tile_info.sort_values("tile_id").reset_index(drop=True),
        check_dtype=True,
    )


@pytest.fixture
def mock_client():
    """Creates a mock pystac_client.Client."""
    client = MagicMock(spec=Client)
    return client


@pytest.fixture
def mock_tile_info_df():
    """Creates a mock DataFrame with tile info."""
    return pd.DataFrame(
        {
            "tile_id": ["tile_1"],
            "start_date": ["2024-03-01"],
            "end_date": ["2024-03-10"],
            "lon_min": [-77.0365],
            "lon_max": [-76.9365],
            "lat_min": [38.8977],
            "lat_max": [38.9977],
        }
    )


@pytest.fixture
def mock_stac_items():
    """Creates mock PySTAC Items."""
    item1 = Item(
        id="item_1",
        geometry=None,
        bbox=[-77.0365, 38.8977, -76.9365, 38.9977],
        properties={"eo:cloud_cover": 5, "datetime": "2024-03-05T12:00:00Z"},
        datetime=datetime.datetime(2024, 3, 5, 12, 0, 0, tzinfo=timezone.utc),
        href="",
    )

    item2 = Item(
        id="item_2",
        geometry=None,
        bbox=[-77.0365, 38.8977, -76.9365, 38.9977],
        properties={"eo:cloud_cover": 12, "datetime": "2024-03-06T14:00:00Z"},
        datetime=datetime.datetime(2024, 3, 6, 14, 0, 0, tzinfo=timezone.utc),
        href="",
    )

    return [item1, item2]


@patch("instageo.data.hls_utils.is_daytime", side_effect=lambda x: x.id == "item_1")
@patch("instageo.data.hls_utils.rename_hls_stac_items", side_effect=lambda x: x)
@patch(
    "instageo.data.geo_utils.make_valid_bbox",
    return_value=[-77.0365, 38.8977, -76.9365, 38.9977],
)
def test_retrieve_hls_stac_metadata(
    mock_make_valid_bbox,
    mock_rename_hls_stac_items,
    mock_is_daytime,
    mock_client,
    mock_tile_info_df,
    mock_stac_items,
):
    """Tests retrieve_hls_stac_metadata with mocked data."""

    mock_client.search.return_value.item_collection.return_value = mock_stac_items
    result = retrieve_hls_stac_metadata(
        mock_client, mock_tile_info_df, cloud_coverage=10, daytime_only=True
    )

    assert isinstance(result, dict)
    assert "tile_1" in result
    assert isinstance(result["tile_1"], list)
    assert len(result["tile_1"]) == 1
    assert result["tile_1"][0].id == "item_1"  # Only item_1 should be in result

    mock_make_valid_bbox.assert_called_once()
    mock_rename_hls_stac_items.assert_called()
    mock_is_daytime.assert_called()


@pytest.fixture
def mock_observations():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "mgrs_tile_id": ["tile1", "tile1"],
            "geometry_4326": [Point(-100, 40), Point(-99, 41)],
        }
    )
    return gpd.GeoDataFrame(df, geometry="geometry_4326", crs="EPSG:4326")


@pytest.fixture
def mock_candidate_items():
    """Creates mock PySTAC Items with geometries."""
    FIXED_DATETIME = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    item1 = Item(
        id="hls_001",
        geometry={
            "type": "Polygon",
            "coordinates": [[[-101, 39], [-101, 42], [-98, 42], [-98, 39], [-101, 39]]],
        },
        bbox=[-101, 39, -98, 42],
        datetime=FIXED_DATETIME,
        properties={},
        collection="HLS",
    )

    item2 = Item(
        id="hls_002",
        geometry={
            "type": "Polygon",
            "coordinates": [[[-100, 40], [-100, 43], [-97, 43], [-97, 40], [-100, 40]]],
        },
        bbox=[-100, 40, -97, 43],
        datetime=FIXED_DATETIME,
        properties={},
        collection="HLS",
    )

    return [item1, item2]


def test_dispatch_hls_candidate_items(mock_observations, mock_candidate_items):
    """Tests that the function correctly assigns candidate items."""
    result = dispatch_hls_candidate_items(mock_observations, mock_candidate_items)

    assert result is not None
    assert "hls_candidate_items" in result.columns
    assert len(result.iloc[0]["hls_candidate_items"]) > 0
    assert len(result.iloc[1]["hls_candidate_items"]) > 0

    expected_output = pd.DataFrame(
        {
            "id": [1, 2],
            "hls_candidate_items": [
                [mock_candidate_items[0]],
                [mock_candidate_items[0], mock_candidate_items[1]],
            ],
        }
    ).set_index("id")
    actual_output = result[["id", "hls_candidate_items"]].set_index("id")

    assert actual_output.equals(expected_output)


@pytest.fixture
def tiles_database(mock_candidate_items):
    """Creates a mock tiles database."""
    return {
        "tile1": mock_candidate_items,
    }


@patch("instageo.data.hls_utils.find_closest_hls_items", return_value="mocked_hls_item")
def test_find_best_hls_items(mock_find_closest, mock_observations, tiles_database):
    """Test the find_best_hls_items function."""
    result = find_best_hls_items(mock_observations, tiles_database)

    assert "tile1" in result
    assert len(result["tile1"]) == 2
    assert result["tile1"]["hls_items"].iloc[0] == "mocked_hls_item"
    assert result["tile1"]["hls_items"].iloc[1] == "mocked_hls_item"
    assert "hls_candidate_items" not in result["tile1"].columns


def test_is_valid_dataset_entry():
    """Test the is_valid_dataset_entry function."""
    valid_obsv = pd.Series({"hls_granules": ["granule_1", "granule_2", "granule_3"]})
    assert is_valid_dataset_entry(valid_obsv) is True

    null_granule_obsv = pd.Series({"hls_granules": ["granule_1", None, "granule_3"]})
    assert is_valid_dataset_entry(null_granule_obsv) is False

    duplicate_granule_obsv = pd.Series(
        {"hls_granules": ["granule_1", "granule_1", "granule_3"]}
    )
    assert is_valid_dataset_entry(duplicate_granule_obsv) is False
