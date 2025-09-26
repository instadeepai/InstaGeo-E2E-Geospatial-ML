import datetime
import os
import shutil
from datetime import timezone
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr
from pystac import Item
from pystac_client import Client
from rasterio.crs import CRS
from shapely.geometry import Point, Polygon

from instageo.data.data_pipeline import apply_mask, get_tile_info
from instageo.data.hls_utils import (
    decode_fmask_value,
    dispatch_hls_candidate_items,
    get_raster_tile_info,
    is_valid_dataset_entry,
    open_mf_tiff_dataset,
    parallel_download,
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
    yield output_dir
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
        result_with_mask.band_data.where(result_with_mask.band_data != -1).count().values.item()
        == num_points - 2 * num_clouds
    )


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


# Tests and mocks for parallel_download
@pytest.fixture
def mock_earthaccess():
    with patch("instageo.data.hls_utils.earthaccess") as mock_ea:
        mock_ea.login.return_value = None
        mock_ea.download.return_value = None
        yield mock_ea


@pytest.fixture
def mock_os():
    with patch("instageo.data.hls_utils.os") as mock_os:
        mock_os.path.exists.return_value = False
        mock_os.path.getsize.return_value = 2000
        mock_os.listdir.return_value = []
        mock_os.remove.return_value = None
        mock_os.cpu_count.return_value = 4
        yield mock_os


@pytest.fixture
def mock_logging():
    with patch("instageo.data.hls_utils.logging") as mock_log:
        mock_log.warning.return_value = None
        yield mock_log


@pytest.fixture
def hls_dataset():
    return {
        "test_key": {
            "granules": [
                {
                    "assets": {
                        "blue": {
                            "href": "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B01.tif"
                        },
                        "green": {
                            "href": "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B02.tif"
                        },
                        "red": {
                            "href": "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B03.tif"
                        },
                        "nir narrow": {
                            "href": "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B04.tif"
                        },
                        "swir 1": {
                            "href": "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B05.tif"
                        },
                        "swir 2": {
                            "href": "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B06.tif"
                        },
                    }
                }
            ]
        }
    }


def test_parallel_download(hls_dataset, setup_and_teardown_output_dir, mock_earthaccess, mock_os):
    outdir = setup_and_teardown_output_dir

    parallel_download(hls_dataset, outdir, max_retries=1)

    mock_earthaccess.login.assert_called_once_with(persist=True)

    # Get the actual calls made to download
    actual_calls = mock_earthaccess.download.call_args_list

    # Verify we got the expected number of calls
    assert len(actual_calls) == 2

    # For each call, verify the arguments
    for call_args in actual_calls:
        args, kwargs = call_args
        urls = args[0]
        assert len(urls) == 6
        assert set(urls) == {
            "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B01.tif",
            "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B02.tif",
            "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B03.tif",
            "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B04.tif",
            "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B05.tif",
            "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B06.tif",
        }
        assert kwargs["local_path"] == outdir
        assert kwargs["threads"] == 4


def test_parallel_download_with_existing_files(
    hls_dataset, setup_and_teardown_output_dir, mock_earthaccess, mock_os
):
    outdir = setup_and_teardown_output_dir

    # Simulate files already exist
    mock_os.path.exists.return_value = True

    parallel_download(hls_dataset, outdir, max_retries=1)

    mock_earthaccess.download.assert_not_called()


def test_parallel_download_retry_on_failure(
    hls_dataset, setup_and_teardown_output_dir, mock_earthaccess, mock_os, mock_logging
):
    outdir = setup_and_teardown_output_dir

    # Simulate download failure
    mock_os.path.exists.return_value = False

    parallel_download(hls_dataset, outdir, max_retries=2)

    # Should have tried to download 3 times (initial + 2 retries)
    assert mock_earthaccess.download.call_count == 3
    mock_logging.warning.assert_called_once()


def test_parallel_download_removes_small_files(
    hls_dataset, setup_and_teardown_output_dir, mock_earthaccess, mock_os
):
    outdir = setup_and_teardown_output_dir

    # Simulate small file exists
    mock_os.listdir.return_value = ["small_file.tif"]
    mock_os.path.getsize.return_value = 500  # File size < 1024 bytes

    parallel_download(hls_dataset, outdir, max_retries=1)

    # Should have tried to remove the small file
    assert mock_os.remove.call_count == 2


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
        datetime=datetime.datetime(2024, 3, 27, 13, 0, 0, tzinfo=timezone.utc),  # daytime
    )


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


def test_is_valid_dataset_entry():
    """Test the is_valid_dataset_entry function."""
    valid_obsv = pd.Series({"hls_granules": ["granule_1", "granule_2", "granule_3"]})
    assert is_valid_dataset_entry(valid_obsv) is True

    null_granule_obsv = pd.Series({"hls_granules": ["granule_1", None, "granule_3"]})
    assert is_valid_dataset_entry(null_granule_obsv) is False

    duplicate_granule_obsv = pd.Series({"hls_granules": ["granule_1", "granule_1", "granule_3"]})
    assert is_valid_dataset_entry(duplicate_granule_obsv) is False


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
    ) as mock_remove, patch("instageo.data.hls_utils.os.path.getsize", return_value=2000):
        yield mock_listdir, mock_remove


@pytest.fixture
def mock_logging_warning():
    with patch("instageo.data.hls_utils.logging.warning") as mock_warn:
        yield mock_warn


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
        ("tile_1", ["2024-01-15T00:00:00", "2024-01-10T00:00:00"]),
        ("tile_2", ["2024-02-20T00:00:00", "2024-02-15T00:00:00"]),
    ]
    assert tile_queries == expected_tile_queries
    expected_tile_info = pd.DataFrame(
        {
            "tile_id": ["tile_1", "tile_2"],
            "min_date": ["2024-01-08T00:00:00", "2024-02-13T00:00:00"],
            "max_date": ["2024-01-17T23:59:59", "2024-02-22T23:59:59"],
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
