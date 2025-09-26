import os
import tempfile
import time
import zipfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pystac import Item

from instageo.data.s2_utils import (
    ItemSearch,
    S2AuthState,
    add_s2_granules,
    create_mask_from_scl,
    create_s2_dataset,
    extract_and_delete_zip_files,
    find_best_tile,
    get_item_collection,
    get_item_search_objs,
    process_s2_metadata,
    retrieve_s2_metadata,
)


@pytest.fixture
def auth_state():
    """Fixture to create an instance of S2AuthState."""
    return S2AuthState(
        client_id="mock_client_id", username="mock_username", password="mock_password"
    )


@patch("instageo.data.s2_utils.requests.post")
def test_get_access_and_refresh_token_success(mock_post, auth_state):
    """Test successful authentication."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "access_token": "mock_access_token",
        "refresh_token": "mock_refresh_token",
        "expires_in": 3600,
    }
    result = auth_state._get_access_and_refresh_token()

    assert result == ("mock_access_token", "mock_refresh_token", 3600)

    mock_post.assert_called_once_with(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "client_id": "mock_client_id",
            "username": "mock_username",
            "password": "mock_password",
            "grant_type": "password",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )


@patch("instageo.data.s2_utils.requests.post")
def test_get_access_and_refresh_token_failure(mock_post, auth_state):
    """Test authentication failure with invalid credentials."""
    mock_post.return_value.status_code = 400
    mock_post.return_value.text = "Invalid user credentials"
    result = auth_state._get_access_and_refresh_token()

    assert result == (None, None, None)

    mock_post.assert_called_once_with(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "client_id": "mock_client_id",
            "username": "mock_username",
            "password": "mock_password",
            "grant_type": "password",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )


@patch("instageo.data.s2_utils.requests.post")
def test_refresh_access_token_success(mock_post, auth_state):
    """Test successful token refresh."""
    auth_state.refresh_token = "mock_refresh_token"

    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "access_token": "new_access_token",
        "expires_in": 3600,
    }
    access_token, expires_in = auth_state._refresh_access_token()

    assert access_token == "new_access_token"
    assert expires_in == 3600

    mock_post.assert_called_once_with(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "client_id": "mock_client_id",
            "refresh_token": "mock_refresh_token",
            "grant_type": "refresh_token",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )


@patch("instageo.data.s2_utils.requests.post")
def test_refresh_access_token_failure(mock_post, auth_state):
    """Test failed token refresh."""
    auth_state.refresh_token = "mock_refresh_token"

    mock_post.return_value.status_code = 400
    mock_post.return_value.text = "Invalid request"

    access_token, expires_in = auth_state._refresh_access_token()

    assert access_token is None
    assert expires_in is None
    assert auth_state.access_token is None
    assert auth_state.token_expiry_time is None

    mock_post.assert_called_once_with(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "client_id": "mock_client_id",
            "refresh_token": "mock_refresh_token",
            "grant_type": "refresh_token",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )


@pytest.fixture
def temp_folder():
    with tempfile.TemporaryDirectory() as folder:
        yield folder


@pytest.fixture
def sample_dataset():
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 2 bands, 2x2 each
    coords = {"x": [0, 1], "y": [0, 1], "band": [1, 2]}
    return xr.Dataset({"bands": (["band", "y", "x"], data)}, coords=coords)


@pytest.fixture
def sample_scl_data():
    return xr.DataArray(np.array([[1, 2], [3, 4]]))  # 2x2 SCL mask


def test_create_mask_from_valid_scl_data(sample_dataset, sample_scl_data):
    class_ids = [2, 3]
    mask = create_mask_from_scl(sample_scl_data, class_ids)
    result = sample_dataset.where(mask.values == 0)
    expected_data = np.array(
        [
            [[1, np.nan], [np.nan, 4]],
            [[5, np.nan], [np.nan, 8]],
        ]
    )
    assert "bands" in result
    assert result["bands"].shape == sample_dataset["bands"].shape
    np.testing.assert_almost_equal(result["bands"].values, expected_data)


def test_create_mask_from_scl_data_no_classes(sample_dataset, sample_scl_data):
    class_ids = []
    mask = create_mask_from_scl(sample_scl_data, class_ids)
    result = sample_dataset.where(mask.values == 0)
    xr.testing.assert_equal(result, sample_dataset)


@pytest.fixture
def s2_metadata():
    return {
        "features": [
            {
                "id": "uuid-1",
                "properties": {
                    "title": "S2A_MSIL2A_20201230T100031_N0214_R122_T33UUU_20201230T120024",
                    "startDate": "2020-12-30T10:00:31Z",
                    "services": {
                        "download": {
                            "url": "https://example.com/granule1",
                            "size": "100MB",
                        }
                    },
                    "cloudCover": 10.5,
                    "thumbnail": "https://example.com/thumbnail1",
                },
            },
            {
                "id": "uuid-2",
                "properties": {
                    "title": "S2B_MSIL2A_20201230T100031_N0214_R122_T33UUP_20201230T120024",
                    "startDate": "2020-12-30T10:00:31Z",
                    "services": {
                        "download": {
                            "url": "https://example.com/granule2",
                            "size": "120MB",
                        }
                    },
                    "cloudCover": 20.0,
                    "thumbnail": "https://example.com/thumbnail2",
                },
            },
        ]
    }


def test_process_s2_metadata_valid(s2_metadata):
    """Test processing valid metadata."""
    metadata = s2_metadata
    tile_id = "33UUU"
    result = process_s2_metadata(metadata, tile_id)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.iloc[0]["title"] == "S2A_MSIL2A_20201230T100031_N0214_R122_T33UUU_20201230T120024"
    assert result.iloc[0]["tile_id"] == "T33UUU"
    assert result.iloc[0]["cloud_cover"] == 10.5


def test_process_s2_metadata_missing_fields():
    """Test metadata with missing fields."""
    metadata = {
        "features": [
            {
                "id": "uuid-1",
                "properties": {
                    "title": "S2A_MSIL2A_20201230T100031_N0214_R122_T33UUU_20201230T120024",
                    "cloudCover": 10.5,
                    "thumbnail": "https://example.com/thumbnail1",
                },
            },
        ]
    }
    tile_id = "33UUU"

    with pytest.raises(KeyError):
        process_s2_metadata(metadata, tile_id)


def test_process_s2_metadata_no_matching_tile_id():
    """Test when no granules match the given tile_id."""
    metadata = {
        "features": [
            {
                "id": "uuid-1",
                "properties": {
                    "title": "S2A_MSIL2A_20201230T100031_N0214_R122_T33UUP_20201230T120024",
                    "startDate": "2020-12-30T10:00:31Z",
                    "services": {
                        "download": {
                            "url": "https://example.com/granule1",
                            "size": "100MB",
                        }
                    },
                    "cloudCover": 10.5,
                    "thumbnail": "https://example.com/thumbnail1",
                },
            },
        ]
    }
    tile_id = "33UUU"

    result = process_s2_metadata(metadata, tile_id)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_retrieve_s2_metadata_success():
    """Test successful retrieval and processing of metadata."""
    tile_info_df = pd.DataFrame(
        {
            "tile_id": ["33UUU"],
            "start_date": ["2023-01-01"],
            "end_date": ["2023-01-31"],
            "lon_min": [-10.0],
            "lon_max": [-5.0],
            "lat_min": [40.0],
            "lat_max": [45.0],
        }
    )

    mock_response_data = {
        "features": [
            {
                "id": "uuid-1",
                "properties": {
                    "title": "S2A_MSIL2A_20230101T100031_N0214_R122_T33UUU_20230101T120024",
                    "startDate": "2023-01-01T10:00:31Z",
                    "services": {
                        "download": {
                            "url": "https://example.com/granule1",
                            "size": "100MB",
                        },
                    },
                    "cloudCover": 10.0,
                    "thumbnail": "https://example.com/thumbnail1",
                },
            },
        ]
    }

    processed_metadata = pd.DataFrame(
        {
            "uuid": ["uuid-1"],
            "title": ["S2A_MSIL2A_20230101T100031_N0214_R122_T33UUU_20230101T120024"],
            "tile_id": ["33UUU"],
            "date": ["2023-01-01T10:00:31Z"],
            "url": ["https://example.com/granule1"],
            "size": ["100MB"],
            "cloud_cover": [10.0],
            "thumbnail": ["https://example.com/thumbnail1"],
        }
    )

    with patch("instageo.data.s2_utils.requests.get") as mock_get, patch(
        "instageo.data.s2_utils.process_s2_metadata"
    ) as mock_process:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: mock_response_data)
        mock_process.return_value = processed_metadata

        result = retrieve_s2_metadata(tile_info_df)

        mock_get.assert_called_once()
        mock_process.assert_called_once_with(mock_response_data, "33UUU")
        assert isinstance(result, dict)
        assert "33UUU" in result
        assert isinstance(result["33UUU"], pd.DataFrame)
        assert len(result["33UUU"]) == 1


def test_retrieve_s2_metadata_api_error():
    """Test handling of an API error."""
    tile_info_df = pd.DataFrame(
        {
            "tile_id": ["33UUU"],
            "start_date": ["2023-01-01"],
            "end_date": ["2023-01-31"],
            "lon_min": [-10.0],
            "lon_max": [-5.0],
            "lat_min": [40.0],
            "lat_max": [45.0],
        }
    )

    with patch("instageo.data.s2_utils.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=500)

        result = retrieve_s2_metadata(tile_info_df)

        mock_get.assert_called_once()
        assert isinstance(result, dict)
        assert "33UUU" not in result


def test_retrieve_s2_metadata_no_features():
    """Test when API response contains no features."""
    tile_info_df = pd.DataFrame(
        {
            "tile_id": ["33UUU"],
            "start_date": ["2023-01-01"],
            "end_date": ["2023-01-31"],
            "lon_min": [-10.0],
            "lon_max": [-5.0],
            "lat_min": [40.0],
            "lat_max": [45.0],
        }
    )

    mock_response_data = {"features": []}

    with patch("instageo.data.s2_utils.requests.get") as mock_get, patch(
        "instageo.data.s2_utils.process_s2_metadata"
    ) as mock_process:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: mock_response_data)
        mock_process.return_value = pd.DataFrame()

        result = retrieve_s2_metadata(tile_info_df)

        mock_get.assert_called_once()
        mock_process.assert_called_once_with(mock_response_data, "33UUU")
        assert isinstance(result, dict)
        assert "33UUU" in result


def test_retrieve_s2_metadata_multiple_tiles():
    """Test processing metadata for multiple tiles."""
    tile_info_df = pd.DataFrame(
        {
            "tile_id": ["33UUU", "33UUP"],
            "start_date": ["2023-01-01", "2023-01-15"],
            "end_date": ["2023-01-31", "2023-01-20"],
            "lon_min": [-10.0, -15.0],
            "lon_max": [-5.0, -10.0],
            "lat_min": [40.0, 35.0],
            "lat_max": [45.0, 40.0],
        }
    )

    mock_response_data = {"features": []}

    with patch("instageo.data.s2_utils.requests.get") as mock_get, patch(
        "instageo.data.s2_utils.process_s2_metadata"
    ) as mock_process:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: mock_response_data)
        mock_process.return_value = pd.DataFrame()

        result = retrieve_s2_metadata(tile_info_df)

        assert isinstance(result, dict)
        assert "33UUU" in result
        assert "33UUP" in result


def test_find_best_tile_prioritize_size():
    """Test prioritization of larger size over temporal difference."""
    tile_queries = {
        "query1": ("33UUU", ["2023-01-10"]),
    }

    tile_database = {
        "33UUU": pd.DataFrame(
            [
                {
                    "title": "S2A_MSIL2A_20230108",
                    "date": "2023-01-08",
                    "size": 500,
                    "url": "https://example.com/tile1",
                    "thumbnail": "https://example.com/thumb1",
                },
                {
                    "title": "S2A_MSIL2A_20230112",
                    "date": "2023-01-12",
                    "size": 700,
                    "url": "https://example.com/tile2",
                    "thumbnail": "https://example.com/thumb2",
                },
                {
                    "title": "S2A_MSIL2A_20230109",
                    "date": "2023-01-09",
                    "size": 600,
                    "url": "https://example.com/tile3",
                    "thumbnail": "https://example.com/thumb3",
                },
            ]
        )
    }

    result = find_best_tile(tile_queries, tile_database, temporal_tolerance=5)

    expected_result = pd.DataFrame(
        [
            {
                "tile_queries": "query1",
                "s2_tiles": ["S2A_MSIL2A_20230112"],  # Largest size, within tolerance
                "thumbnails": ["https://example.com/thumb2"],
                "urls": ["https://example.com/tile2"],
            }
        ]
    )

    pd.testing.assert_frame_equal(result, expected_result)


def test_find_best_tile_temporal_tiebreaker():
    """Test tiebreaker for tiles with the same size but different temporal differences."""
    tile_queries = {
        "query1": ("33UUU", ["2023-01-10"]),
    }

    tile_database = {
        "33UUU": pd.DataFrame(
            [
                {
                    "title": "S2A_MSIL2A_20230108",
                    "date": "2023-01-08",
                    "size": 700,
                    "url": "https://example.com/tile1",
                    "thumbnail": "https://example.com/thumb1",
                },
                {
                    "title": "S2A_MSIL2A_20230109",
                    "date": "2023-01-09",
                    "size": 700,
                    "url": "https://example.com/tile2",
                    "thumbnail": "https://example.com/thumb2",
                },
            ]
        )
    }

    result = find_best_tile(tile_queries, tile_database, temporal_tolerance=5)

    expected_result = pd.DataFrame(
        [
            {
                "tile_queries": "query1",
                "s2_tiles": ["S2A_MSIL2A_20230109"],  # Same size, closer temporal difference
                "thumbnails": ["https://example.com/thumb2"],
                "urls": ["https://example.com/tile2"],
            }
        ]
    )

    pd.testing.assert_frame_equal(result, expected_result)


def test_find_best_tile_no_match():
    """Test when no tile matches within temporal tolerance."""
    tile_queries = {
        "query1": ("33UUU", ["2023-01-10"]),
    }

    tile_database = {
        "33UUU": pd.DataFrame(
            [
                {
                    "title": "S2A_MSIL2A_20230101",
                    "date": "2023-01-01",
                    "size": 500,
                    "url": "https://example.com/tile1",
                    "thumbnail": "https://example.com/thumb1",
                },
            ]
        )
    }

    result = find_best_tile(tile_queries, tile_database, temporal_tolerance=5)

    expected_result = pd.DataFrame(
        [
            {
                "tile_queries": "query1",
                "s2_tiles": [None],  # No tile within tolerance
                "thumbnails": [None],
                "urls": [None],
            }
        ]
    )

    pd.testing.assert_frame_equal(result, expected_result)


def test_find_best_tile_multiple_queries():
    """Test multiple tile queries."""
    tile_queries = {
        "query1": ("33UUU", ["2023-01-10"]),
        "query2": ("33UUP", ["2023-01-15"]),
    }

    tile_database = {
        "33UUU": pd.DataFrame(
            [
                {
                    "title": "S2A_MSIL2A_20230108",
                    "date": "2023-01-08",
                    "size": 700,
                    "url": "https://example.com/tile1",
                    "thumbnail": "https://example.com/thumb1",
                },
                {
                    "title": "S2A_MSIL2A_20230109",
                    "date": "2023-01-09",
                    "size": 600,
                    "url": "https://example.com/tile2",
                    "thumbnail": "https://example.com/thumb2",
                },
            ]
        ),
        "33UUP": pd.DataFrame(
            [
                {
                    "title": "S2B_MSIL2A_20230114",
                    "date": "2023-01-14",
                    "size": 800,
                    "url": "https://example.com/tile3",
                    "thumbnail": "https://example.com/thumb3",
                },
            ]
        ),
    }

    result = find_best_tile(tile_queries, tile_database, temporal_tolerance=5)

    expected_result = pd.DataFrame(
        [
            {
                "tile_queries": "query1",
                "s2_tiles": ["S2A_MSIL2A_20230108"],  # Larger size
                "thumbnails": ["https://example.com/thumb1"],
                "urls": ["https://example.com/tile1"],
            },
            {
                "tile_queries": "query2",
                "s2_tiles": ["S2B_MSIL2A_20230114"],  # Within tolerance
                "thumbnails": ["https://example.com/thumb3"],
                "urls": ["https://example.com/tile3"],
            },
        ]
    )

    pd.testing.assert_frame_equal(result, expected_result)


def test_create_s2_dataset_valid():
    """Test creation of S2 dataset with valid data."""
    data_with_tiles = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-10"), pd.Timestamp("2023-01-15")],
            "mgrs_tile_id": ["33UUU", "33UUP"],
            "s2_tiles": [
                ["S2A_MSIL2A_20230110", "S2A_MSIL2A_20230112"],
                ["S2B_MSIL2A_20230115"],
            ],
            "urls": [
                ["https://example.com/tile1", "https://example.com/tile2"],
                ["https://example.com/tile3"],
            ],
        }
    )
    outdir = "/tmp/s2_dataset"

    s2_dataset, granules_to_download = create_s2_dataset(data_with_tiles, outdir)

    expected_s2_dataset = {
        "2023-01-10_33UUU": {
            "granules": [
                os.path.join(outdir, "s2_tiles", "S2A_MSIL2A_20230110"),
                os.path.join(outdir, "s2_tiles", "S2A_MSIL2A_20230112"),
            ],
        },
        "2023-01-15_33UUP": {
            "granules": [
                os.path.join(outdir, "s2_tiles", "S2B_MSIL2A_20230115"),
            ],
        },
    }

    expected_granules_to_download = pd.DataFrame(
        {
            "tiles": [
                "S2A_MSIL2A_20230110",
                "S2A_MSIL2A_20230112",
                "S2B_MSIL2A_20230115",
            ],
            "urls": [
                "https://example.com/tile1",
                "https://example.com/tile2",
                "https://example.com/tile3",
            ],
        }
    )

    assert s2_dataset == expected_s2_dataset
    pd.testing.assert_frame_equal(granules_to_download, expected_granules_to_download)


def test_create_s2_dataset_no_valid_tiles():
    """Test creation of S2 dataset with invalid data (no valid Sentinel-2 granules)."""
    data_with_tiles = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-10")],
            "mgrs_tile_id": ["33UUU"],
            "s2_tiles": [["invalid_tile_1", "invalid_tile_2"]],
            "urls": [["https://example.com/tile1", "https://example.com/tile2"]],
        }
    )
    outdir = "/tmp/s2_dataset"

    with pytest.raises(
        AssertionError, match="No observation record with valid Sentinel-2 granules"
    ):
        create_s2_dataset(data_with_tiles, outdir)


def test_add_s2_granules_valid():
    """Test adding S2 granules with valid data."""
    data = pd.DataFrame(
        {
            "mgrs_tile_id": ["33UUU", "33UUP"],
            "input_features_date": [
                pd.Timestamp("2023-01-15"),
                pd.Timestamp("2023-01-20"),
            ],
            "observation_date": [
                pd.Timestamp("2023-01-15"),
                pd.Timestamp("2023-01-20"),
            ],
            "x": [19.0, 17.0],
            "y": [19.0, 17.0],
        }
    )

    mock_tiles_info = pd.DataFrame(
        {
            "tile_id": ["33UUU", "33UUP"],
            "start_date": ["2023-01-10T00:00:00", "2023-01-15T00:00:00"],
            "end_date": ["2023-01-20T23:59:59", "2023-01-25T23:59:59"],
        }
    )
    mock_tile_queries = [
        ("33UUU", ["2023-01-15T00:00:00"]),
        ("33UUP", ["2023-01-20T00:00:00"]),
    ]
    mock_tile_database = {
        "33UUU": pd.DataFrame(
            [
                {
                    "title": "S2A_MSIL2A_20230114",
                    "date": "2023-01-14",
                    "size": 500,
                    "url": "https://example.com/tile1",
                    "thumbnail": "https://example.com/thumb1",
                },
            ]
        ),
        "33UUP": pd.DataFrame(
            [
                {
                    "title": "S2B_MSIL2A_20230119",
                    "date": "2023-01-19",
                    "size": 700,
                    "url": "https://example.com/tile2",
                    "thumbnail": "https://example.com/thumb2",
                },
            ]
        ),
    }
    mock_query_result = pd.DataFrame(
        {
            "tile_queries": [
                "33UUU_2023-01-15T00:00:00_2023-01-05T00:00:00_2022-12-26T00:00:00",
                "33UUP_2023-01-20T00:00:00_2023-01-10T00:00:00_2022-12-31T00:00:00",
            ],
            "s2_tiles": [["S2A_MSIL2A_20230114"], ["S2B_MSIL2A_20230119"]],
            "urls": [["https://example.com/tile1"], ["https://example.com/tile2"]],
        }
    )

    with patch(
        "instageo.data.data_pipeline.get_tile_info",
        return_value=(mock_tiles_info, mock_tile_queries),
    ), patch("instageo.data.s2_utils.retrieve_s2_metadata", return_value=mock_tile_database), patch(
        "instageo.data.s2_utils.find_best_tile", return_value=mock_query_result
    ):
        result = add_s2_granules(data, num_steps=3, temporal_step=10, temporal_tolerance=5)
    expected_result = pd.DataFrame(
        {
            "mgrs_tile_id": ["33UUU", "33UUP"],
            "input_features_date": [
                pd.Timestamp("2023-01-15"),
                pd.Timestamp("2023-01-20"),
            ],
            "observation_date": [
                pd.Timestamp("2023-01-15"),
                pd.Timestamp("2023-01-20"),
            ],
            "x": [19.0, 17.0],
            "y": [19.0, 17.0],
            "tile_queries": [
                "33UUU_2023-01-15T00:00:00_2023-01-05T00:00:00_2022-12-26T00:00:00",
                "33UUP_2023-01-20T00:00:00_2023-01-10T00:00:00_2022-12-31T00:00:00",
            ],
            "s2_tiles": [["S2A_MSIL2A_20230114"], ["S2B_MSIL2A_20230119"]],
            "urls": [["https://example.com/tile1"], ["https://example.com/tile2"]],
        }
    )

    pd.testing.assert_frame_equal(result, expected_result)


def test_extract_and_delete_zip_files():
    """Test extracting and deleting ZIP files."""
    with tempfile.TemporaryDirectory() as parent_dir:
        # Create subdirectories and sample ZIP files
        sub_dir = os.path.join(parent_dir, "subdir")
        os.makedirs(sub_dir, exist_ok=True)

        zip_file_path = os.path.join(sub_dir, "test.zip")
        extracted_file_path = os.path.join(sub_dir, "test_file.txt")

        # Create a ZIP file with a test file
        with zipfile.ZipFile(zip_file_path, "w") as zipf:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(b"This is a test file.")
                temp_file.seek(0)
                zipf.write(temp_file.name, arcname="test_file.txt")
                temp_file.close()

        # Ensure the ZIP file exists before extraction
        assert os.path.exists(zip_file_path)

        # Run the function
        extract_and_delete_zip_files(parent_dir)

        # Check if the ZIP file is deleted and files are extracted
        assert not os.path.exists(zip_file_path)
        assert os.path.exists(extracted_file_path)

        # Validate the content of the extracted file
        with open(extracted_file_path) as extracted_file:
            content = extracted_file.read()
            assert content == "This is a test file."


@pytest.fixture
def auth_instance():
    """Fixture to create an instance of S2AuthState."""
    return S2AuthState(client_id="test_client", username="test_user", password="test_pass")


def test_authenticate_success(auth_instance):
    """Test successful authentication."""
    with patch.object(
        auth_instance,
        "_get_access_and_refresh_token",
        return_value=("access123", "refresh123", 3600),
    ):
        auth_instance.authenticate()

        assert auth_instance.access_token == "access123"
        assert auth_instance.refresh_token == "refresh123"
        assert auth_instance.token_expiry_time is not None
        assert auth_instance.token_expiry_time > time.time()


def test_authenticate_failure(auth_instance):
    """Test authentication failure scenario."""
    with patch.object(
        auth_instance, "_get_access_and_refresh_token", return_value=(None, None, None)
    ):
        with pytest.raises(ValueError, match="Failed to authenticate and obtain tokens."):
            auth_instance.authenticate()


class MockClient:
    def search(self, collections, datetime, query):
        mock_item_search = MagicMock(spec=ItemSearch)
        mock_item_search.collections = ["sentinel-2-l2a"]
        mock_item_search.datetime = datetime
        mock_item_search.query = {"s2:mgrs_tile": {"eq": "some_tile"}}
        return mock_item_search


@pytest.fixture
def mock_client():
    return MockClient()


def test_get_item_search_objs(mock_client):
    tile_dict = {
        "granules": [
            "some/path/to/granule_20220101T123456_20220101T123457_some_other_data",
            "some/path/to/granule_20220102T123456_20220102T123457_some_other_data",
        ]
    }
    result = get_item_search_objs(mock_client, tile_dict)

    assert len(result) == 2
    assert isinstance(result[0], ItemSearch)
    assert result[0].collections == ["sentinel-2-l2a"]
    assert result[1].datetime == "2022-01-02"
    assert result[1].query == {"s2:mgrs_tile": {"eq": "some_tile"}}


def test_single_item_in_search_object():
    search_obj = MagicMock(spec=ItemSearch)
    geometry = {"type": "Point", "coordinates": [102.0, 0.5]}
    bbox = [100.0, 0.0, 105.0, 1.0]
    datetime_str = "2023-03-28T12:00:00Z"
    datetime_obj = datetime.fromisoformat(datetime_str[:-1])
    item = Item(id="item1", geometry=geometry, bbox=bbox, datetime=datetime_obj, properties={})
    search_obj.item_collection.return_value = [item]
    result = get_item_collection([search_obj])

    assert result == [item]


def test_multiple_items_in_search_object():
    search_obj = MagicMock(spec=ItemSearch)
    geometry = {"type": "Point", "coordinates": [102.0, 0.5]}
    bbox = [100.0, 0.0, 105.0, 1.0]
    datetime_str = "2023-03-28T12:00:00Z"
    datetime_obj = datetime.fromisoformat(datetime_str[:-1])

    item1 = Item(id="item1", geometry=geometry, bbox=bbox, datetime=datetime_obj, properties={})
    item2 = Item(id="item2", geometry=geometry, bbox=bbox, datetime=datetime_obj, properties={})

    search_obj.item_collection.return_value = [item1, item2]
    result = get_item_collection([search_obj])

    assert result == [item1]


def test_multiple_search_objects():
    search_obj1 = MagicMock(spec=ItemSearch)
    geometry = {"type": "Point", "coordinates": [102.0, 0.5]}
    bbox = [100.0, 0.0, 105.0, 1.0]
    datetime_str = "2023-03-28T12:00:00Z"
    datetime_obj = datetime.fromisoformat(datetime_str[:-1])
    item1 = Item(id="item1", geometry=geometry, bbox=bbox, datetime=datetime_obj, properties={})
    search_obj1.item_collection.return_value = [item1]
    search_obj2 = MagicMock(spec=ItemSearch)
    item2 = Item(id="item2", geometry=geometry, bbox=bbox, datetime=datetime_obj, properties={})
    search_obj2.item_collection.return_value = [item2]
    result = get_item_collection([search_obj1, search_obj2])

    assert result == [item1, item2]


def test_edge_case_no_search_objects():
    result = get_item_collection([])
    assert result == []
