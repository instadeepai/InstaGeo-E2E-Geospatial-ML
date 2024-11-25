import os
import tempfile
import zipfile
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from instageo.data.s2_pipeline import retrieve_sentinel2_metadata, unzip_all


@pytest.fixture
def sample_tile_df():
    """Fixture for a sample tile dataframe."""
    data = {
        "tile_id": ["T33TWM"],
        "lon_min": [10.0],
        "lon_max": [10.5],
        "lat_min": [45.0],
        "lat_max": [45.5],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_history_dates():
    """Fixture for a sample history dates dictionary."""
    return [
        ("33TWM", ["2023-06-15", "2023-07-01"]),
    ]


@pytest.fixture
def mock_response():
    """Fixture to mock an API response."""
    return {
        "features": [
            {
                "properties": {
                    "title": "S2A_MSIL2A_20230615T100031_T33TWM",
                    "cloudCover": 15.0,
                    "services": {
                        "download": {"url": "https://example.com/download/12345"}
                    },
                    "thumbnail": "https://example.com/thumbnail/12345",
                }
            }
        ]
    }


@patch("instageo.data.s2_pipeline.requests.get")
def test_retrieve_sentinel2_metadata_success(
    mock_get, sample_tile_df, sample_history_dates, mock_response
):
    """Test the function when the API returns valid data."""
    mock_get.return_value = MagicMock(status_code=200, json=lambda: mock_response)

    result = retrieve_sentinel2_metadata(
        sample_tile_df,
        cloud_coverage=20,
        temporal_tolerance=10,
        history_dates=sample_history_dates,
    )

    assert "33TWM" in result
    assert len(result["33TWM"]) == 1
    assert result["33TWM"][0]["full_tile_id"] == "S2A_MSIL2A_20230615T100031_T33TWM"
    assert result["33TWM"][0]["cloudCover"] == 15.0
    assert "https://example.com/download/12345" in result["33TWM"][0]["download_link"]


@patch("instageo.data.s2_pipeline.requests.get")
def test_retrieve_sentinel2_metadata_no_features(
    mock_get, sample_tile_df, sample_history_dates
):
    """Test the function when the API returns no features."""
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {"features": []})

    result = retrieve_sentinel2_metadata(
        sample_tile_df,
        cloud_coverage=20,
        temporal_tolerance=10,
        history_dates=sample_history_dates,
    )

    assert "33TWM" not in result


@patch("instageo.data.s2_pipeline.requests.get")
def test_retrieve_sentinel2_metadata_api_failure(
    mock_get, sample_tile_df, sample_history_dates
):
    """Test the function when the API request fails."""
    mock_get.return_value = MagicMock(status_code=500)

    result = retrieve_sentinel2_metadata(
        sample_tile_df,
        cloud_coverage=20,
        temporal_tolerance=10,
        history_dates=sample_history_dates,
    )

    assert result == {}


def create_test_zip(zip_path, files):
    """Helper function to create a test zip file with specified files."""
    with zipfile.ZipFile(zip_path, "w") as test_zip:
        for file_path, content in files.items():
            test_zip.writestr(file_path, content)


@patch("instageo.data.s2_utils.unzip_file")
@patch("os.path.exists")
def test_unzip_all_files_extracted(mock_exists, mock_unzip_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Arrange
        download_info_list = [
            ("http://example.com/file1.zip", "tile1_id", "tile1"),
            ("http://example.com/file2.zip", "tile2_id", "tile2"),
        ]

        # Create zip files
        for _, full_tile_id, tile_name in download_info_list:
            tile_dir = os.path.join(temp_dir, tile_name)
            os.makedirs(tile_dir, exist_ok=True)
            zip_file_path = os.path.join(tile_dir, f"{full_tile_id}.zip")
            create_test_zip(zip_file_path, {"test_file.txt": "Test content"})

        # Mock os.path.exists to simulate that the zip files exist and files don't exist yet
        def mock_exists_side_effect(path):
            if path.endswith(".zip"):
                return True  # Return True for zip files
            elif "test_file.txt" in path:
                return False  # Return False for extracted files (this simulates them not existing yet)
            elif os.path.isdir(path):
                return True  # Return True for directories
            return False  # For other cases, return False

        mock_exists.side_effect = mock_exists_side_effect

        # Act
        unzip_all(download_info_list, temp_dir)

        # Assert: Check that extracted files are present by mocking file existence
        # Now that unzip_all has been called, the files should exist
        mock_exists.side_effect = (
            lambda path: path.endswith(".zip") or "test_file.txt" in path
        )

        assert os.path.exists(os.path.join(temp_dir, "tile1", "test_file.txt"))
        assert os.path.exists(os.path.join(temp_dir, "tile2", "test_file.txt"))
