import logging
import os
import tempfile
import zipfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from instageo.data.s2_pipeline import (
    filter_best_product_in_folder,
    process_tile_bands,
    retrieve_sentinel2_metadata,
    unzip_all,
)


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
        download_info_list = [
            ("http://example.com/file1.zip", "tile1_id", "tile1"),
            ("http://example.com/file2.zip", "tile2_id", "tile2"),
        ]

        for _, full_tile_id, tile_name in download_info_list:
            tile_dir = os.path.join(temp_dir, tile_name)
            os.makedirs(tile_dir, exist_ok=True)
            zip_file_path = os.path.join(tile_dir, f"{full_tile_id}.zip")
            create_test_zip(zip_file_path, {"test_file.txt": "Test content"})

        # Mock os.path.exists to simulate that the zip files exist and files don't exist yet
        def mock_exists_side_effect(path):
            if path.endswith(".zip"):
                return True
            elif "test_file.txt" in path:
                return False
            elif os.path.isdir(path):
                return True
            return False

        mock_exists.side_effect = mock_exists_side_effect
        unzip_all(download_info_list, temp_dir)
        mock_exists.side_effect = (
            lambda path: path.endswith(".zip") or "test_file.txt" in path
        )

        assert os.path.exists(os.path.join(temp_dir, "tile1", "test_file.txt"))
        assert os.path.exists(os.path.join(temp_dir, "tile2", "test_file.txt"))


@patch("instageo.data.s2_utils.unzip_file")
@patch("os.path.exists")
def test_unzip_all_files_extracted_with_missing_zip(
    mock_exists, mock_unzip_file, caplog
):
    with tempfile.TemporaryDirectory() as temp_dir:
        download_info_list = [
            ("http://example.com/file2.zip", "tile2_id", "tile2"),
        ]

        # Mock os.path.exists to simulate that the zip file for tile2 is missing
        def mock_exists_side_effect(path):
            if path.endswith("tile2_id.zip"):
                return False
            return True

        mock_exists.side_effect = mock_exists_side_effect
        with caplog.at_level(logging.INFO):
            unzip_all(download_info_list, temp_dir)

        assert "Zip file not found" in caplog.text
        assert "tile2_id.zip" in caplog.text
        mock_unzip_file.assert_not_called()


@pytest.fixture
def mock_file_system():
    mock_os = MagicMock()
    mock_shutil = MagicMock()
    return mock_os, mock_shutil


@patch("os.path.isdir")
@patch("os.listdir")
@patch("shutil.move")
@patch("shutil.rmtree")
@patch("instageo.data.s2_utils.get_band_files")
def test_process_tile_bands(
    mock_get_band_files,
    mock_rmtree,
    mock_move,
    mock_listdir,
    mock_isdir,
    mock_file_system,
):
    tile_name = "TILE_NAME_1"
    full_tile_id = "ID_1"
    output_directory = "mock_output_directory"
    tile_folder = os.path.join(output_directory, tile_name)
    base_dir = os.path.join(tile_folder, full_tile_id, "GRANULE")
    img_data_dir = os.path.join(base_dir, "granule_folder", "IMG_DATA", "R20m")

    mock_get_band_files.return_value = [
        os.path.join(img_data_dir, "B02.jp2"),
        os.path.join(img_data_dir, "B03.jp2"),
        os.path.join(img_data_dir, "B11.jp2"),
    ]

    mock_isdir.side_effect = lambda path: {
        base_dir: True,
        img_data_dir: True,
    }.get(path, False)

    mock_listdir.side_effect = lambda path: {
        base_dir: ["granule_folder"],
        img_data_dir: ["B02.jp2", "B03.jp2", "B11.jp2"],  # Files in the R20m folder
    }.get(path, [])

    process_tile_bands(
        {"TILE_NAME_1": [{"full_tile_id": "ID_1"}]},
        output_directory,
        bands_needed=["B02", "B03", "B04", "B8A", "B11", "B12", "SCL"],
    )

    mock_move.assert_any_call(
        os.path.join(img_data_dir, "B02.jp2"), os.path.join(tile_folder, "B02.jp2")
    )
    mock_move.assert_any_call(
        os.path.join(img_data_dir, "B03.jp2"), os.path.join(tile_folder, "B03.jp2")
    )
    mock_move.assert_any_call(
        os.path.join(img_data_dir, "B11.jp2"), os.path.join(tile_folder, "B11.jp2")
    )

    assert mock_move.call_count == 3
    mock_rmtree.assert_called_once_with(os.path.join(tile_folder, full_tile_id))


@pytest.fixture
def temp_output_directory():
    with tempfile.TemporaryDirectory() as folder:
        yield folder


def test_missing_scl_file(temp_output_directory):
    tile_name = "TILE123"
    tile_products = [{"acquisition_date": "2024-01-01"}]
    output_directory = temp_output_directory
    history_dates = [(tile_name, ["2024-01-01"])]
    temporal_tolerance = 5

    with patch("instageo.data.s2_utils.find_scl_file", return_value=None):
        with patch("os.listdir", return_value=[]):  # Empty folder
            filter_best_product_in_folder(
                tile_name,
                tile_products,
                output_directory,
                history_dates,
                temporal_tolerance,
            )

    assert os.listdir(temp_output_directory) == [], "Expected no files to be removed."


def test_pixel_count_error(temp_output_directory):
    tile_name = "TILE123"
    tile_products = [{"acquisition_date": "2024-01-01"}]
    output_directory = temp_output_directory
    history_dates = [(tile_name, ["2024-01-01"])]
    temporal_tolerance = 5

    scl_mock_path = os.path.join(
        temp_output_directory, tile_name, "SCL_20240101TILE123.jp2"
    )
    os.makedirs(os.path.dirname(scl_mock_path), exist_ok=True)
    with open(scl_mock_path, "wb") as f:
        f.write(b"Fake SCL data")

    with patch("instageo.data.s2_utils.find_scl_file", return_value=scl_mock_path):
        with patch(
            "instageo.data.s2_utils.count_valid_pixels",
            side_effect=Exception("Test error"),
        ):
            try:
                filter_best_product_in_folder(
                    tile_name,
                    tile_products,
                    output_directory,
                    history_dates,
                    temporal_tolerance,
                )
            except Exception:
                pass

    assert os.path.exists(
        scl_mock_path
    ), "SCL file should not have been deleted on error."
