import os
import tempfile
import zipfile
from datetime import datetime

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile

from instageo.data.s2_utils import count_valid_pixels, find_scl_file, unzip_file


def create_mock_scl_band(data: np.ndarray):
    """Creates a mock SCL band as a raster file in memory."""
    transform = rasterio.transform.from_origin(1000, 2000, 10, 10)
    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        transform=transform,
        crs="EPSG:4326",
    ) as dataset:
        dataset.write(data, 1)
    return memfile


@pytest.fixture
def mock_scl_band():
    """Provides a mock SCL band for testing."""
    data = np.array([[0, 1, 2], [3, 0, 0], [4, 5, 0]], dtype=np.uint8)
    return create_mock_scl_band(data)


def test_count_valid_pixels(mock_scl_band):
    """Test the count_valid_pixels function."""
    with mock_scl_band.open() as mock_file:
        path = mock_file.name
        result = count_valid_pixels(path)
        assert result == 5


@pytest.fixture
def mock_scl_directory(tmp_path):
    """Creates a mock directory structure with mock SCL files."""
    tile_name = "30UXB"
    acquisition_date = datetime(2024, 11, 19)

    # Create mock directory
    mock_dir = tmp_path / "mock_data"
    mock_dir.mkdir()

    # Create some mock SCL files
    valid_file = f"{tile_name}_{acquisition_date.strftime('%Y%m%d')}_SCL_20m.jp2"
    invalid_file_1 = f"{tile_name}_20240101_SCL_20m.jp2"
    invalid_file_2 = f"31UXB_{acquisition_date.strftime('%Y%m%d')}_SCL_20m.jp2"

    (mock_dir / valid_file).write_text("Valid SCL content")
    (mock_dir / invalid_file_1).write_text("Invalid date content")
    (mock_dir / invalid_file_2).write_text("Invalid tile content")

    return mock_dir, tile_name, acquisition_date, valid_file


def test_find_scl_file(mock_scl_directory):
    """Test the find_scl_file function."""
    mock_dir, tile_name, acquisition_date, valid_file = mock_scl_directory

    result = find_scl_file(str(mock_dir), tile_name, acquisition_date)
    assert result == str(mock_dir / valid_file)

    result = find_scl_file(str(mock_dir), "31XYZ", acquisition_date)
    assert result is None, "Non-matching file test failed."


def test_unzip_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as test_zip_file:
            test_zip_file.writestr("test_file.txt", "This is a test file.")
            test_zip_file.writestr(
                "nested_dir/test_file2.txt", "This is another test file."
            )

        with tempfile.TemporaryDirectory() as extract_dir:
            unzip_file(zip_path, extract_dir)

            assert os.path.exists(os.path.join(extract_dir, "test_file.txt"))
            assert os.path.exists(
                os.path.join(extract_dir, "nested_dir/test_file2.txt")
            )
