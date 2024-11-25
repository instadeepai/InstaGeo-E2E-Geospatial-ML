import os
import tempfile
import zipfile
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile

from instageo.data.s2_utils import (
    count_valid_pixels,
    find_scl_file,
    get_access_and_refresh_token,
    refresh_access_token,
    unzip_file,
)


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

    mock_dir = tmp_path / "mock_data"
    mock_dir.mkdir()

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


@pytest.fixture
def mock_success_response():
    """Fixture to mock a successful authentication response."""
    return {
        "access_token": "mock_access_token",
        "refresh_token": "mock_refresh_token",
        "expires_in": 3600,
    }


@pytest.fixture
def mock_failure_response():
    """Fixture to mock a failed authentication response."""
    return {"error": "invalid_grant", "error_description": "Invalid user credentials"}


@patch("instageo.data.s2_utils.requests.post")
def test_get_access_and_refresh_token_success(mock_post, mock_success_response):
    """Test successful authentication."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = mock_success_response

    client_id = "mock_client_id"
    username = "mock_username"
    password = "mock_password"

    access_token, refresh_token, expires_in = get_access_and_refresh_token(
        client_id, username, password
    )

    assert access_token == mock_success_response["access_token"]
    assert refresh_token == mock_success_response["refresh_token"]
    assert expires_in == mock_success_response["expires_in"]
    mock_post.assert_called_once_with(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "client_id": client_id,
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )


@patch("instageo.data.s2_utils.requests.post")
def test_get_access_and_refresh_token_failure(mock_post, mock_failure_response):
    """Test authentication failure with invalid credentials."""
    mock_post.return_value.status_code = 400
    mock_post.return_value.text = "Invalid user credentials"

    client_id = "mock_client_id"
    username = "wrong_username"
    password = "wrong_password"

    access_token, refresh_token, expires_in = get_access_and_refresh_token(
        client_id, username, password
    )

    assert access_token is None
    assert refresh_token is None
    assert expires_in is None
    mock_post.assert_called_once_with(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "client_id": client_id,
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )


@patch("instageo.data.s2_utils.requests.post")
def test_get_access_and_refresh_token_server_error(mock_post):
    """Test server error response."""
    mock_post.return_value.status_code = 500
    mock_post.return_value.text = "Internal server error"

    client_id = "mock_client_id"
    username = "mock_username"
    password = "mock_password"

    access_token, refresh_token, expires_in = get_access_and_refresh_token(
        client_id, username, password
    )

    assert access_token is None
    assert refresh_token is None
    assert expires_in is None
    mock_post.assert_called_once_with(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "client_id": client_id,
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )


@patch("requests.post")
def test_refresh_access_token_success(mock_post):
    """Test successful token refresh."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "new_access_token",
        "expires_in": 3600,
    }
    mock_post.return_value = mock_response

    client_id = "test_client_id"
    refresh_token = "test_refresh_token"

    access_token, expires_in = refresh_access_token(client_id, refresh_token)

    assert access_token == "new_access_token"
    assert expires_in == 3600


@patch("requests.post")
def test_refresh_access_token_failure(mock_post):
    """Test failed token refresh."""
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = "Invalid request"
    mock_post.return_value = mock_response

    client_id = "test_client_id"
    refresh_token = "test_refresh_token"

    access_token, expires_in = refresh_access_token(client_id, refresh_token)

    assert access_token is None
    assert expires_in is None
