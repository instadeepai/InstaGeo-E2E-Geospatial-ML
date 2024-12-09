import logging
import os
import tempfile
import zipfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.io import MemoryFile

from instageo.data.s2_utils import (
    S2AuthState,
    apply_class_mask,
    count_valid_pixels,
    create_scl_data,
    find_scl_file,
    get_band_files,
    open_mf_jp2_dataset,
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


@patch("os.path.isdir")
@patch("os.listdir")
@patch("shutil.move")
@patch("shutil.rmtree")
def test_get_band_files(mock_rmtree, mock_move, mock_listdir, mock_isdir):
    tile_name = "tile1"
    full_tile_id = "tile1_product1"
    output_directory = "/mock/output/directory"
    tile_folder = os.path.join(output_directory, tile_name)
    base_dir = os.path.join(tile_folder, full_tile_id, "GRANULE")
    img_data_dir = os.path.join(base_dir, "granule_folder", "IMG_DATA", "R20m")

    mock_isdir.side_effect = lambda path: {
        base_dir: True,
        img_data_dir: True,
    }.get(path, False)

    mock_listdir.side_effect = lambda path: {
        base_dir: ["granule_folder"],
        img_data_dir: ["B02.jp2", "B03.jp2", "B11.jp2"],  # Files in the R20m folder
    }.get(path, [])

    get_band_files(
        tile_name,
        full_tile_id,
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


@patch("os.path.isdir")
@patch("os.listdir")
@patch("shutil.rmtree")
@patch("shutil.move")
def test_get_band_files_no_granule_folder(
    mock_move, mock_rmtree, mock_listdir, mock_isdir, caplog
):
    tile_name = "tile1"
    full_tile_id = "tile1_product1"
    output_directory = "/mock/output/directory"
    mock_isdir.return_value = False
    with caplog.at_level(logging.INFO):
        get_band_files(
            tile_name,
            full_tile_id,
            output_directory,
            bands_needed=["B02", "B03", "B04", "B8A", "B11", "B12", "SCL"],
        )

    assert f"GRANULE folder not found in {full_tile_id}" in caplog.text
    mock_move.assert_not_called()
    mock_rmtree.assert_not_called()


@pytest.fixture
def temp_folder():
    with tempfile.TemporaryDirectory() as folder:
        yield folder


def test_missing_folder():
    datasets, crs = open_mf_jp2_dataset(
        band_folder="non_existent_folder",
        history_dates=[("id1", ["20240101"])],
        mask_cloud=False,
        water_mask=False,
        temporal_tolerance=5,
        num_bands_per_timestamp=6,
    )
    assert datasets == [None], "Expected None for datasets when folder is missing."
    assert crs is None, "Expected None for CRS when folder is missing."


def test_empty_band_file(temp_folder):
    empty_file_path = os.path.join(temp_folder, "empty_band_20240101T.jp2")
    open(empty_file_path, "w").close()  # Create an empty file

    with patch("os.listdir", return_value=["empty_band_20240101T.jp2"]), patch(
        "os.path.getsize", return_value=0
    ):
        datasets, crs = open_mf_jp2_dataset(
            band_folder=temp_folder,
            history_dates=[("id1", ["20240101"])],
            mask_cloud=False,
            water_mask=False,
            temporal_tolerance=5,
            num_bands_per_timestamp=6,
        )
    assert datasets == [None], "Expected None for datasets when files are empty."
    assert crs is None, "Expected None for CRS when files are empty."


def test_mismatched_band_count(temp_folder):
    valid_file_path = os.path.join(temp_folder, "band_20240101T.jp2")
    scl_file_path = os.path.join(temp_folder, "SCL_20240101T.jp2")

    with open(valid_file_path, "wb") as f, open(scl_file_path, "wb") as scl_f:
        f.write(b"Non-empty band data")
        scl_f.write(b"Non-empty SCL data")

    with patch(
        "os.listdir", return_value=["band_20240101T.jp2", "SCL_20240101T.jp2"]
    ), patch("os.path.getsize", return_value=10), patch(
        "rioxarray.open_rasterio",
        return_value=MagicMock(rio=MagicMock(crs="EPSG:4326")),
    ):
        datasets, crs = open_mf_jp2_dataset(
            band_folder=temp_folder,
            history_dates=[("id1", ["20240101"])],
            mask_cloud=False,
            water_mask=False,
            temporal_tolerance=5,
            num_bands_per_timestamp=6,
        )
    assert datasets == [
        None
    ], "Expected None for datasets when band count is mismatched."
    assert crs is None, "Expected None for CRS when band count is mismatched."


@pytest.fixture
def mock_rasterio_open():
    """Fixture to mock rasterio.open."""
    with patch("rasterio.open") as mock_open:
        yield mock_open


def test_create_scl_data_valid_files(mock_rasterio_open):
    mock_rasterio_open.return_value.__enter__.return_value.read.side_effect = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
    ]
    scl_band_files = ["file1.tif", "file2.tif"]
    result = create_scl_data(scl_band_files)

    expected = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )

    assert result.shape == expected.shape
    assert np.array_equal(result, expected)


def test_create_scl_data_empty_file_list():
    scl_band_files = []
    result = create_scl_data(scl_band_files)

    assert result.size == 0
    assert isinstance(result, np.ndarray)


@pytest.fixture
def sample_dataset():
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 2 bands, 2x2 each
    coords = {"x": [0, 1], "y": [0, 1], "band": [1, 2]}
    return xr.Dataset({"bands": (["band", "y", "x"], data)}, coords=coords)


@pytest.fixture
def sample_scl_data():
    return np.array([[1, 2], [3, 4]])  # 2x2 SCL mask


def test_apply_class_mask_valid_mask(sample_dataset, sample_scl_data):
    class_ids = [2, 3]
    result = apply_class_mask(sample_dataset, sample_scl_data, class_ids)

    expected_data = np.array(
        [
            [[1, np.nan], [np.nan, 4]],
            [[5, np.nan], [np.nan, 8]],
        ]
    )

    assert "bands" in result
    assert result["bands"].shape == sample_dataset["bands"].shape
    np.testing.assert_array_equal(result["bands"].isnull(), np.isnan(expected_data))
    np.testing.assert_array_almost_equal(
        result["bands"].values, expected_data, decimal=6
    )


def test_apply_class_mask_no_classes_to_mask(sample_dataset, sample_scl_data):
    class_ids = []
    result = apply_class_mask(sample_dataset, sample_scl_data, class_ids)

    xr.testing.assert_equal(result, sample_dataset)
