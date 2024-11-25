from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from instageo.data.s2_pipeline import retrieve_sentinel2_metadata


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
