import datetime
from datetime import timezone
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
import shapely
from pystac import Item
from pystac_client import Client
from shapely.geometry import Point, Polygon

from instageo.data.settings import HLSAPISettings, HLSBandsSettings
from instageo.data.stac_utils import (
    dispatch_candidate_items,
    find_best_items,
    get_raster_tile_info,
    is_daytime,
    is_valid_dataset_entry,
    retrieve_stac_metadata,
)

BANDS = HLSBandsSettings()
API = HLSAPISettings()


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
        ("tile_1", ["2024-01-15T00:00:00", "2024-01-10T00:00:00"]),
        ("tile_2", ["2024-02-20T00:00:00", "2024-02-15T00:00:00"]),
    ]
    assert tile_queries == expected_tile_queries
    expected_tile_info = pd.DataFrame(
        {
            "tile_id": ["tile_1", "tile_2"],
            "min_date": ["2024-01-08", "2024-02-13"],
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
def tiles_database(mock_candidate_items):
    """Creates a mock tiles database."""
    return {
        "tile1": mock_candidate_items,
    }


@patch("instageo.data.stac_utils.find_closest_items", return_value="mocked_hls_item")
def test_find_best_hls_items(mock_find_closest, mock_observations, tiles_database):
    """Test the find_best_hls_items function."""
    result = find_best_items(
        mock_observations,
        tiles_database,
        item_id_field="hls_item_id",
        candidate_items_field="hls_candidate_items",
        items_field="hls_items",
    )

    assert "tile1" in result
    assert len(result["tile1"]) == 2
    assert result["tile1"]["hls_items"].iloc[0] == "mocked_hls_item"
    assert result["tile1"]["hls_items"].iloc[1] == "mocked_hls_item"
    assert "hls_candidate_items" not in result["tile1"].columns


def test_is_valid_dataset_entry():
    """Test the is_valid_dataset_entry function."""
    valid_obsv = pd.Series({"hls_granules": ["granule_1", "granule_2", "granule_3"]})
    assert is_valid_dataset_entry(valid_obsv, item_id_field="hls_granules") is True

    null_granule_obsv = pd.Series({"hls_granules": ["granule_1", None, "granule_3"]})
    assert (
        is_valid_dataset_entry(null_granule_obsv, item_id_field="hls_granules") is False
    )

    duplicate_granule_obsv = pd.Series(
        {"hls_granules": ["granule_1", "granule_1", "granule_3"]}
    )
    assert (
        is_valid_dataset_entry(duplicate_granule_obsv, item_id_field="hls_granules")
        is False
    )


@patch("instageo.data.stac_utils.is_daytime", side_effect=lambda x: x.id == "item_1")
@patch("instageo.data.stac_utils.rename_stac_items", side_effect=lambda x, _: x)
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
    result = retrieve_stac_metadata(
        mock_client,
        mock_tile_info_df,
        cloud_coverage=10,
        daytime_only=True,
        bands_nameplate=BANDS.NAMEPLATE,
        collections=API.COLLECTIONS,
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
    result = dispatch_candidate_items(
        mock_observations,
        mock_candidate_items,
        item_id_field="hls_item_id",
        candidate_items_field="hls_candidate_items",
    )

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
