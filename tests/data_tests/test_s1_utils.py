import geopandas as gpd
import pandas as pd
import pytest
from pystac.item import Item
from pystac_client import Client
from shapely.geometry import Point

from instageo.data.data_pipeline import get_pystac_client, get_tile_info
from instageo.data.s1_utils import API, S1PointsPipeline, add_s1_stac_items


@pytest.fixture
def pystac_client():
    return Client.open(API.URL)


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "x": [28.80148067],
            "y": [-3.22772705],
            "mgrs_tile_id": ["35MQS"],
            "input_features_date": [pd.to_datetime("2021-01-01")],
            "label": [1],
        }
    )


def test_add_s1_stac_items(pystac_client, data):
    """Tests adding S1 STAC items to observations."""

    data_gdf = gpd.GeoDataFrame(
        data, geometry=gpd.points_from_xy(data["x"], data["y"]), crs="EPSG:4326"
    )
    data_gdf.rename_geometry("geometry_4326", inplace=True)

    result = add_s1_stac_items(
        pystac_client,
        data_gdf,
        num_steps=1,
        temporal_tolerance=12,
    )

    assert isinstance(result, dict)
    tile_id = list(result.keys())[0]
    tile_data = result[tile_id]
    assert "35MQS" in result
    tile_data = result["35MQS"]
    assert "s1_items" in tile_data.columns


def test_s1_points_pipeline_initialization():
    """Tests S1PointsPipeline initialization."""

    pipeline = S1PointsPipeline(
        output_directory="/tmp/test",
        chip_size=256,
        mask_types=[],
        masking_strategy="none",
        src_crs=4326,
        spatial_resolution=10.0,
        window_size=5,
        task_type="classification",
    )

    assert pipeline.output_directory == "/tmp/test"
    assert pipeline.chip_size == 256
    assert pipeline.mask_types == []
    assert pipeline.masking_strategy == "none"
    assert pipeline.src_crs == 4326
    assert pipeline.spatial_resolution == 10.0
    assert pipeline.window_size == 5
    assert pipeline.task_type == "classification"
