import os
import shutil
from datetime import datetime
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
from absl import flags
from pystac import Asset, Item
from shapely.geometry import Point

from instageo.data.hls_utils import HLSRasterPipeline
from instageo.data.raster_chip_creator import main

FLAGS = flags.FLAGS


@pytest.fixture
def setup_and_teardown_output_dir():
    output_dir = "/tmp/test_raster_chip_creator"
    os.makedirs(output_dir, exist_ok=True)
    yield output_dir
    shutil.rmtree(output_dir)


@pytest.fixture
def sample_records():
    data = pd.DataFrame(
        {
            "date": ["2022-06-08", "2022-06-08"],
            "x": [44.48, 44.48865],
            "y": [15.115617, 15.099767],
            "label_filename": ["mask_1.tif", "mask_2.tif"],
            "stac_items_str": ["20220608_38PMB", "20220608_38PMB"],
            "mgrs_tile_id": ["38PMB", "38PMB"],
        }
    )
    data["date"] = pd.to_datetime(data["date"])
    gdf = gpd.GeoDataFrame(
        data, geometry=[Point(xy) for xy in zip(data.x, data.y)], crs="EPSG:4326"
    )
    return gdf


@pytest.fixture
def mock_hls_dataset():
    # Create a proper STAC Item
    item = Item(
        id="HLS.S30.T38PMB.2022145T072619.v2.0",
        geometry=None,
        bbox=None,
        datetime=datetime(2022, 1, 1, 0, 0, 0),
        properties={},
    )

    # Create proper Asset objects
    b02_asset = Asset(href="test_b02.tif", media_type="image/tiff")
    b03_asset = Asset(href="test_b03.tif", media_type="image/tiff")
    fmask_asset = Asset(href="test_fmask.tif", media_type="image/tiff")

    # Add assets to item
    item.add_asset("B02", b02_asset)
    item.add_asset("B03", b03_asset)
    item.add_asset("Fmask", fmask_asset)

    return {"20220608_38PMB": {"granules": [item.to_dict()]}}


def test_hls_raster_pipeline_init():
    pipeline = HLSRasterPipeline(
        output_directory="/tmp/test",
        chip_size=256,
        raster_path="/tmp/raster",
        mask_types=["cloud"],
        masking_strategy="each",
        src_crs=4326,
        spatial_resolution=30.0,
        qa_check=True,
    )
    assert pipeline.output_directory == "/tmp/test"
    assert pipeline.chip_size == 256
    assert pipeline.raster_path == "/tmp/raster"
    assert pipeline.mask_types == ["cloud"]
    assert pipeline.masking_strategy == "each"
    assert pipeline.src_crs == 4326
    assert pipeline.spatial_resolution == 30.0
    assert pipeline.qa_check is True


@patch("instageo.data.raster_chip_creator.HLSRasterPipeline")
@patch("instageo.data.raster_chip_creator.create_records_with_items")
@patch("instageo.data.hls_utils.add_hls_stac_items")
@patch("instageo.data.raster_chip_creator.Client.open")
def test_main_hls_pipeline(
    mock_client,
    mock_add_items,
    mock_create_records,
    mock_pipeline,
    setup_and_teardown_output_dir,
    sample_records,
    mock_hls_dataset,
):
    # Setup mocks
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_add_items.return_value = sample_records
    mock_create_records.return_value = (sample_records, mock_hls_dataset)
    mock_pipeline_instance = MagicMock()
    mock_pipeline.return_value = mock_pipeline_instance

    # Create test files
    os.makedirs(os.path.join(setup_and_teardown_output_dir, "chips"), exist_ok=True)
    os.makedirs(os.path.join(setup_and_teardown_output_dir, "seg_maps"), exist_ok=True)

    # Save sample records
    records_file = os.path.join(setup_and_teardown_output_dir, "test_records.gpkg")
    sample_records.to_file(records_file, driver="GPKG")

    # Parse flags
    flags.FLAGS.unparse_flags()
    flags.FLAGS(
        [
            "raster_chip_creator",
            "--records_file",
            records_file,
            "--raster_path",
            "/tmp/raster",
            "--output_directory",
            setup_and_teardown_output_dir,
            "--data_source",
            "HLS",
        ]
    )

    # Run main function
    main(None)

    # Verify pipeline was called
    mock_pipeline.assert_called_once()


@patch("instageo.data.raster_chip_creator.Client.open")
@patch("instageo.data.hls_utils.add_hls_stac_items")
@patch("instageo.data.raster_chip_creator.create_records_with_items")
@patch("instageo.data.raster_chip_creator.HLSRasterPipeline")
def test_main_hls_pipeline_existing_dataset(
    mock_pipeline,
    mock_create_records,
    mock_add_items,
    mock_client,
    setup_and_teardown_output_dir,
    sample_records,
    mock_hls_dataset,
):
    # Setup mocks
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_add_items.return_value = sample_records
    mock_create_records.return_value = (sample_records, mock_hls_dataset)
    mock_pipeline_instance = MagicMock()
    mock_pipeline.return_value = mock_pipeline_instance

    # Create existing dataset files
    with open(os.path.join(setup_and_teardown_output_dir, "hls_dataset.json"), "w") as f:
        import json

        json.dump(mock_hls_dataset, f)

    sample_records.to_file(
        os.path.join(setup_and_teardown_output_dir, "filtered_obsv_records.gpkg"),
        driver="GPKG",
    )

    # Save sample records
    records_file = os.path.join(setup_and_teardown_output_dir, "test_records.gpkg")
    sample_records.to_file(records_file, driver="GPKG")

    # Parse flags
    flags.FLAGS.unparse_flags()
    flags.FLAGS(
        [
            "raster_chip_creator",
            "--records_file",
            records_file,
            "--raster_path",
            "/tmp/raster",
            "--output_directory",
            setup_and_teardown_output_dir,
            "--data_source",
            "HLS",
        ]
    )

    # Run main function
    main(None)

    # Verify existing dataset was used
    mock_add_items.assert_not_called()
    mock_create_records.assert_not_called()


def test_main_unsupported_data_source(setup_and_teardown_output_dir, sample_records):
    # Save sample records
    records_file = os.path.join(setup_and_teardown_output_dir, "test_records.gpkg")
    sample_records.to_file(records_file, driver="GPKG")

    # Parse flags
    flags.FLAGS.unparse_flags()
    flags.FLAGS(
        [
            "raster_chip_creator",
            "--records_file",
            records_file,
            "--raster_path",
            "/tmp/raster",
            "--output_directory",
            setup_and_teardown_output_dir,
            "--data_source",
            "S1",
        ]
    )

    # Test with unsupported data source
    with pytest.raises(NotImplementedError):
        main(None)
