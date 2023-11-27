import os
import shutil

import pytest
from rasterio import open as open_tiff

from instageo.data.chip_creator import create_and_save_chips_with_seg_maps
from instageo.data.geo_utils import open_mf_tiff_dataset, read_csv_gdf


@pytest.fixture
def setup_and_teardown_output_dir():
    output_dir = "/tmp/output"
    os.makedirs(output_dir, exist_ok=True)
    yield
    shutil.rmtree(output_dir)


def test_with_fixture(setup_and_teardown_output_dir):
    geotiff_path = "tests/data/sample.tif"
    chip_size = 64
    output_directory = "/tmp/output"
    no_data_value = -1
    df = read_csv_gdf("tests/data/sample_4326.csv", dst_crs=32613)

    num_chips = create_and_save_chips_with_seg_maps(
        {"band1": geotiff_path}, df, chip_size, output_directory, no_data_value
    )

    # Load the original dataset to calculate expected number of chips
    ds = open_mf_tiff_dataset({"band1": geotiff_path})
    expected_chips_x = ds.dims["x"] // chip_size
    expected_chips_y = ds.dims["y"] // chip_size
    expected_num_chips = expected_chips_x * expected_chips_y

    # Check the number of chips created
    assert num_chips == expected_num_chips

    # Verify that the files are created
    for i in range(num_chips):
        chip_path = os.path.join(output_directory, f"chip_{i}.tif")
        seg_map_path = os.path.join(output_directory, f"seg_map_{i}.tif")
        assert os.path.exists(chip_path)
        assert os.path.exists(seg_map_path)

        # Open and check each chip and segmentation map file
        with open_tiff(chip_path) as chip_file:
            assert chip_file.width == chip_size
            assert chip_file.height == chip_size

        with open_tiff(seg_map_path) as seg_map_file:
            assert seg_map_file.width == chip_size
            assert seg_map_file.height == chip_size
