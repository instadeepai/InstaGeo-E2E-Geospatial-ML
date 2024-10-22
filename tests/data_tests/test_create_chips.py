import os
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from instageo.data.chip_creator import create_and_save_chips_with_seg_maps


@pytest.fixture
def setup_and_teardown_output_dir():
    output_dir = "/tmp/output"
    os.makedirs(output_dir, exist_ok=True)
    yield
    shutil.rmtree(output_dir)


def test_create_chips(setup_and_teardown_output_dir):
    geotiff_path = "tests/data/HLS.S30.T38PMB.2022145T072619.v2.0.B02.tif"
    fmask_path = "tests/data/fmask.tif"
    chip_size = 64
    output_directory = "/tmp/output"
    no_data_value = -1
    df = pd.read_csv("tests/data/sample_4326.csv")
    df["date"] = pd.to_datetime("2020-01-01")
    os.makedirs(os.path.join(output_directory, "chips"), exist_ok=True)
    os.makedirs(os.path.join(output_directory, "seg_maps"), exist_ok=True)
    chips, labels = create_and_save_chips_with_seg_maps(
        {
            "tiles": {"B02_0": geotiff_path, "B04_0": geotiff_path},
            "fmasks": {"Fmask_0": fmask_path},
        },
        df,
        chip_size,
        output_directory,
        no_data_value,
        src_crs=4326,
        mask_cloud=False,
        water_mask=False,
    )
    num_chips = len(chips)

    assert num_chips == 3
    for i in range(num_chips):
        chip_path = os.path.join(
            output_directory, "chips", "chip_20200101_S30_T38PMB_2022145T072619_1_2.tif"
        )
        seg_map_path = os.path.join(
            output_directory,
            "seg_maps",
            "seg_map_20200101_S30_T38PMB_2022145T072619_1_2.tif",
        )
        chip = xr.open_dataset(chip_path)
        seg_map = xr.open_dataset(seg_map_path)

        assert chip.band_data.shape == (2, 64, 64)
        assert np.unique(chip.band_data).size > 1
        assert seg_map.band_data.shape == (1, 64, 64)
        assert np.unique(seg_map.band_data).size > 1
