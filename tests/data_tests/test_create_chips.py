import os
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from instageo.data.data_pipeline import create_and_save_chips_with_seg_maps
from instageo.data.hls_utils import open_mf_tiff_dataset


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
        "download",
        {
            "tiles": {"B02_0": geotiff_path, "B04_0": geotiff_path},
            "fmasks": {"Fmask_0": fmask_path},
        },
        "38PLB",
        df,
        chip_size,
        output_directory,
        no_data_value,
        src_crs=4326,
        mask_types=["water"],
        masking_strategy="any",
        window_size=0,
    )
    num_chips = len(chips)
    assert num_chips == 3
    for i in range(num_chips):
        chip_path = os.path.join(
            output_directory, "chips", "chip_20200101_38PLB_1_2.tif"
        )
        seg_map_path = os.path.join(
            output_directory,
            "seg_maps",
            "seg_map_20200101_38PLB_1_2.tif",
        )
        chip = xr.open_dataset(chip_path)
        seg_map = xr.open_dataset(seg_map_path)

        assert chip.band_data.shape == (2, 64, 64)
        assert np.unique(chip.band_data).size > 1
        assert seg_map.band_data.shape == (1, 64, 64)
        assert np.unique(seg_map.band_data).size > 1


def test_seg_map_validity():
    geotiff_path = "tests/data/HLS.S30.T38PMB.2022145T072619.v2.0.B02.tif"
    fmask_path = "tests/data/fmask.tif"
    chip_size = 64
    output_directory = "/tmp/output"
    no_data_value = -1
    hls_tile_dict = {
        "tiles": {"B02_0": geotiff_path},
        "fmasks": {"Fmask_0": fmask_path},
    }
    label_val = 1
    obsv = pd.DataFrame(
        dict(
            x=[-107.11335902745832],
            y=[41.37894599955397],
            label=[label_val],
            date=[pd.to_datetime("2022-06-01")],
        )
    )

    # Test for different window sizes (1*1, 3*3, 5*5)
    for window_size in range(3):
        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(os.path.join(output_directory, "chips"), exist_ok=True)
        os.makedirs(os.path.join(output_directory, "seg_maps"), exist_ok=True)
        _, labels = create_and_save_chips_with_seg_maps(
            processing_method="download",
            hls_tile_dict=hls_tile_dict,
            df=obsv,
            chip_size=chip_size,
            output_directory=output_directory,
            no_data_value=no_data_value,
            src_crs=4326,
            mask_types=["water"],
            masking_strategy="any",
            window_size=window_size,
        )

        num_seg_maps = len(labels)
        assert num_seg_maps == 1

        seg_map_path = os.path.join(
            output_directory,
            "seg_maps",
            "seg_map_20220601_S30_T38PMB_2022145T072619_2_0.tif",
        )
        seg_map = xr.open_dataset(seg_map_path)
        assert seg_map.band_data.shape == (1, 64, 64)

        obsv_values_count_ref = (2 * window_size + 1) ** 2
        assert (seg_map.band_data.values == label_val).sum() == obsv_values_count_ref
        shutil.rmtree(output_directory)
