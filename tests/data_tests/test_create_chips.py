import os
import shutil

import numpy as np
import pandas as pd
import pytest
import rioxarray
import xarray as xr

from instageo.data.data_pipeline import (
    NO_DATA_VALUES,
    apply_mask,
    create_and_save_chips_with_seg_maps,
    mask_segmentation_map,
)
from instageo.data.hls_utils import decode_fmask_value, open_mf_tiff_dataset


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
    no_data_value = NO_DATA_VALUES.HLS
    df = pd.read_csv("tests/data/sample_4326.csv")
    df["date"] = pd.to_datetime("2020-01-01")
    os.makedirs(os.path.join(output_directory, "chips"), exist_ok=True)
    os.makedirs(os.path.join(output_directory, "seg_maps"), exist_ok=True)
    chips, labels = create_and_save_chips_with_seg_maps(
        data_reader=open_mf_tiff_dataset,
        mask_fn=apply_mask,
        processing_method="download",
        tile_dict={
            "tiles": {"B02_0": geotiff_path, "B04_0": geotiff_path},
            "fmasks": {"Fmask_0": fmask_path},
        },
        data_source="HLS",
        df=df,
        chip_size=chip_size,
        output_directory=output_directory,
        no_data_value=no_data_value,
        src_crs=4326,
        mask_decoder=decode_fmask_value,
        mask_types=[],
        masking_strategy="any",
        window_size=0,
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
        chip_invalid_mask = (chip == no_data_value).any(dim="band").to_array().values
        seg_no_data_value = NO_DATA_VALUES.SEG_MAP
        seg_invalid_mask = (seg_map == seg_no_data_value).to_array().values[0]
        assert np.all(seg_invalid_mask[chip_invalid_mask])


def test_segmentation_map_masking():
    chip_path = "tests/data/chip_178_022.tif"
    seg_map_path = "tests/data/chip_178_022.mask.tif"
    chip_no_data_value = -9999
    chip = rioxarray.open_rasterio(chip_path)
    seg_map = rioxarray.open_rasterio(seg_map_path).astype(chip.dtype)
    seg_map = seg_map.assign_coords(x=chip.x.values, y=chip.y.values)
    seg_map = mask_segmentation_map(chip, seg_map, chip_no_data_value)
    assert seg_map.where(seg_map != NO_DATA_VALUES.SEG_MAP).count().values == 0


def test_segmentation_map_masking_pass():
    chip = np.array([[1, 2, 3, 4], [1, 3, -9, 7], [6, 7, 3, 9]])
    chip = xr.DataArray(
        chip,
        dims=["band", "x"],
        coords={"band": np.arange(chip.shape[0]), "x": np.arange(chip.shape[1])},
    )
    seg_map = np.array([[1, -1, 1, 2]])
    seg_map = xr.DataArray(
        seg_map,
        dims=["band", "x"],
        coords={"band": np.arange(seg_map.shape[0]), "x": np.arange(seg_map.shape[1])},
    )
    seg_no_data_value = -1
    chip_no_data_value = -9

    # test each masking strategy
    seg_map = mask_segmentation_map(
        chip, seg_map, chip_no_data_value=chip_no_data_value, masking_strategy="each"
    )
    assert seg_map.where(seg_map != seg_no_data_value).count().values > 0
    np.testing.assert_array_equal(np.array([[1, -1, 1, 2]]), seg_map.values)

    # test any masking strategy
    seg_map = mask_segmentation_map(
        chip, seg_map, chip_no_data_value=chip_no_data_value, masking_strategy="any"
    )
    assert seg_map.where(seg_map != seg_no_data_value).count().values > 0
    np.testing.assert_array_equal(np.array([[1, -1, -1, 2]]), seg_map.values)


def test_segmentation_map_masking_fail():
    chip = np.array([[1, 2, 3, 4], [-9, -9, -9, -9], [6, 7, 3, 9]])
    chip = xr.DataArray(
        chip,
        dims=["band", "x"],
        coords={"band": np.arange(chip.shape[0]), "x": np.arange(chip.shape[1])},
    )
    seg_map = np.array([[1, -1, 1, 2]])
    seg_map = xr.DataArray(
        seg_map,
        dims=["band", "x"],
        coords={"band": np.arange(seg_map.shape[0]), "x": np.arange(seg_map.shape[1])},
    )
    seg_no_data_value = -1
    chip_no_data_value = -9
    seg_map = mask_segmentation_map(chip, seg_map, chip_no_data_value=chip_no_data_value)
    assert seg_map.where(seg_map != seg_no_data_value).count().values == 0
    np.testing.assert_array_equal(np.array([[-1, -1, -1, -1]]), seg_map.values)


@pytest.mark.parametrize("window_size", [0, 3, 5, 7])
def test_seg_map_validity(setup_and_teardown_output_dir, window_size):
    geotiff_path = "tests/data/HLS.S30.T38PMB.2022145T072619.v2.0.B02.tif"
    fmask_path = "tests/data/fmask.tif"
    chip_size = 64
    output_directory = "/tmp/output"
    no_data_value = -1
    df = pd.read_csv("tests/data/sample_4326.csv")
    df["date"] = pd.to_datetime("2020-01-01")
    os.makedirs(os.path.join(output_directory, "chips"), exist_ok=True)
    os.makedirs(os.path.join(output_directory, "seg_maps"), exist_ok=True)

    chips, seg_maps = create_and_save_chips_with_seg_maps(
        data_reader=open_mf_tiff_dataset,
        mask_fn=apply_mask,
        processing_method="download",
        tile_dict={
            "tiles": {"B02_0": geotiff_path, "B04_0": geotiff_path},
            "fmasks": {"Fmask_0": fmask_path},
        },
        data_source="HLS",
        df=df,
        chip_size=chip_size,
        output_directory=output_directory,
        no_data_value=no_data_value,
        src_crs=4326,
        mask_decoder=decode_fmask_value,
        mask_types=[],
        masking_strategy="any",
        window_size=window_size,
    )
    # Verify chips and segmentation maps exist and match expectations
    assert len(chips) > 0
    assert len(seg_maps) > 0

    for chip_name, seg_map_name in zip(chips, seg_maps):
        chip_path = os.path.join(output_directory, "chips", chip_name)
        seg_map_path = os.path.join(output_directory, "seg_maps", seg_map_name)

        assert os.path.exists(chip_path), f"Chip file missing: {chip_path}"
        assert os.path.exists(seg_map_path), f"Segmentation map file missing: {seg_map_path}"

        chip = xr.open_dataset(chip_path)
        seg_map = xr.open_dataset(seg_map_path)

        assert chip.band_data.shape == (2, chip_size, chip_size), "Chip shape mismatch"
        assert np.unique(chip.band_data).size > 1, "Chip contains only one unique value"

        assert seg_map.band_data.shape == (
            1,
            chip_size,
            chip_size,
        ), "Segmentation map shape mismatch"
        assert (
            np.unique(seg_map.band_data).size > 1
        ), "Segmentation map contains only one unique value"
