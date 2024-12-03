import os
import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from absl import flags

from instageo.data import chip_creator
from instageo.data.chip_creator import app, check_required_flags

FLAGS = flags.FLAGS

test_root = pathlib.Path(__file__).parent.resolve()


@pytest.fixture
def setup_and_teardown_output_dir():
    output_dir = "/tmp/csv_chip_creator"
    os.makedirs(output_dir, exist_ok=True)
    yield
    try:
        shutil.rmtree(os.path.join(output_dir, "chips"))
        shutil.rmtree(os.path.join(output_dir, "seg_maps"))
        os.remove(os.path.join(output_dir, "granules_to_download.csv"))
        os.remove(os.path.join(output_dir, "hls_dataset.json"))
        os.remove(os.path.join(output_dir, "hls_chips_dataset.csv"))
    except FileNotFoundError:
        pass


@pytest.mark.auth
def test_chip_creator(setup_and_teardown_output_dir):
    output_directory = "/tmp/csv_chip_creator"
    argv = [
        "chip_creator",
        "--dataframe_path",
        os.path.join(os.path.dirname(test_root), "data/test_breeding_data.csv"),
        "--output_directory",
        output_directory,
        "--min_count",
        "4",
        "--chip_size",
        "512",
        "--no_data_value",
        "-1",
        "--temporal_tolerance",
        "1",
        "--temporal_step",
        "30",
        "--num_steps",
        "1",
        "--data_source",
        "HLS",
    ]
    FLAGS(argv)
    chip_creator.main("None")
    chips = os.listdir(os.path.join(output_directory, "chips"))
    seg_maps = os.listdir(os.path.join(output_directory, "seg_maps"))
    assert len(chips) == len(seg_maps)
    assert len(chips) == 3
    chip_path = os.path.join(output_directory, "chips", chips[0])
    seg_map_path = os.path.join(output_directory, "seg_maps", seg_maps[0])
    chip = xr.open_dataset(chip_path)
    seg_map = xr.open_dataset(seg_map_path)
    assert chip.band_data.shape == (6, 512, 512)
    assert np.unique(chip.band_data).size > 1
    assert seg_map.band_data.shape == (1, 512, 512)
    assert np.unique(seg_map.band_data).size > 1
    assert (
        len(
            pd.read_csv(os.path.join(output_directory, "granules_to_download.csv"))[
                "tiles"
            ].tolist()
        )
        == 21
    )


@pytest.mark.auth
def test_chip_creator_download_only(setup_and_teardown_output_dir):
    output_directory = "/tmp/csv_chip_creator"
    argv = [
        "chip_creator",
        "--dataframe_path",
        os.path.join(os.path.dirname(test_root), "data/test_breeding_data.csv"),
        "--output_directory",
        output_directory,
        "--min_count",
        "4",
        "--chip_size",
        "512",
        "--no_data_value",
        "-1",
        "--temporal_tolerance",
        "1",
        "--temporal_step",
        "30",
        "--num_steps",
        "1",
        "--download_only",
        "True",
        "--data_source",
        "HLS",
    ]
    FLAGS(argv)
    chip_creator.main("None")
    assert os.path.exists(os.path.join(output_directory, "hls_dataset.json"))
    assert os.path.exists(os.path.join(output_directory, "granules_to_download.csv"))
    assert not os.path.exists(os.path.join(output_directory, "chips"))
    assert not os.path.exists(os.path.join(output_directory, "seg_maps"))


def test_missing_flags_raises_error():
    """Test missing flags."""
    FLAGS([__file__])
    with pytest.raises(app.UsageError) as excinfo:
        check_required_flags()
    assert "Flag --dataframe_path is required" in str(
        excinfo.value
    ) or "Flag --output_directory is required" in str(
        excinfo.value
    ), "Expected UsageError with a message about missing required flags"


def test_no_missing_flags():
    """Test correct flags."""
    FLAGS(
        [
            __file__,
            "--dataframe_path=/path/to/dataframe",
            "--output_directory=/path/to/output",
        ]
    )
    try:
        check_required_flags()
    except app.UsageError:
        pytest.fail("UsageError was raised even though no flags were missing.")
