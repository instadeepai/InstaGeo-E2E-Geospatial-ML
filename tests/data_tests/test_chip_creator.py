import os
import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from absl import flags

from instageo.data.chip_creator import app, check_required_flags, main

FLAGS = flags.FLAGS

test_root = pathlib.Path(__file__).parent.resolve()


@pytest.fixture
def setup_and_teardown_output_dir():
    output_dir = "/tmp/test_chip_creator"
    os.makedirs(output_dir, exist_ok=True)
    yield output_dir
    shutil.rmtree(output_dir, ignore_errors=True)


@pytest.mark.auth
@pytest.mark.parametrize(
    "data_source, temporal_tolerance, chip_counts, tile_counts, num_bands",
    [("HLS", "1", 4, 28, 6), ("S2", "1", 3, 3, 6), ("S1", "3", 2, None, 2)],
)
def test_chip_creator(
    setup_and_teardown_output_dir,
    data_source,
    temporal_tolerance,
    chip_counts,
    tile_counts,
    num_bands,
):
    output_directory = setup_and_teardown_output_dir
    flags.FLAGS.unparse_flags()
    flags.FLAGS(
        [
            "chip_creator",
            "--dataframe_path",
            os.path.join(os.path.dirname(test_root), "data/test_breeding_data.csv"),
            "--output_directory",
            output_directory,
            "--min_count",
            "4",
            "--chip_size",
            "256",
            "--temporal_tolerance",
            temporal_tolerance,
            "--temporal_step",
            "30",
            "--num_steps",
            "1",
            "--data_source",
            data_source,
            "--masking_strategy",
            "any",
            "--mask_types",
            "water",
            "--processing_method",
            "download",
            "--cloud_coverage",
            "30",
        ]
    )
    main(None)

    chips = os.listdir(os.path.join(output_directory, "chips"))
    seg_maps = os.listdir(os.path.join(output_directory, "seg_maps"))
    assert len(chips) == len(seg_maps)
    assert len(chips) == chip_counts

    chip_path = os.path.join(output_directory, "chips", chips[0])
    seg_map_path = os.path.join(output_directory, "seg_maps", seg_maps[0])
    chip = xr.open_dataset(chip_path)
    seg_map = xr.open_dataset(seg_map_path)

    assert chip.band_data.shape == (num_bands, 256, 256)
    assert np.unique(chip.band_data).size > 1
    assert seg_map.band_data.shape == (1, 256, 256)
    assert np.unique(seg_map.band_data).size > 1

    if data_source != "S1":
        assert (
            len(
                pd.read_csv(
                    os.path.join(
                        output_directory,
                        f"{data_source.lower()}_granules_to_download.csv",
                    )
                )
            )
            == tile_counts
        )


@pytest.mark.auth
def test_chip_creator_download_only(setup_and_teardown_output_dir):
    output_directory = setup_and_teardown_output_dir
    flags.FLAGS.unparse_flags()
    flags.FLAGS(
        [
            "chip_creator",
            "--dataframe_path",
            os.path.join(os.path.dirname(test_root), "data/test_breeding_data.csv"),
            "--output_directory",
            output_directory,
            "--min_count",
            "4",
            "--chip_size",
            "256",
            "--temporal_tolerance",
            "1",
            "--temporal_step",
            "30",
            "--num_steps",
            "1",
            "--processing_method",
            "download-only",
            "--data_source",
            "HLS",
            "--cloud_coverage",
            "30",
        ]
    )
    main(None)

    assert os.path.exists(os.path.join(output_directory, "hls_dataset.json"))
    assert os.path.exists(os.path.join(output_directory, "hls_granules_to_download.csv"))
    assert not os.path.exists(os.path.join(output_directory, "chips"))
    assert not os.path.exists(os.path.join(output_directory, "seg_maps"))


@pytest.mark.auth
def test_chip_creator_cog(setup_and_teardown_output_dir):
    output_directory = setup_and_teardown_output_dir
    flags.FLAGS.unparse_flags()
    flags.FLAGS(
        [
            "chip_creator",
            "--dataframe_path",
            os.path.join(os.path.dirname(test_root), "data/test_breeding_data.csv"),
            "--output_directory",
            output_directory,
            "--min_count",
            "4",
            "--chip_size",
            "256",
            "--temporal_tolerance",
            "1",
            "--temporal_step",
            "30",
            "--num_steps",
            "1",
            "--masking_strategy",
            "any",
            "--mask_types",
            "water",
            "--processing_method",
            "cog",
            "--cloud_coverage",
            "30",
        ]
    )
    main(None)

    chips = os.listdir(os.path.join(output_directory, "chips"))
    seg_maps = os.listdir(os.path.join(output_directory, "seg_maps"))
    assert len(chips) == len(seg_maps)
    assert len(chips) == 4

    chip_path = os.path.join(output_directory, "chips", chips[0])
    seg_map_path = os.path.join(output_directory, "seg_maps", seg_maps[0])
    chip = xr.open_dataset(chip_path)
    seg_map = xr.open_dataset(seg_map_path)

    assert chip.band_data.shape == (6, 256, 256)
    assert np.unique(chip.band_data).size > 1
    assert seg_map.band_data.shape == (1, 256, 256)
    assert np.unique(seg_map.band_data).size > 1


def test_missing_flags_raises_error():
    """Test missing flags."""
    flags.FLAGS.unparse_flags()
    flags.FLAGS(["test"])
    flags.FLAGS.dataframe_path = None
    flags.FLAGS.output_directory = None

    with pytest.raises(app.UsageError) as excinfo:
        check_required_flags()
    assert "Flag --dataframe_path is required" in str(
        excinfo.value
    ) or "Flag --output_directory is required" in str(excinfo.value)


def test_no_missing_flags():
    """Test correct flags."""
    flags.FLAGS.unparse_flags()
    flags.FLAGS(
        [
            "chip_creator",
            "--dataframe_path=/path/to/dataframe",
            "--output_directory=/path/to/output",
        ]
    )
    try:
        check_required_flags()
    except app.UsageError:
        pytest.fail("UsageError was raised even though no flags were missing.")
