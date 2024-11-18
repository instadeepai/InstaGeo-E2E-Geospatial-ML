import os
import shutil

import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS

from instageo.data.geo_utils import get_tiles
from instageo.data.hls_pipeline import (
    add_hls_granules,
    create_hls_dataset,
    parallel_download,
)


@pytest.fixture
def observation_data():
    data = pd.DataFrame(
        {
            "date": {
                0: "2022-06-08",
                1: "2022-06-08",
                2: "2022-06-08",
                3: "2022-06-08",
                4: "2022-06-09",
                5: "2022-06-09",
                6: "2022-06-09",
                7: "2022-06-08",
                8: "2022-06-09",
                9: "2022-06-09",
            },
            "x": {
                0: 44.48,
                1: 44.48865,
                2: 46.437787,
                3: 49.095545,
                4: -0.1305,
                5: 44.6216,
                6: 49.398908,
                7: 44.451435,
                8: 49.435228,
                9: 44.744167,
            },
            "y": {
                0: 15.115617,
                1: 15.099767,
                2: 14.714659,
                3: 16.066929,
                4: 28.028967,
                5: 16.16195,
                6: 16.139727,
                7: 15.209633,
                8: 16.151837,
                9: 15.287778,
            },
            "year": {
                0: 2022,
                1: 2022,
                2: 2022,
                3: 2022,
                4: 2022,
                5: 2022,
                6: 2022,
                7: 2022,
                8: 2022,
                9: 2022,
            },
        }
    )
    data["date"] = pd.to_datetime(data["date"])
    data["input_features_date"] = data["date"]
    return data


@pytest.fixture
def setup_and_teardown_output_dir():
    output_dir = "/tmp/test_hls"
    os.makedirs(output_dir, exist_ok=True)
    yield
    shutil.rmtree(output_dir)


def test_add_hls_granules(observation_data):
    data = get_tiles(observation_data, min_count=3)
    result = add_hls_granules(data)
    assert list(result["hls_tiles"]) == [
        [
            "HLS.L30.T38PMB.2022154T072604.v2.0",
            "HLS.S30.T38PMB.2022145T072619.v2.0",
            "HLS.L30.T38PMB.2022139T071922.v2.0",
        ],
        [
            "HLS.L30.T38PMB.2022154T072604.v2.0",
            "HLS.S30.T38PMB.2022145T072619.v2.0",
            "HLS.L30.T38PMB.2022139T071922.v2.0",
        ],
        [
            "HLS.L30.T38PMB.2022154T072604.v2.0",
            "HLS.S30.T38PMB.2022145T072619.v2.0",
            "HLS.L30.T38PMB.2022139T071922.v2.0",
        ],
        [
            "HLS.L30.T38PMB.2022155T071923.v2.0",
            "HLS.S30.T38PMB.2022145T072619.v2.0",
            "HLS.L30.T38PMB.2022139T071922.v2.0",
        ],
    ]


def test_create_hls_dataset(observation_data):
    data = get_tiles(observation_data, min_count=3)
    data_with_tiles = add_hls_granules(
        data, num_steps=3, temporal_step=10, temporal_tolerance=5
    )
    hls_dataset, tiles_to_download = create_hls_dataset(data_with_tiles, outdir="")
    assert len(tiles_to_download) == 28
    assert len(hls_dataset) == 2
    assert len(hls_dataset["2022-06-08_T38PMB"]["tiles"]) == 18
    assert len(hls_dataset["2022-06-08_T38PMB"]["fmasks"]) == 3
    assert (
        hls_dataset["2022-06-08_T38PMB"]["tiles"]["B02_0"]
        == "hls_tiles/HLS.L30.T38PMB.2022154T072604.v2.0.B02.tif"
    )
    assert (
        hls_dataset["2022-06-08_T38PMB"]["fmasks"]["Fmask_0"]
        == "hls_tiles/HLS.L30.T38PMB.2022154T072604.v2.0.Fmask.tif"
    )
    assert (
        hls_dataset["2022-06-08_T38PMB"]["tiles"]["B02_1"]
        == "hls_tiles/HLS.S30.T38PMB.2022145T072619.v2.0.B02.tif"
    )
    assert (
        hls_dataset["2022-06-08_T38PMB"]["fmasks"]["Fmask_1"]
        == "hls_tiles/HLS.S30.T38PMB.2022145T072619.v2.0.Fmask.tif"
    )
    assert (
        hls_dataset["2022-06-08_T38PMB"]["tiles"]["B02_2"]
        == "hls_tiles/HLS.L30.T38PMB.2022139T071922.v2.0.B02.tif"
    )
    assert (
        hls_dataset["2022-06-08_T38PMB"]["fmasks"]["Fmask_2"]
        == "hls_tiles/HLS.L30.T38PMB.2022139T071922.v2.0.Fmask.tif"
    )


@pytest.mark.auth
def test_download_hls_tile():
    urls = [
        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B01.tif"  # noqa
    ]
    parallel_download(urls, outdir="/tmp")
    out_filename = "/tmp/HLS.L30.T38PMB.2022139T071922.v2.0.B01.tif"  # noqa
    assert os.path.exists(out_filename)
    src = rasterio.open(out_filename)
    assert isinstance(src.crs, CRS)


@pytest.mark.auth
def test_download_hls_tile_with_retry(setup_and_teardown_output_dir):
    outdir = "/tmp/test_hls"
    open(
        os.path.join(outdir, "HLS.L30.T38PMB.2022139T071922.v2.0.B02.tif"), "w"
    ).close()
    urls = {
        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B03.tif",  # noqa
        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T38PMB.2022139T071922.v2.0/HLS.L30.T38PMB.2022139T071922.v2.0.B02.tif",  # noqa
    }
    parallel_download(urls, outdir=outdir)
    out_filename = os.path.join(outdir, "HLS.L30.T38PMB.2022139T071922.v2.0.B02.tif")
    assert os.path.exists(out_filename)
    src = rasterio.open(out_filename)
    assert isinstance(src.crs, CRS)
