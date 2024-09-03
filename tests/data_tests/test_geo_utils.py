import pytest
import xarray as xr
from rasterio.crs import CRS

from instageo.data.geo_utils import decode_fmask_value, open_mf_tiff_dataset


def test_open_mf_tiff_dataset():
    band_files = {
        "tiles": {
            "band1": "tests/data/sample.tif",
            "band2": "tests/data/sample.tif",
        },
        "fmasks": {
            "band1": "tests/data/fmask.tif",
            "band2": "tests/data/fmask.tif",
        },
    }

    result, crs = open_mf_tiff_dataset(band_files, mask_cloud=False)
    assert isinstance(result, xr.Dataset)
    assert isinstance(crs, CRS)
    assert crs == 32613
    assert result["band_data"].shape == (2, 224, 224)


def test_open_mf_tiff_dataset_cloud_mask():
    band_files = {
        "tiles": {
            "band1": "tests/data/sample.tif",
            "band2": "tests/data/sample.tif",
        },
        "fmasks": {
            "band1": "tests/data/fmask.tif",
            "band2": "tests/data/fmask.tif",
        },
    }
    result_no_mask, crs = open_mf_tiff_dataset(band_files, mask_cloud=False)
    num_points = result_no_mask.band_data.count().values.item()
    result_with_mask, crs = open_mf_tiff_dataset(band_files, mask_cloud=True)
    fmask = xr.open_dataset("tests/data/fmask.tif")
    cloud_mask = decode_fmask_value(fmask, 1)
    num_clouds = cloud_mask.where(cloud_mask == 1).band_data.count().values.item()
    assert (
        result_with_mask.band_data.count().values.item() == num_points - 2 * num_clouds
    )
    assert isinstance(result_with_mask, xr.Dataset)
    assert isinstance(crs, CRS)
    assert crs == 32613
    assert result_with_mask["band_data"].shape == (2, 224, 224)


@pytest.mark.parametrize(
    "value, position, result",
    [
        (100, 0, 0),
        (100, 1, 0),
        (100, 2, 1),
        (100, 3, 0),
        (100, 4, 0),
        (100, 5, 1),
        (100, 6, 1),
        (100, 7, 0),
    ],
)
def test_decode_fmask_value(value, position, result):
    assert decode_fmask_value(value, position) == result
