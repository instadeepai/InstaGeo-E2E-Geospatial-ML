import xarray as xr
from rasterio.crs import CRS

from instageo.data.geo_utils import open_mf_tiff_dataset


def test_open_mf_tiff_dataset():
    band_files = {
        "band1": "tests/data/sample.tif",
        "band2": "tests/data/sample.tif",
    }
    result, crs = open_mf_tiff_dataset(band_files)
    assert isinstance(result, xr.Dataset)
    assert isinstance(crs, CRS)
    assert crs == 32613
    assert result["band_data"].shape == (2, 224, 224)
