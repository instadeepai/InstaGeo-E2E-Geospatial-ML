import numpy as np

# import plotly.graph_objects as go
import xarray as xr
from PIL import Image

from instageo.apps.viz import (  # create_map_with_geotiff_tiles,
    add_raster_to_plotly_figure,
    get_crs,
    read_geotiff_to_xarray,
)


def test_read_geotiff_to_xarray():
    data, crs = read_geotiff_to_xarray("tests/data/sample.tif")
    assert isinstance(data, xr.Dataset)
    assert crs == 32613


def test_get_crs():
    assert get_crs("tests/data/sample.tif") == 32613


def test_get_image_and_coordinates():
    ds, crs = read_geotiff_to_xarray("tests/data/sample.tif")
    img, coords = add_raster_to_plotly_figure(ds, crs)
    assert isinstance(img, Image.Image)
    np.testing.assert_almost_equal(
        coords,
        [
            [-107.16932831572083, 41.38046859355541],
            [-107.08744863572721, 41.38046859355541],
            [-107.08744863572721, 41.31873255032166],
            [-107.16932831572083, 41.31873255032166],
        ],
    )


# def test_create_map_with_geotiff_tiles():
#     fig = create_map_with_geotiff_tiles(["tests/data/sample.tif"])
#     assert isinstance(fig, go.Figure)
#     assert len(fig.layout.mapbox.layers) == 1
