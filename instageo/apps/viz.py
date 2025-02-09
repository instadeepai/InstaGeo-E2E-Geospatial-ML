# ------------------------------------------------------------------------------
# This code is licensed under the Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License.
#
# You are free to:
# - Share: Copy and redistribute the material in any medium or format
# - Adapt: Remix, transform, and build upon the material
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license,
#   and indicate if changes were made. You may do so in any reasonable manner,
#   but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
# - ShareAlike: If you remix, transform, or build upon the material, you must
#   distribute your contributions under the same license as the original.
#
# For more details, see https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------

"""Utils for Raster Visualisation."""
from functools import lru_cache
import math
import os
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.cm
import plotly.graph_objects as go
import rasterio
import xarray as xr
from pyproj import CRS, Transformer

epsg3857_to_epsg4326 = Transformer.from_crs(3857, 4326, always_xy=True)

MAP_STYLE="https://tiles.stadiamaps.com/styles/alidade_smooth.json"

def get_crs(filepath: str) -> CRS:
    """Retrieves the CRS of a GeoTiff data.

    Args:
        filepath: Path to a GeoTiff file.

    Returns:
        CRS of data stored in `filepath`
    """
    src = rasterio.open(filepath)
    return src.crs


def add_raster_to_plotly_figure(
    xarr_dataset: xr.Dataset,
    from_crs: CRS,
    column_name: str = "band_data",
    scale: float = 1.0,
) -> go.Figure:
    """Add a raster plot on a Plotly graph object figure.

    This function overlays raster data from an xarray dataset onto a Plotly map figure.
    The data is reprojected to EPSG:3857 CRS for compatibility with Mapbox's projection
    system.

    Args:
        xarr_dataset (xr.Dataset): xarray dataset containing the raster data.
        from_crs (CRS): Coordinate Reference System of data stored in xarr_dataset.
        column_name (str): Name of the column in `xarr_dataset` to be plotted. Defaults
            to "band_data".
        scale (float): Scale factor for adjusting the plot resolution. Defaults to 1.0.

    Returns:
        Figure: The modified Plotly figure with the raster data overlaid.
    """
    # Reproject to EPSG:3857 CRS
    xarr_dataset = xarr_dataset.rio.write_crs(from_crs).rio.reproject("EPSG:3857")
    xarr_dataset = xarr_dataset.where(xarr_dataset <= 1, 0)
    # Get Raster dimension and range
    numpy_data = xarr_dataset[column_name].squeeze().to_numpy()
    plot_height, plot_width = numpy_data.shape

    # Data aggregation
    canvas = ds.Canvas(
        plot_width=int(plot_width * scale), plot_height=int(plot_height * scale)
    )
    agg = canvas.raster(xarr_dataset[column_name].squeeze(), interpolate="linear")

    coords_lat_min, coords_lat_max = (
        agg.coords["y"].values.min(),
        agg.coords["y"].values.max(),
    )
    coords_lon_min, coords_lon_max = (
        agg.coords["x"].values.min(),
        agg.coords["x"].values.max(),
    )
    # xarr_dataset CRS was converted to EPSG:3857 because when EPSG:4326 is used the
    # overlaid image doesn't overlap properly resulting in misrepresentation. The actual
    # cause of this behavior is that 'Mapbox supports the popular Web Mercator
    # projection, and does not support any other projections.'
    (
        coords_lon_min,
        coords_lon_max,
    ), (
        coords_lat_min,
        coords_lat_max,
    ) = epsg3857_to_epsg4326.transform(
        [coords_lon_min, coords_lon_max], [coords_lat_min, coords_lat_max]
    )
    # Corners of the image, which need to be passed to mapbox
    coordinates = [
        [coords_lon_min, coords_lat_max],
        [coords_lon_max, coords_lat_max],
        [coords_lon_max, coords_lat_min],
        [coords_lon_min, coords_lat_min],
    ]

    # Apply color map
    img = tf.shade(
        agg,
        cmap=matplotlib.colormaps["Reds"],
        alpha=100,
        how="linear",
    )[::-1].to_pil()
    return img, coordinates

def is_tile_in_viewport(tile_bounds: dict, viewport: dict, zoom: float) -> bool:
    """Check if a tile is within the current viewport."""
    lat_min, lat_max = viewport['latitude']['min'], viewport['latitude']['max']
    lon_min, lon_max = viewport['longitude']['min'], viewport['longitude']['max']
    lat_min -=  1 - math.exp(-0.1 * zoom)
    lat_max += 1 - math.exp(-0.1 * zoom)
    lon_min -= 1 - math.exp(-0.1 * zoom)
    lon_max += 1 - math.exp(-0.1 * zoom)
    tile_lat_min, tile_lat_max = tile_bounds['lat_min'], tile_bounds['lat_max']
    tile_lon_min, tile_lon_max = tile_bounds['lon_min'], tile_bounds['lon_max']
    return not (tile_lat_max < lat_min or tile_lat_min > lat_max or
                tile_lon_max < lon_min or tile_lon_min > lon_max)

@lru_cache(maxsize = 8)
def read_geotiff_to_xarray(filepath: str) -> tuple[xr.Dataset, CRS]:
    """Read GeoTIFF file into an xarray Dataset."""
    xarr_dataset = xr.open_dataset(filepath).sel(band=1)
    crs = rasterio.open(filepath).crs
    return xarr_dataset, crs

def zoom_to_scale(zoom: float):
    zoom_dict = {1:0.1,2:0.1,3:0.1,4:0.25,5:0.5,6:0.6,7:0.1,8:0.1}
    zoom_ceiled = math.ceil(zoom)
    if zoom_ceiled in zoom_dict.keys():  
        scale = zoom_dict[zoom_ceiled]  
    else:
        scale = 1.0
    return scale


def add_raster_to_plotly_figure(xarr_dataset: xr.Dataset, from_crs: CRS, scale: float = 1.0) -> tuple:
    """Convert raster data to an image and coordinates for Plotly."""
    # Ensure the raster has the correct CRS
    xarr_dataset = xarr_dataset.rio.write_crs(from_crs).rio.reproject("EPSG:3857")
    xarr_dataset = xarr_dataset.where(xarr_dataset <= 1, 0)  # Mask values <= 1
    
    # Extract the variable containing raster data ('band_data' in this case)
    band_data = xarr_dataset['band_data']
    
    numpy_data = band_data.squeeze().to_numpy()  # Ensure the array is 2D
    plot_height, plot_width = numpy_data.shape

    canvas = ds.Canvas(plot_width=int(plot_width * scale), plot_height=int(plot_height * scale))
    
    # Use 'band_data' to aggregate
    agg = canvas.raster(band_data, interpolate="linear")  # Specify the variable to aggregate

    # Calculate coordinates for the image
    coords_lat_min, coords_lat_max = agg.coords["y"].values.min(), agg.coords["y"].values.max()
    coords_lon_min, coords_lon_max = agg.coords["x"].values.min(), agg.coords["x"].values.max()

    (coords_lon_min, coords_lon_max), (coords_lat_min, coords_lat_max) = epsg3857_to_epsg4326.transform(
        [coords_lon_min, coords_lon_max], [coords_lat_min, coords_lat_max]
    )

    coordinates = [
        [coords_lon_min, coords_lat_max],
        [coords_lon_max, coords_lat_max],
        [coords_lon_max, coords_lat_min],
        [coords_lon_min, coords_lat_min],
    ]

    # Generate the image using Datashader
    img = tf.shade(agg, cmap=matplotlib.colormaps["Reds"], alpha=100, how="linear")[::-1].to_pil()
    return img, coordinates

def read_geotiff_to_xarray(filepath: str) -> tuple[xr.Dataset, CRS]:
    """Read a GeoTIFF file into an xarray Dataset.

    Args:
        filepath (str): Path to the GeoTIFF file.

    Returns:
        xr.Dataset: The loaded xarray dataset.
    """
    return xr.open_dataset(filepath).sel(band=1), get_crs(filepath)


def create_map_with_geotiff_tiles(tile_metadata: list, viewport: dict, zoom: float, base_dir: str) -> go.Figure:
    """Create a map with multiple GeoTIFF tiles overlaid."""
    
    fig = go.Figure(go.Scattermapbox())
    fig.update_layout(
        mapbox_style=MAP_STYLE if MAP_STYLE else "open-street-map" ,
        mapbox=dict(
            center=go.layout.mapbox.Center(
                lat=(viewport["latitude"]["min"] + viewport["latitude"]["max"]) / 2,
                lon=(viewport["longitude"]["min"] + viewport["longitude"]["max"]) / 2,
            ),
            zoom=zoom,
        ),
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )
    mapbox_layers = []
    for tile in tile_metadata:
        if len(mapbox_layers) > 15:
            break
        if is_tile_in_viewport(tile['bounds'], viewport, zoom=zoom):
            tile_path = os.path.join(base_dir, tile['name'])
            xarr_dataset, crs = read_geotiff_to_xarray(tile_path)
            scale = zoom_to_scale(zoom)
            img, coordinates = add_raster_to_plotly_figure(xarr_dataset, crs, scale=scale)
            mapbox_layers.append({"sourcetype": "image", "source": img, "coordinates": coordinates})
    fig.update_layout(mapbox_layers=mapbox_layers)
    return fig
