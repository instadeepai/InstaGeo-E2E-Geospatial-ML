"""Utils for Raster Visualisation."""

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.cm
import plotly.graph_objects as go
import plotly.graph_objects as Figure
import rasterio
import xarray as xr
from pyproj import CRS, Transformer

epsg3857_to_epsg4326 = Transformer.from_crs(3857, 4326, always_xy=True)


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
) -> Figure:
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
    xarr_dataset = xarr_dataset.fillna(1)
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
        cmap=matplotlib.cm.get_cmap(name="Reds", lut=256),
        alpha=100,
        how="linear",
    )[::-1].to_pil()
    return img, coordinates


def read_geotiff_to_xarray(filepath: str) -> tuple[xr.Dataset, CRS]:
    """Read a GeoTIFF file into an xarray Dataset.

    Args:
        filepath (str): Path to the GeoTIFF file.

    Returns:
        xr.Dataset: The loaded xarray dataset.
    """
    return xr.open_dataset(filepath).sel(band=1), get_crs(filepath)


def create_map_with_geotiff_tiles(tiles_to_overlay: list[str]) -> Figure:
    """Create a map with multiple GeoTIFF tiles overlaid.

    This function reads GeoTIFF files from a specified directory and overlays them on a
    Plotly map.

    Args:
        tiles_to_overlay (list[str]): Path to tiles to overlay on map.

    Returns:
        Figure: A Plotly figure with overlaid GeoTIFF tiles.
    """
    fig = go.Figure(go.Scattermapbox())
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=go.layout.mapbox.Center(lat=0, lon=20), zoom=2.0),
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    mapbox_layers = []
    for tile in tiles_to_overlay:
        if tile.endswith(".tif") or tile.endswith(".tiff"):
            xarr_dataset, crs = read_geotiff_to_xarray(tile)
            img, coordinates = add_raster_to_plotly_figure(
                xarr_dataset, crs, "band_data", scale=0.1
            )
            mapbox_layers.append(
                {"sourcetype": "image", "source": img, "coordinates": coordinates}
            )
    # Overlay the resulting image
    fig.update_layout(mapbox_layers=mapbox_layers)
    return fig
