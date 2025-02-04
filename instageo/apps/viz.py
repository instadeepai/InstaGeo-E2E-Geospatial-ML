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

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.cm
import plotly.graph_objects as go
import rasterio
import xarray as xr
from pyproj import CRS, Transformer
import math
import numpy as np

epsg3857_to_epsg4326 = Transformer.from_crs(3857, 4326, always_xy=True)
MIN_ZOOM = 5


def calculate_zoom(lats: list[float], lons: list[float]) -> float:
    """Calculates the appropriate zoom level for a map given latitude and longitude extents.

    This function estimates the zoom level needed to display a region defined by its
    latitude and longitude boundaries on a map.  It assumes a Mercator projection
    (used by Mapbox and similar web mapping libraries).

    Args:
        lats: A list of latitudes defining the region.
        lons: A list of longitudes defining the region.

    Returns:
        float: The estimated zoom level. Returns 0 if no lats or lons are provided.
    """
    if not lats or not lons:
        return 0

    max_lat, min_lat = max(lats), min(lats)
    max_lon, min_lon = max(lons), min(lons)

    # Calculate latitude and longitude spans
    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon

    # Estimate zoom based on both latitude and longitude ranges.
    # This calculation is based on approximate formulas for Mercator projection
    # and world dimensions in pixels at different zoom levels.  You can fine-tune
    # the constant factors (15 and 22) if needed for your specific map display.
    zoom_lat = 8 - math.log2(lat_diff)
    zoom_lon = 10 - math.log2(lon_diff)

    # Choose the more restrictive zoom level (the smaller one) to fit the entire area
    return max(MIN_ZOOM, min(zoom_lat, zoom_lon))


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
        (
            coords_lon_min,
            coords_lon_max,
        ),
        (
            coords_lat_min,
            coords_lat_max,
        ),
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


# @lru_cache(maxsize=32)
def read_geotiff_to_xarray(filepath: str) -> tuple[xr.Dataset, CRS]:
    """Read a GeoTIFF file into an xarray Dataset.

    Args:
        filepath (str): Path to the GeoTIFF file.

    Returns:
        xr.Dataset: The loaded xarray dataset.
    """
    return xr.open_dataset(filepath).sel(band=1), get_crs(filepath)


def create_map_with_geotiff_tiles(
    tiles_to_overlay: list[str],
) -> tuple[go.Figure, dict[str, xr.Dataset]]:
    """Create a map with multiple GeoTIFF tiles overlaid, centered on the tiles' extent."""

    fig = go.Figure(go.Scattermapbox())
    mapbox_layers = []
    all_lats = []
    all_lons = []
    all_rasters = {}

    for tile in tiles_to_overlay:
        if tile.endswith(".tif") or tile.endswith(".tiff"):
            xarr_dataset, crs = read_geotiff_to_xarray(tile)
            img, coordinates = add_raster_to_plotly_figure(xarr_dataset, crs)
            coordinates_np = np.array(coordinates)
            all_rasters[tile] = xarr_dataset

            # Extract lat/lon from coordinates
            all_lons.extend(coordinates_np[:, 0])
            all_lats.extend(coordinates_np[:, 1])

            mapbox_layers.append(
                {"sourcetype": "image", "source": img, "coordinates": coordinates}
            )

    # Calculate center and zoom based on all tile extents
    if all_lats and all_lons:  # Check if any tiles were added
        center_lat = (min(all_lats) + max(all_lats)) / 2
        center_lon = (min(all_lons) + max(all_lons)) / 2
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",  # Or "satellite", "satellite-streets", etc.
                center=go.layout.mapbox.Center(lat=center_lat, lon=center_lon),
                #  Adjust zoom as needed based on data extent
                zoom=calculate_zoom(
                    all_lats, all_lons
                ),  # Or calculate zoom based on lat/lon range if you want to automate it further
            ),
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
        )
    else:  # Default center and zoom if no tiles are provided
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(center=go.layout.mapbox.Center(lat=0, lon=20), zoom=2.0),
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
        )

    fig.update_layout(mapbox_layers=mapbox_layers)
    return fig, all_rasters
