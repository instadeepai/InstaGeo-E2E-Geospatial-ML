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

import alphashape
from shapely.geometry import MultiPoint

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.cm
import plotly.graph_objects as go
import rasterio
import xarray as xr
from pyproj import CRS, Transformer
import math
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from shapely.ops import unary_union
import folium


epsg3857_to_epsg4326 = Transformer.from_crs(3857, 4326, always_xy=True)
MIN_ZOOM = 5


# def calculate_zoom(lats: list[float], lons: list[float]) -> float:
#     """Calculates the appropriate zoom level for a map given latitude and longitude extents.

#     This function estimates the zoom level needed to display a region defined by its
#     latitude and longitude boundaries on a map.  It assumes a Mercator projection
#     (used by Mapbox and similar web mapping libraries).

#     Args:
#         lats: A list of latitudes defining the region.
#         lons: A list of longitudes defining the region.

#     Returns:
#         float: The estimated zoom level. Returns 0 if no lats or lons are provided.
#     """
#     if not lats or not lons:
#         return 0

#     max_lat, min_lat = max(lats), min(lats)
#     max_lon, min_lon = max(lons), min(lons)

#     # Calculate latitude and longitude spans
#     lat_diff = max_lat - min_lat
#     lon_diff = max_lon - min_lon

#     # Estimate zoom based on both latitude and longitude ranges.
#     # This calculation is based on approximate formulas for Mercator projection
#     # and world dimensions in pixels at different zoom levels.  You can fine-tune
#     # the constant factors (15 and 22) if needed for your specific map display.
#     zoom_lat = 8 - math.log2(lat_diff)
#     zoom_lon = 10 - math.log2(lon_diff)

#     # Choose the more restrictive zoom level (the smaller one) to fit the entire area
#     return max(MIN_ZOOM, min(zoom_lat, zoom_lon))


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

    clusters = []
    sites = []
    all_lats = []
    all_lons = []
    crs = "EPSG:4326"
    for tile in tiles_to_overlay:
        if tile.endswith(".tif") or tile.endswith(".tiff"):
            xarr_dataset, _ = read_geotiff_to_xarray(tile)
            gpd_sites = get_activated_coords(xarr_dataset, threshold=0.5, coarsen=5)
            cluster = clusterize_prediction(gpd_sites)
            cluster = cluster.to_crs(crs)
            gpd_sites = gpd_sites.to_crs(crs)

            clusters.append(cluster)
            sites.append(gpd_sites)
            all_lats.extend(gpd_sites.geometry.y)
            all_lons.extend(gpd_sites.geometry.x)

    all_sites = pd.concat(sites)  # Combine all site GeoDataFrames
    bounds = all_sites.total_bounds
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    fig = folium.Map(location=[center_lat, center_lon])  # Set initial map center
    if clusters:
        all_clusters = pd.concat(clusters)
        # all_clusters = all_clusters[all_clusters.geometry.area > 0]
        all_clusters["density"] = all_clusters["count"] / all_clusters.geometry.area
        all_clusters = all_clusters.sort_values(by="density")
        all_sites = all_sites.to_crs(crs)
        all_clusters = all_clusters.to_crs(crs)
        fig = all_sites.explore(m=fig, color="black")
        fig = all_clusters.explore(
            m=fig,
            column="density",
            scheme="NaturalBreaks",
            k=5,
            cmap="YlOrRd",
            # m=fig,
            # column="density",
            # cmap="YlOrRd",
        )

    folium.LayerControl().add_to(fig)

    return fig


def get_activated_coords(
    xarr_dataset: xr.DataArray, threshold: float, coarsen: int = 5
) -> gpd.GeoDataFrame:
    # Apply where() to replace values below the threshold with NaN
    coarsen_xarr = xarr_dataset.coarsen(x=coarsen, y=coarsen, boundary="trim").mean()
    filtered_xarr = coarsen_xarr.where(coarsen_xarr > threshold)

    # Get coordinates where the values are above the threshold
    activated_coords = (
        filtered_xarr.stack(point=["x", "y"]).dropna("point").point.values
    )

    sites = pd.Series(activated_coords).apply(pd.Series)  # slow
    sites.columns = ["x", "y"]

    return gpd.GeoDataFrame(
        data=sites,
        geometry=gpd.points_from_xy(sites["x"], sites["y"]),
        crs=filtered_xarr.rio.crs,
    )


def clusterize_prediction(
    gpd_sites: gpd.GeoDataFrame,
    distance_threshold: float = 1e3,
    linkage: str = "single",
) -> gpd.GeoDataFrame:
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage=linkage,
    )
    gpd_sites["cluster"] = clustering.fit_predict(gpd_sites[["x", "y"]])

    cluster_geometry = gpd_sites.groupby("cluster").agg(
        {
            "geometry": lambda x: combine_coords(x, distance_threshold),
            "x": "size",
        }  # Adding the size aggregation
    )

    cluster_geometry = cluster_geometry.rename(
        columns={"x": "count", "combine_coords": "geometry"}
    )  # Correctly setting the column name
    cluster_geometry = gpd.GeoDataFrame(
        geometry=cluster_geometry["geometry"],
        data=cluster_geometry["count"],
        crs=gpd_sites.crs,  # Setting the crs
    )

    return cluster_geometry


def combine_coords(x, distance_threshold):
    polygon = unary_union(list(x.buffer(distance_threshold / 2)))
    # cv = polygon.convex_hull
    return polygon
