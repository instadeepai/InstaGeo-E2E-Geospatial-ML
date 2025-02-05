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

import matplotlib.colors
import streamlit as st

import math
import plotly.graph_objects as go
import rasterio
import xarray as xr
from pyproj import CRS, Transformer
import geopandas as gpd
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from shapely.ops import unary_union
import folium


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

    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon

    zoom_lat = 9 - math.log2(lat_diff)
    zoom_lon = 11 - math.log2(lon_diff)

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

    all_sites = gpd.GeoDataFrame(pd.concat(sites), geometry="geometry")
    bounds = all_sites.total_bounds
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    fig = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=calculate_zoom(all_lats, all_lons),
    )

    if clusters:
        all_clusters = pd.concat(clusters)
        all_clusters["density"] = all_clusters["count"] / all_clusters.geometry.area
        thresholds = [0, 10000, 20000, 70000, 200000, float("inf")]
        risk_levels = ["very low", "low", "medium", "high", "very high"]

        all_clusters["risk"] = pd.cut(
            all_clusters["density"], bins=thresholds, labels=risk_levels, right=False
        )

        colors = ["green", "lightgreen", "yellow", "orange", "red"]

        fig = all_sites.explore(m=fig, color="black")
        fig = all_clusters.explore(
            m=fig,
            column="risk",
            cmap=matplotlib.colors.ListedColormap(colors),
            legend=False,
            categorical=True,
        )

        legend_html_streamlit = ""
        for risk_level, color in zip(risk_levels, colors):
            legend_html_streamlit += f"<span style='display:inline-block;width:10px;height:10px;background-color:{color};margin-right:5px;'></span>{risk_level}<br>"

        st.markdown(
            f"<div style='position:absolute;top:100px;left:10px;z-index:1000;display:flex;flex-direction:column;padding:10px;background-color:rgba(255,255,255,0.7);'>{legend_html_streamlit}</div>",  # Modified line
            unsafe_allow_html=True,
        )

    folium.LayerControl().add_to(fig)
    return fig, all_clusters


def get_activated_coords(
    xarr_dataset: xr.DataArray, threshold: float, coarsen: int = 5
) -> gpd.GeoDataFrame:
    coarsen_xarr = xarr_dataset.coarsen(x=coarsen, y=coarsen, boundary="trim").mean()
    filtered_xarr = coarsen_xarr.where(coarsen_xarr > threshold)
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
        }
    )

    cluster_geometry = cluster_geometry.rename(
        columns={"x": "count", "combine_coords": "geometry"}
    )
    cluster_geometry = gpd.GeoDataFrame(
        geometry=cluster_geometry["geometry"],
        data=cluster_geometry["count"],
        crs=gpd_sites.crs,
    )

    return cluster_geometry


def combine_coords(x, distance_threshold):
    polygon = unary_union(list(x.buffer(distance_threshold / 2)))
    return polygon
