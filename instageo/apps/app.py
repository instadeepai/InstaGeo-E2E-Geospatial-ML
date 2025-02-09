import os
import json
import dash
import plotly.graph_objects as go
import math
from dash import dcc, html
from dash.dependencies import Input, Output, State
from functools import lru_cache

import rasterio
import xarray as xr
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.cm
from pathlib import Path
from pyproj import CRS, Transformer
# Initialize Dash App
app = dash.Dash(__name__)
server = app.server  # Needed for deployment

# Transformer for coordinate conversion
epsg3857_to_epsg4326 = Transformer.from_crs(3857, 4326, always_xy=True)
relayoutData = 10
# Default viewport
default_viewport = {
    "latitude": {"min": -2.91, "max": -1.13},
    "longitude": {"min": 29.02, "max": 30.81},
}
default_zoom = 5.0
MAP_STYLE="https://tiles.stadiamaps.com/styles/alidade_smooth.json"
def load_tile_metadata(json_path: str) -> list:
    """Load tile metadata from a JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)

def is_tile_in_viewport(tile_bounds: dict, viewport: dict, zoom: float) -> bool:
    """Check if a tile is within the current viewport."""
    lat_min, lat_max = viewport['latitude']['min'], viewport['latitude']['max']
    lon_min, lon_max = viewport['longitude']['min'], viewport['longitude']['max']
    lat_min -= 1/zoom
    lat_max += 1/zoom
    lon_min -= 1/zoom
    lon_max += 1/zoom
    tile_lat_min, tile_lat_max = tile_bounds['lat_min'], tile_bounds['lat_max']
    tile_lon_min, tile_lon_max = tile_bounds['lon_min'], tile_bounds['lon_max']
    return not (tile_lat_max < lat_min or tile_lat_min > lat_max or
                tile_lon_max < lon_min or tile_lon_min > lon_max)

@lru_cache(maxsize = 16)
def read_geotiff_to_xarray(filepath: str) -> tuple[xr.Dataset, CRS]:
    """Read GeoTIFF file into an xarray Dataset."""
    xarr_dataset = xr.open_dataset(filepath).sel(band=1)
    crs = rasterio.open(filepath).crs
    return xarr_dataset, crs
# get scale based on zoom level
# def zoom_to_scale(zoom: float, growth_rate: float = 0.2) -> float:
#     scale = 1 - math.exp(-growth_rate * zoom)
#     return round(scale, 3)

def zoom_to_scale(zoom: float):
    zoom_dict = {1:0.1,2:0.15,3:0.2,4:0.25,5:0.5,6:0.6,7:0.7,8:0.8}
    zoom_ceiled = math.ceil(zoom)
    if zoom_ceiled in zoom_dict.keys():  
        scale = zoom_dict[zoom_ceiled]  
    else:
        scale = 1.0
    return scale

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
        if is_tile_in_viewport(tile['bounds'], viewport, zoom=zoom):
            tile_path = os.path.join(base_dir, tile['name'])
            xarr_dataset, crs = read_geotiff_to_xarray(tile_path)
            scale = zoom_to_scale(zoom)
            print("--zoom--", zoom)
            print("----sclale",scale)
            img, coordinates = add_raster_to_plotly_figure(xarr_dataset, crs, scale=scale if zoom > 8 else 0.5)
            mapbox_layers.append({"sourcetype": "image", "source": img, "coordinates": coordinates})
    fig.update_layout(mapbox_layers=mapbox_layers)
    return fig

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


# Layout with Sidebar
app.layout = html.Div([
    dcc.Store(id="stored-viewport", data=default_viewport),
    
    html.Div([
        html.H2("Settings", style={"textAlign": "center", "padding": "10px", "color": "white"}),
        html.Label("GeoTIFF Directory:", style={"color": "white"}),
        dcc.Input(id="directory", type="text", placeholder="Enter directory path", 
                  style={"width": "97%", "marginBottom": "10px", "padding": "5px", "borderRadius": "5px"}),
        html.Label("Year:", style={"color": "white"}),
        dcc.Dropdown(id="year", options=[{"label": str(y), "value": y} for y in range(2023, 2025)], 
                     value=2023, style={"width": "100%","marginBottom": "10px", "borderRadius": "5px"}),
        html.Label("Month:", style={"color": "white"}),
        dcc.Dropdown(id="month", options=[{"label": str(m), "value": m} for m in range(1, 13)], 
                     value=6, style={"marginBottom": "10px"}),
    ], style={"width": "20%", "backgroundColor": "#343a40", "position": "fixed", "height": "100vh", 
              "padding": "20px", "top": 0, "left": 0, "boxShadow": "2px 0px 10px rgba(0,0,0,0.2)", "borderRadius": "0px 10px 10px 0px"}),
    
    html.Div([
        html.H1("Dash GeoTIFF Viewer", style={"textAlign": "center", "marginTop": "20px", "color": "#343a40"}),
        dcc.Graph(id="map", config={"scrollZoom": True}, style={"width": "78vw", "height": "90vh", "borderRadius": "10px", "boxShadow": "0px 4px 10px rgba(0,0,0,0.2)"}),
    ], style={"marginLeft": "22%", "padding": "20px", "backgroundColor": "#f8f9fa", "height": "100vh"}),
])

tile_metadata_path = "instageo/apps/tile_metadata.json"
tile_metadata = load_tile_metadata(tile_metadata_path) 
# Callback to update the map
@app.callback(
    Output("map", "figure"),
    Input("map", "relayoutData"),
    Input("directory", "value"),
    Input("year", "value"),
    Input("month", "value"),
    State("stored-viewport", "data"),
)
def update_map(relayout_data, directory, year, month, current_viewport):
    """Update the map based on viewport, zoom, and directory selection."""
    if not directory or not Path(directory).is_dir():
        fig = go.Figure(go.Scattermapbox())
        fig.update_layout(
        mapbox_style=MAP_STYLE if MAP_STYLE else "open-street-map" ,
        mapbox=dict(
            center=go.layout.mapbox.Center(
                lat=(current_viewport["latitude"]["min"] + current_viewport["latitude"]["max"]) / 2,
                lon=(current_viewport["longitude"]["min"] + current_viewport["longitude"]["max"]) / 2,
            ),
            zoom=default_zoom,
        ),
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        )
        return fig

    zoom = relayout_data.get("mapbox.zoom", default_zoom)
    base_dir = os.path.join(directory, f"{year}/{month}")

    if relayout_data and "mapbox.center" in relayout_data:
        relayout_data
        new_viewport = {
            "latitude": {
                "min": relayout_data["mapbox.center"]["lat"] - 0.1,
                "max": relayout_data["mapbox.center"]["lat"] + 0.1,
            },
            "longitude": {
                "min": relayout_data["mapbox.center"]["lon"] - 0.1,
                "max": relayout_data["mapbox.center"]["lon"] + 0.1,
            },
        }
    else:
        new_viewport = current_viewport
    return create_map_with_geotiff_tiles(tile_metadata, new_viewport, zoom, base_dir)
if __name__ == "__main__":
    app.run_server(debug=True)
