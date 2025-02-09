import os
import json
import dash
import plotly.graph_objects as go
import math
from dash import dcc, html,clientside_callback, ClientsideFunction
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
APP_DIR = Path(__file__).resolve().parent

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

MODELS_LIST=[
    "Locust Breeding Ground",
    "Biomass Estimation",
    "Aerosol Optical Depth Estimation"
]

def load_tile_metadata(json_path: str) -> list:
    """Load tile metadata from a JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)

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
    print("zoom ciel",zoom_ceiled)
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
        if len(mapbox_layers) > 15:
            break
        if is_tile_in_viewport(tile['bounds'], viewport, zoom=zoom):
            tile_path = os.path.join(base_dir, tile['name'])
            xarr_dataset, crs = read_geotiff_to_xarray(tile_path)
            scale = zoom_to_scale(zoom)
            print("--zoom--", zoom)
            print("----sclale",scale)
            img, coordinates = add_raster_to_plotly_figure(xarr_dataset, crs, scale=scale)
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
    html.Div(id='container-dimensions-store', style={'display': 'none'}),
    

    html.Div([
        html.Div([
            html.Img(src='assets/logo.png', style={'height': '50px'}),
        ],style={'display':'flex', "justifyContent":'center',"alignItems":'center','marginBottom':'20px'}),

        html.Div([

            html.Label("Model Name:"),
            dcc.Dropdown(id="model_name", options=[{"label": str(model), "value": model} for model in MODELS_LIST], 
                        value=6, style={"marginBottom": "10px"}),

        ],style={'display':'flex', "flexDirection":'column',"flex":"1","gap":"6px"}),

       
        html.Div([

            html.Div([

                html.Label("Year:"),
                dcc.Dropdown(id="year", options=[{"label": str(y), "value": y} for y in range(2023, 2025)], 
                        value=2023, style={"width": "100%","marginBottom": "10px", "borderRadius": "5px"}),

            ],style={'display':'flex', "flexDirection":'column',"flex":"1","gap":"6px"}),

            html.Div([
                html.Label("Month:"),
                dcc.Dropdown(id="month", options=[{"label": str(m), "value": m} for m in range(1, 13)], 
                        value=6, style={"marginBottom": "10px"}),

            ],style={'display':'flex', "flexDirection":'column',"flex":"1","gap":"6px"}),
            
            


        ],style={'display':'flex', "gap":"20px"}),
        
        html.Div([

            html.Label("Directory Path:"),
            dcc.Input(id="directory", type="text", placeholder="Enter directory path", 
                    style={"width": "100%", "marginBottom": "10px", "padding": "5px", "borderRadius": "5px",'height':'40px'}),

        ],style={'display':'flex', "flexDirection":'column',"flex":"1","gap":"6px"}),
        
        
            
        html.Button('Submit', id='submit-val', n_clicks=0,style={'width':'100%','height':'40px','backgroundColor':'#3892FF','color':'white','borderRadius':'5px','border':'none','cursor':'pointer','marginTop':'auto'}),
    
    ], style={'padding':'20px','backgroundColor':'white','height':'100vh',"width":"20%"}),
    



    html.Div([
        dcc.Graph(id="map", config={"scrollZoom": True,'displayModeBar':False}, style={"height": "100vh"}),
    ], style={'flex':'1',"height": "100vh"},id="plot-container"),

],style={'display':'flex','height':'100vh'})

tile_metadata_path = "instageo/apps/tile_metadata.json"
tile_metadata = load_tile_metadata(tile_metadata_path) 



# Clientside callback to get dimensions
clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='getDimensions'
    ),
    Output('container-dimensions-store', 'children'),
    Input('container-dimensions-store', 'id')
)

# Callback to update the map
@app.callback(
    Output("map", "figure"),
    Input("map", "relayoutData"),
    Input("directory", "value"),
    Input("year", "value"),
    Input("month", "value"),
    State("stored-viewport", "data"),
    Input('container-dimensions-store', 'children')

)
def update_map(relayout_data, directory, year, month, current_viewport,dimensions):
    """Update the map based on viewport, zoom, and directory selection."""
    print(dimensions)
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





# Create assets folder and JavaScript file
assets_dir = APP_DIR / 'assets'
if not assets_dir.exists():
    assets_dir.mkdir()

js_file = assets_dir / 'clientside.js'
if not js_file.exists():
    js_content = """
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        getDimensions: function() {
            const container = document.getElementById('plot-container');
            if (container) {
                const rect = container.getBoundingClientRect();
                return JSON.stringify({
                    width: rect.width,
                    height: window.innerHeight,
                    containerWidth: rect.width
                });
            }
            return JSON.stringify({
                width: window.innerWidth,
                height: window.innerHeight,
                containerWidth: window.innerWidth
            });
        }
    }
});
"""
    js_file.write_text(js_content)

if __name__ == "__main__":
    app.run_server(debug=True)
