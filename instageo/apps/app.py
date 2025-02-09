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
from instageo.apps.viz import create_map_with_geotiff_tiles

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
