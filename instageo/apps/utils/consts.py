
MAP_STYLE="https://tiles.stadiamaps.com/styles/alidade_smooth.json"

MODELS_LIST=[
    "Locust Breeding Ground",
    "Biomass Estimation",
    "Aerosol Optical Depth Estimation"
]


PREDICTIONS=[
    {"name": "Prediction 1", "path": "output_rgba3_opt.tif"},
    {"name": "Prediction 2", "path": "prediction_2.json"},
]

TILE_SERVER_URL="http://localhost:8000/cog/tiles/WebMercatorQuad/{z}/{x}/{y}?url=/data/"






