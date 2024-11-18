import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point

from instageo.data.geo_utils import get_chip_coords, get_tile_info, get_tiles


@pytest.fixture
def observation_data():
    data = pd.DataFrame(
        {
            "date": {
                0: "2022-06-08",
                1: "2022-06-08",
                2: "2022-06-08",
                3: "2022-06-08",
                4: "2022-06-09",
                5: "2022-06-09",
                6: "2022-06-09",
                7: "2022-06-08",
                8: "2022-06-09",
                9: "2022-06-09",
            },
            "x": {
                0: 44.48,
                1: 44.48865,
                2: 46.437787,
                3: 49.095545,
                4: -0.1305,
                5: 44.6216,
                6: 49.398908,
                7: 44.451435,
                8: 49.435228,
                9: 44.744167,
            },
            "y": {
                0: 15.115617,
                1: 15.099767,
                2: 14.714659,
                3: 16.066929,
                4: 28.028967,
                5: 16.16195,
                6: 16.139727,
                7: 15.209633,
                8: 16.151837,
                9: 15.287778,
            },
            "year": {
                0: 2022,
                1: 2022,
                2: 2022,
                3: 2022,
                4: 2022,
                5: 2022,
                6: 2022,
                7: 2022,
                8: 2022,
                9: 2022,
            },
        }
    )
    data["date"] = pd.to_datetime(data["date"])
    data["input_features_date"] = data["date"]
    return data


def test_get_tiles(observation_data):
    hls_tiles = get_tiles(data=observation_data, min_count=1)
    assert list(hls_tiles["mgrs_tile_id"]) == [
        "38PMB",
        "38PMB",
        "38PPB",
        "39QTT",
        "30RYS",
        "38QMC",
        "39QUT",
        "38PMB",
        "39QUT",
        "38PMB",
    ]


def test_get_chip_coords():
    df = pd.read_csv("tests/data/sample_4326.csv")
    df = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.x, df.y)])
    df.set_crs(epsg=4326, inplace=True)
    df = df.to_crs(crs=32613)

    ds = xr.open_dataset("tests/data/HLS.S30.T38PMB.2022145T072619.v2.0.B02.tif")
    chip_coords = get_chip_coords(df, ds, 64)
    assert chip_coords == [
        (2, 0),
        (0, 3),
        (2, 2),
        (0, 3),
        (2, 0),
        (3, 2),
        (2, 3),
        (0, 3),
        (2, 3),
        (1, 2),
    ]


def test_get_tile_info(observation_data):
    hls_tiles = get_tiles(observation_data, min_count=3)
    tiles_info, tile_queries = get_tile_info(hls_tiles, num_steps=3, temporal_step=5)
    pd.testing.assert_frame_equal(
        tiles_info,
        pd.DataFrame(
            {
                "tile_id": ["38PMB"],
                "min_date": ["2022-05-29"],
                "max_date": ["2022-06-09"],
                "lon_min": [44.451435],
                "lon_max": [44.744167],
                "lat_min": [15.099767],
                "lat_max": [15.287778],
            }
        ),
        check_like=True,
    )
    assert tile_queries == [
        ("38PMB", ["2022-06-08", "2022-06-03", "2022-05-29"]),
        ("38PMB", ["2022-06-08", "2022-06-03", "2022-05-29"]),
        ("38PMB", ["2022-06-08", "2022-06-03", "2022-05-29"]),
        ("38PMB", ["2022-06-09", "2022-06-04", "2022-05-30"]),
    ]
