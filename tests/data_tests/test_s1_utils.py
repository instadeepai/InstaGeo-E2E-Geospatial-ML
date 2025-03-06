import pandas as pd
import pytest
from pystac.item import Item

from instageo.data.data_pipeline import get_pystac_client, get_tile_info
from instageo.data.s1_utils import (
    add_s1_items,
    dispatch_candidate_items,
    find_closest_items,
    is_valid_dataset_entry,
    retrieve_s1_metadata,
)


@pytest.fixture
def pystac_client():
    return get_pystac_client()


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "x": [-15.303611111],
            "y": [21.6697222220001],
            "mgrs_tile_id": ["28QDJ"],
            "input_features_date": [pd.to_datetime("2016-03-19")],
            "label": [1],
        }
    )


@pytest.fixture
def data_2():
    return pd.DataFrame(
        {
            "x": [28.80148067, 28.87],
            "y": [-3.22772705, -3.065],
            "mgrs_tile_id": ["35MQS", "35MQS"],
            "input_features_date": [
                pd.to_datetime("2017-12-31"),
                pd.to_datetime("2017-12-31"),
            ],
            "label": [1, 0],
        }
    )


@pytest.fixture
def data_3():
    return pd.DataFrame(
        {
            "x": [28.80148067, 28.87],
            "y": [-3.22772705, -3.065],
            "mgrs_tile_id": ["35MQS", "35MQS"],
            "input_features_date": [
                pd.to_datetime("2017-12-31"),
                pd.to_datetime("2018-01-02"),
            ],
            "label": [1, 0],
        }
    )


@pytest.fixture
def tile_info_df():
    return pd.DataFrame(
        {
            "tile_id": ["28QDJ"],
            "start_date": ["2015-11-20"],
            "end_date": ["2016-03-19"],
            "lon_min": [-15.303611111],
            "lon_max": [-15.303611111],
            "lat_min": [21.6697222220001],
            "lat_max": [21.6697222220001],
        }
    )


def test_retrieve_s1_metadata(pystac_client, tile_info_df):
    """Tests retrieval of potential S1 PySTAC items for a given observation."""

    metadata = retrieve_s1_metadata(pystac_client, tile_info_df)
    item_ids = [item.id for item in metadata["28QDJ"]]
    assert item_ids == [
        "S1A_IW_GRDH_1SDV_20151227T191124_20151227T191149_009232_00D4FD_rtc",
        "S1A_IW_GRDH_1SDV_20160103T190305_20160103T190330_009334_00D7E3_rtc",
        "S1A_IW_GRDH_1SDV_20160108T191123_20160108T191148_009407_00D9FB_rtc",
        "S1A_IW_GRDH_1SDV_20160115T190305_20160115T190330_009509_00DCE3_rtc",
        "S1A_IW_GRDH_1SDV_20160120T191123_20160120T191148_009582_00DF05_rtc",
        "S1A_IW_GRDH_1SDV_20160127T190304_20160127T190329_009684_00E204_rtc",
        "S1A_IW_GRDH_1SDV_20160201T191123_20160201T191148_009757_00E423_rtc",
        "S1A_IW_GRDH_1SDV_20160208T190304_20160208T190329_009859_00E706_rtc",
        "S1A_IW_GRDH_1SDV_20160213T191122_20160213T191147_009932_00E936_rtc",
        "S1A_IW_GRDH_1SDV_20160220T190304_20160220T190329_010034_00EC3B_rtc",
        "S1A_IW_GRDH_1SDV_20160225T191122_20160225T191147_010107_00EE58_rtc",
        "S1A_IW_GRDH_1SDV_20160303T190304_20160303T190329_010209_00F126_rtc",
        "S1A_IW_GRDH_1SDV_20160308T191122_20160308T191147_010282_00F347_rtc",
    ]


def test_find_best_s1_items(pystac_client, data):
    """Tests retrieval of best PySTAC items for a given observation."""
    src_crs = 4326
    best_items = add_s1_items(pystac_client, data, src_crs)
    assert isinstance(best_items, dict)
    assert isinstance(best_items[data["mgrs_tile_id"].values[0]], pd.DataFrame)
    best_item_ids = [
        item.id
        for item in best_items[data["mgrs_tile_id"].values[0]].s1_items.values[0]
    ]
    assert best_item_ids == [
        "S1A_IW_GRDH_1SDV_20160320T191123_20160320T191148_010457_00F83E_rtc",
        "S1A_IW_GRDH_1SDV_20160308T191122_20160308T191147_010282_00F347_rtc",
        "S1A_IW_GRDH_1SDV_20160225T191122_20160225T191147_010107_00EE58_rtc",
    ]


def test_is_valid_dataset_entry(pystac_client, data):
    """Tests validity of an entry in the S1 dataset."""

    src_crs = 4326
    best_items = add_s1_items(pystac_client, data, src_crs)
    entry = best_items[data["mgrs_tile_id"].values[0]]

    entry["s1_granules"] = [[item.id for item in entry.s1_items.values[0]]]
    is_valid = is_valid_dataset_entry(entry.iloc[0])
    assert is_valid

    # Let's test a case where we should not find an item for every timestep.
    best_items = add_s1_items(pystac_client, data, src_crs, temporal_tolerance=1)
    invalid_entry = best_items[data["mgrs_tile_id"].values[0]]
    invalid_entry["s1_granules"] = [
        [
            item.id if isinstance(item, Item) else None
            for item in invalid_entry.s1_items.values[0]
        ]
    ]
    is_valid = is_valid_dataset_entry(invalid_entry.iloc[0])

    assert not is_valid


def test_dispatch_candidate_items(pystac_client, data_2):
    """Tests dispatching of appropriate PySTAC items to observation points."""

    tiles_info, tile_queries = get_tile_info(
        data_2,
        num_steps=1,
    )
    data_2["tile_queries"] = tile_queries
    tiles_database = retrieve_s1_metadata(pystac_client, tiles_info)["35MQS"]

    # We confirm that for our dates and points of observations (for our MGRS
    # grid) we have two possible S1 tiles that overlap the bounding box
    # defined by the two points.
    assert [item.id for item in tiles_database] == [
        "S1A_IW_GRDH_1SDV_20171226T034508_20171226T034533_019868_021CE6_rtc",
        "S1A_IW_GRDH_1SDV_20171226T034533_20171226T034558_019868_021CE6_rtc",
    ]

    # Without dispatching, when applying the function to extract the temporally
    # closest tile, the first tile will be attributed to the first point
    # because both tiles have the same date in the toy dataset and the first closest
    # item is kept. But in reality the first point is not inside the first tile.
    # The chip that would be created would then result in no data values for the
    # observation.

    # We make sure each point is assigned the right tile by filtering based
    # on the geometry of the tile. Essentially, for the data to be extracted
    # the point needs to be inside the tile. We thus make sure that each of
    # the two points in our toy dataset is attributed the corresponding tile.
    data_with_items = dispatch_candidate_items(data_2, 4326, tiles_database)[
        "s1_candidate_items"
    ]
    assert [item.id for item in data_with_items.iloc[0]] == [
        "S1A_IW_GRDH_1SDV_20171226T034533_20171226T034558_019868_021CE6_rtc"
    ]
    assert [item.id for item in data_with_items.iloc[1]] == [
        "S1A_IW_GRDH_1SDV_20171226T034508_20171226T034533_019868_021CE6_rtc"
    ]


def test_find_closest_items(pystac_client, data_3):
    """Tests retrieval of temporally closest PySTAC items to observation dates."""

    tiles_info, tile_queries = get_tile_info(data_3, num_steps=1, temporal_tolerance=6)
    data_3["tile_queries"] = tile_queries
    tiles_database = retrieve_s1_metadata(pystac_client, tiles_info)["35MQS"]

    assert [item.id for item in tiles_database] == [
        "S1A_IW_GRDH_1SDV_20171226T034508_20171226T034533_019868_021CE6_rtc",
        "S1A_IW_GRDH_1SDV_20171226T034533_20171226T034558_019868_021CE6_rtc",
        "S1A_IW_GRDH_1SDV_20180107T034508_20180107T034533_020043_022268_rtc",
        "S1A_IW_GRDH_1SDV_20180107T034533_20180107T034558_020043_022268_rtc",
    ]

    # For sake of testing the `find_closest_items` func let's simply
    # assign the original list of the retrieved items as candidate items

    data_3["s1_candidate_items"] = [tiles_database] * data_3.shape[0]
    data_3["closest_s1_items"] = data_3.apply(
        lambda obsv: find_closest_items(obsv, 6), axis=1
    )

    # Date: "2017-12-31"
    assert (
        data_3["closest_s1_items"].iloc[0][0].id
        == "S1A_IW_GRDH_1SDV_20171226T034533_20171226T034558_019868_021CE6_rtc"
    )

    # Date: "2018-01-02"
    assert (
        data_3["closest_s1_items"].iloc[1][0].id
        == "S1A_IW_GRDH_1SDV_20180107T034508_20180107T034533_020043_022268_rtc"
    )
