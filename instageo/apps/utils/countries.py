from instageo import INSTAGEO_APPS_PATH
import os
import glob
import json
import streamlit as st


def get_available_countries(predictions_path: str, year: int, month: int) -> list[str]:
    """Returns a list of available countries based on the GeoTIFF files present for the specified year and month."""
    available_countries = set()
    with open(
        INSTAGEO_APPS_PATH / "utils/country_name_to_mgrs_tiles.json"
    ) as json_file:
        countries_to_tiles_map = json.load(json_file)

    try:
        month_path = os.path.join(predictions_path, str(year), str(month))
        if not os.path.exists(month_path):  # Check if the path exists
            return list(available_countries)

        available_tiles = set(
            os.path.basename(filename).split("_")[3][1:]
            for filename in glob.glob(os.path.join(month_path, "*.tif"))
        )

        for country, tiles in countries_to_tiles_map.items():
            if any(tile in available_tiles for tile in tiles):
                available_countries.add(country)

    except Exception as e:
        st.error(f"An error occurred while checking available countries: {e}")

    return list(available_countries)
