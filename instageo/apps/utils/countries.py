from instageo import INSTAGEO_APPS_PATH
import os
import glob
import json
import streamlit as st


def get_available_countries(predictions_path: str) -> list[str]:
    """Returns a list of available countries based on the GeoTIFF files present for the specified year and month."""
    available_countries = set()
    with open(
        INSTAGEO_APPS_PATH / "utils/country_name_to_mgrs_tiles.json"
    ) as json_file:
        countries_to_tiles_map = json.load(json_file)

    try:
        if not os.path.exists(predictions_path):
            return list(available_countries)

        available_tiles = set(
            os.path.basename(filename).split("_")[3][1:]
            for filename in glob.glob(os.path.join(predictions_path, "*.tif"))
        )

        for country, tiles in countries_to_tiles_map.items():
            if any(tile in available_tiles for tile in tiles):
                available_countries.add(country)

    except Exception as e:
        st.error(f"An error occurred while checking available countries: {e}")

    return list(available_countries)


def get_latest_forecast_year_month(predictions_path: str) -> tuple[str, str] | None:
    """Finds the latest forecast year and month in the "Latest" directory."""

    latest_forecast_path = os.path.join(predictions_path, "Latest")
    if not os.path.exists(latest_forecast_path):
        st.error("Latest forecast directory not found!")
        return None

    latest_year = None
    latest_month = None

    for year_dir in os.listdir(latest_forecast_path):
        if year_dir.isdigit():
            year = int(year_dir)
            year_path = os.path.join(latest_forecast_path, year_dir)
            for month_dir in os.listdir(year_path):
                if month_dir.isdigit():
                    month = int(month_dir)
                    if (
                        latest_year is None
                        or year > latest_year
                        or (year == latest_year and month > latest_month)
                    ):
                        latest_year = year
                        latest_month = month
    if latest_year is not None:
        return str(latest_year), str(latest_month)
    else:
        st.warning("No valid forecast directories found in Latest")
        return None
