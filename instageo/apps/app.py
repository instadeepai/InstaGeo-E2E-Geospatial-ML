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
# Edited 5.2.2025 by LocustBusters (Lorenzo FURLAN, Alexis VIOLEAU, Victor XING, Kais CHEIKH)

"""InstaGeo Serve Module.

InstaGeo Serve is a web application that enables the visualisation of GeoTIFF files in an
interactive map.
"""

import xarray as xr
from streamlit_folium import st_folium

import glob
import json
import os
from pathlib import Path
from instageo.apps.reporting import (
    generate_high_density_report,
    # format_report,
    send_email,
)
import streamlit as st
import folium

from instageo import INSTAGEO_APPS_PATH
from instageo.apps.utils.countries import (
    get_available_countries,
    get_latest_forecast_year_month,
)
from instageo.apps.viz import create_map_with_geotiff_tiles


def generate_map(
    full_predictions_path: str, country_tiles: list[str]
) -> list[xr.Dataset]:
    """Generate the plotly map.

    Arguments:
        full_predictions_path (str): Directory containing GeoTiff files.
        country_tiles (list[str]): List of MGRS tiles for the selected country.

    Returns:
        None.
    """
    try:
        if (
            not full_predictions_path or not Path(full_predictions_path).is_dir()
        ):  # redudant
            raise ValueError("Invalid directory path.")

        prediction_tiles = glob.glob(os.path.join(full_predictions_path, "*.tif"))
        tiles_to_consider = [
            tile
            for tile in prediction_tiles
            if os.path.basename(tile).split("_")[3][1:] in country_tiles
        ]

        if not tiles_to_consider:
            raise FileNotFoundError(
                "No GeoTIFF files found for the given year, month, and country."
            )

        fig = create_map_with_geotiff_tiles(tiles_to_consider)
        return fig

    except (ValueError, FileNotFoundError, Exception) as e:
        st.error(f"An error occurred: {str(e)}")
        return folium.Map((0, 0))


def main() -> None:
    """Instageo Serve Main Entry Point, edited by LocustBusters"""
    predictions_path = str(Path(__file__).parent / "predictions_new")
    with open(
        INSTAGEO_APPS_PATH / "utils/country_name_to_mgrs_tiles.json"
    ) as json_file:
        countries_to_tiles_map = json.load(json_file)
    st.set_page_config(
        page_title="Locust busters", page_icon=":cricket:", layout="wide"
    )
    st.title("Locust busters :cricket:")

    st.sidebar.subheader(
        "This application enables the visualisation of GeoTIFF files on an interactive map. You can also receive an alert report",
        divider="rainbow",
    )
    st.markdown(
        """
        <style>
        .st-folium {  /* Target the streamlit-folium container */
            width: 100%;
            margin: 0 auto; /* Center the map horizontally */
        }
        .block-container {  /* Streamlit's main container */
             padding-top: 1.6rem; /* Reduce top padding for more map space*/
             padding-bottom: 1rem; /* Reduce top padding for more map space*/
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    user_emails = st.sidebar.text_area(
        "Enter recipient emails for the report (comma-separated):"
    )
    send_report = st.sidebar.checkbox("Send me a risk report")
    st.sidebar.header("Settings")
    forecast_option = st.sidebar.radio("Select Forecast:", ("Latest", "Specific Date"))

    with st.sidebar.container():
        if forecast_option == "Latest":
            year, month = get_latest_forecast_year_month(predictions_path)
            full_predictions_path = os.path.join(
                predictions_path, "Latest", year, month
            )
        elif forecast_option == "Specific Date":
            year = st.sidebar.number_input("Select Year", 2020, 2024)
            month = st.sidebar.number_input("Select Month", 1, 12)
            full_predictions_path = os.path.join(
                predictions_path, str(year), str(month)
            )
        available_countries = get_available_countries(full_predictions_path)

        if available_countries:
            country_codes = st.sidebar.multiselect(
                "Countries available for inspection:",
                options=available_countries,
            )
        else:
            st.sidebar.warning("No data available for the selected year and month.")
            country_codes = []

    if st.sidebar.button("Generate Map"):
        country_tiles = [
            tile
            for country_code in country_codes
            for tile in countries_to_tiles_map.get(country_code, [])
        ]
        fig, all_clusters = generate_map(full_predictions_path, country_tiles)
        st_folium(fig, width="100%", height=550, returned_objects=[])
        if send_report and user_emails:
            email_list = [
                email.strip() for email in user_emails.split(",") if email.strip()
            ]
            if not email_list:
                st.error("Please enter valid email addresses.")
            else:
                with st.spinner("Generating and sending report..."):
                    map_image_path, report_text = generate_high_density_report(
                        fig=fig,
                        all_clusters=all_clusters,
                        year=year,
                        month=month,
                    )
                    success_count = 0
                    for email in email_list:
                        if send_email(
                            email,
                            "Desert Locust Risk Report",
                            map_image_path,
                            report_text,
                            year,
                            month,
                        ):
                            success_count += 1
                        else:
                            st.error(f"Error sending email to {email}")

                    try:
                        os.remove(map_image_path)
                    except Exception:
                        pass

                    if success_count == len(email_list):
                        st.success("Report sent successfully to all recipients!")
                    elif success_count > 0:
                        st.warning(
                            f"Report sent successfully to {success_count} out of {len(email_list)} recipients."
                        )
    else:
        fig = folium.Map((10, 35), zoom_start=3)
        st_folium(fig, width="100%", height=550, returned_objects=[])


if __name__ == "__main__":
    main()
