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
from instageo.apps.utils.countries import get_available_countries
from instageo.apps.viz import create_map_with_geotiff_tiles


# @st.cache_data
def generate_map(  # add a better legend
    directory: str, year: int, month: int, country_tiles: list[str]
) -> list[xr.Dataset]:
    """Generate the plotly map.

    Arguments:
        directory (str): Directory containing GeoTiff files.
        year (int): Selected year.
        month (int): Selected month formatted as an integer in the range 1-12.
        country_tiles (list[str]): List of MGRS tiles for the selected country.

    Returns:
        None.
    """
    try:
        if not directory or not Path(directory).is_dir():
            raise ValueError("Invalid directory path.")

        prediction_tiles = glob.glob(os.path.join(directory, f"{year}/{month}/*.tif"))
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
    """Instageo Serve Main Entry Point."""
    predictions_path = str(Path(__file__).parent / "predictions_new")
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
             padding-top: 1.7rem; /* Reduce top padding for more map space*/
             padding-bottom: 1rem; /* Reduce top padding for more map space*/
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    user_email = st.sidebar.text_input(
        "Enter your email for the report (optional):"
    )  # add list email
    send_report = st.sidebar.checkbox("Send me a risk report")
    st.sidebar.header("Settings")
    with open(
        INSTAGEO_APPS_PATH / "utils/country_name_to_mgrs_tiles.json"
    ) as json_file:
        countries_to_tiles_map = json.load(json_file)

    with st.sidebar.container():
        year = st.sidebar.number_input("Select Year", 2020, 2024)  # PLEASE USE 2021
        month = st.sidebar.number_input("Select Month", 1, 12)  # PLEASE USE 6

        available_countries = get_available_countries(predictions_path, year, month)

        if available_countries:
            country_codes = st.sidebar.multiselect(
                "ISO 3166-1 Alpha-2 Country Codes:",
                options=available_countries,
                default=available_countries[6],  # Defaults to first two available
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
        fig = generate_map(predictions_path, year, month, country_tiles)
        st_folium(fig, width="100%", height=600, returned_objects=[])
        if send_report and user_email:
            with st.spinner("Generating and sending report..."):
                map_image_path = generate_high_density_report(fig=fig)
                if send_email(
                    user_email,
                    "Desert Locust Risk Report",
                    img_path=map_image_path,
                ):
                    st.success("Report sent successfully!")
                    try:  # clean up map
                        os.remove(map_image_path)
                    except Exception as e:
                        print(e)
                else:
                    st.error("Error sending report.")
    else:  # this is to init an empty map
        fig = folium.Map((10, 35), zoom_start=3)
        st_folium(fig, width="100%", height=600, returned_objects=[])


if __name__ == "__main__":
    main()
