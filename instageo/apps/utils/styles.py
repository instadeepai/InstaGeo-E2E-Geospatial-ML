import streamlit as st

def general_styles() -> None:
    """General Styles."""

    st.markdown(
        """
    <style>
        .main {
            overflow: hidden;
        }
        div[data-testid="stAppViewBlockContainer"]  {
            padding: 0px !important;
        }
        div[data-testid="stAppViewBlockContainer"] .stPlotlyChart {
            height: 100vh;
        }
         div[data-testid="stAppViewBlockContainer"] .stPlotlyChart .plot-container {
            height: 100%;
        }
        div[data-testid="stVerticalBlock"] {
            gap:0;
        }
        div[data-testid="stFullScreenFrame"] {
            width:100%;
            height:100%;
        }
        .modebar{
            display: none !important;
        }
        .mapboxgl-map {
            height: 100vh !important;
            top: 0px !important;
        }
        header {
            display: none !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )