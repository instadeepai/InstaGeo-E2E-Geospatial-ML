import streamlit as st

def general_styles() -> None:
    """General Styles."""

    st.markdown(
        """
    <style>
        body {
            background-color: white !important;
        }
        body * {
            color: #404F65 !important;
        }
        .main {
            overflow: hidden;
        }
        div[data-testid="stMainBlockContainer"]  {
            padding: 0px !important;
            background-color: white !important;

        }
        div[data-testid="stMain"]  {
            background-color: white !important;
        }
        div[data-testid="stMainBlockContainer"] .stPlotlyChart {
            height: 100vh;
        }
        .js-plotly-plot {
            height:100%;
            z-index: 99;
        }
         div[data-testid="stMainBlockContainer"] .stPlotlyChart .plot-container {
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
        section[data-testid="stSidebar"] {
            background-color: white !important;
           border-right: 1px solid #E2E8F0 !important;
        }
        div[data-testid="stHorizontalBlock"] {
            margin-bottom: 200px !important;
            flex-grow: 0;

        }
        div[data-testid="stSidebarContent"] {
            display: flex;
            flex-direction: column;
            gap: 20px;

        }
        div[data-testid="stSidebarUserContent"] {
            flex-grow: 1;
            padding-bottom: 40px !important;

        }
        div[data-testid="stSidebarUserContent"]>div {
            height: 100%;

        }
        div[data-testid="stSidebarUserContent"]>div>div {
            height: 100%;

        }
        div[data-testid="stSidebarUserContent"]>div>div>div {
            height: 100%;

        }
        div[data-testid="stSidebarUserContent"]>div>div>div>div {
            height: 100%;

        }
        div[data-testid="stVerticalBlock"]> div[data-testid="stElementContainer"]:last-of-type {
            margin-top: auto !important;

        }
        section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:last-of-type button{
            background: #3892FF !important;
            border-radius: 8px !important;
            border: none !important;
            & p {
                color: white !important;
                font-size: 14px !important;
            }

        }
         
        div[data-baseweb="select"] {
            border: 1px solid #eeeeee !important;
            border-radius: 6px; !important
        }
        div[data-baseweb="select"]:focus-visible * {
            border-color: #265BF2 !important;
        }
         label *{
            color: #8694A9 !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )