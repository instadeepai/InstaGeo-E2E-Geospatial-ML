"""InstaGeo Package Setup."""

from setuptools import find_packages, setup

data_dependencies = [
    # Add dependencies specific to the data component
    "geopandas==0.14.1",
    "shapely",
    "cftime",
    "h5pyd",
    "Bottleneck",
    "absl-py",
    "mgrs==1.4.6",
    "earthaccess==0.12.0",
    "pydantic==2.10.4",
    "pydantic-settings==2.7.0",
    "python-dotenv==1.0.1",
    "pystac_client==0.8.5",
    "stackstac==0.5.1",
    "planetary_computer==1.0.0",
]
model_dependencies = [
    # Add dependencies specific to the model component
    "pytorch_lightning",
    "torch",
    "timm==0.4.12",
    "einops",
    "tensorboard",
    "hydra-core",
    "omegaconf",
]
apps_dependencies = [
    # Add dependencies specific to the apps component
    "plotly",
    "dask==2024.12.1",
    "datashader",
    "matplotlib",
    "streamlit==1.31.1",
]
setup(
    name="instageo",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        # Add common dependencies here
        "pandas",
        "numpy",
        "xarray[complete]",
        "rasterio",
        "rioxarray",
    ],
    extras_require={
        "data": data_dependencies,
        "model": model_dependencies,
        "apps": apps_dependencies,
        "all": data_dependencies + model_dependencies + apps_dependencies,
    },
    python_requires=">=3.10",
    tests_require=["pytest", "pytest-cov"],
    setup_requires=["pytest-runner"],
    test_suite="tests",
    author="Ibrahim Salihu Yusuf",
    author_email="i.yusuf@instadeep.com",
    description="""A modular Python package for geospatial data processing, modeling,
                    and applications using Harmonized Landsat Sentinel (HLS) Data""",
    keywords="geospatial, machine learning, data, model, applications",
    url="https://github.com/instadeepai/InstaGeo",
)
