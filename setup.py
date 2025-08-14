"""InstaGeo Package Setup."""

from setuptools import find_packages, setup

data_dependencies = [
    # Add dependencies specific to the data component
    "geopandas==0.14.4",
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
    "pystac==1.11.0",
    "pystac_client==0.8.5",
    "stackstac==0.5.1",
    "planetary_computer==1.0.0",
    "pyarrow==18.1.0",
    "haversine==2.8.1",
    "cartopy==0.24.1",
    "seaborn==0.13.2",
    "scikit-learn==1.6.0",
    "backoff==2.2.1",
    "ratelimit==2.2.1",
    "astral==3.2",
]
model_dependencies = [
    # Add dependencies specific to the model component
    "pytorch_lightning",
    "torch",
    "timm==1.0.15",
    "einops",
    "tensorboard",
    "hydra-core",
    "omegaconf",
    "huggingface_hub",
]
apps_dependencies = [
    # Add dependencies specific to the apps component
    "plotly",
    "dask==2024.12.1",
    "distributed==2024.12.1",
    "datashader",
    "matplotlib",
    "streamlit==1.31.1",
    # new apps components dependencies
    "fastapi==0.115.12",
    "uvicorn==0.34.3",
    "python-dotenv==1.0.1",
    "pydantic==2.10.4",
    "redis==6.2.0",
    "rq==2.4.0",
    "rq-dashboard==0.8.3.2",
    "types-redis>=4.6.0",
    "titiler.core==0.22.4",
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
        "scikit-learn",
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
