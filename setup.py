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
    "earthaccess==0.8.2",
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
