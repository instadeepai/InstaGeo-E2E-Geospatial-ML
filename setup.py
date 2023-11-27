from setuptools import find_packages, setup

setup(
    name="instageo",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        # Add common dependencies here
    ],
    extras_require={
        "data": [
            # Add dependencies specific to the data component
        ],
        "model": [
            # Add dependencies specific to the model component
        ],
        "apps": [
            # Add dependencies specific to the apps component
        ],
    },
    python_requires=">=3.10",
    tests_require=["pytest"],
    setup_requires=["pytest-runner"],
    test_suite="tests",
    author="Ibrahim Salihu Yusuf",
    author_email="i.yusuf@instadeep.com",
    description="""A modular Python package for geospatial data processing, modeling,
                    and applications using Harmonized Landsat Sentinel (HLS) Data""",
    keywords="geospatial, machine learning, data, model, applications",
    url="https://github.com/instadeepai/InstaGeo",
)
