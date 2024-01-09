# InstaGeo

## Overview

InstaGeo is geospatial deep learning Python package designed to facilitate geospatial tasks using satellite imagery data from [Harmonized Landsat and Sentinel-2 (HLS)](https://hls.gsfc.nasa.gov/) Data Product as well as [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M). It consists of three core components: Data, Model, and Apps, each tailored to support various aspects of geospatial data manipulation, model training, and application deployment.

### Components

1. [**Data**](./instageo/data/README.md): Focuses on reading, manipulating, and processing Harmonized Landsat Sentinel-2 (HLS) data for tasks like disaster mapping, crop mapping, and breeding ground prediction.
2. [**Model**](./instageo/model/README.md): Centers around data loading, training, and evaluating models, particularly leveraging the Prithvi model for various tasks. It includes pre-training custom foundational models.
3. [**Apps**](./instageo/apps/README.md): Aims to operationalize models developed in the Model component for practical applications, such as locust infestation dashboards.

## Installation

To install InstaGeo, run the following command:

```bash
pip install instageo
```
## Installation

To run InstaGeo tests, run the following command:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
pytest --verbose .
```
## Usage

### Data Component

- **Mirror HLS Data**: Access up-to-date HLS data mirrored on GCP for your projects.
- **Create Chips and Segmentation Maps**: Use the provided tools to create chips and segmentation maps from HLS data.

### Model Component

- **Train Models**: Leverage the Prithvi model for training on your specific geospatial task.
- **Evaluate Performance**: Use the evaluation tools to assess the performance of your models.

### Apps Component

- **Operationalize Models**: Deploy models into applications like the Locust Dashboard to make real-world impacts.

## Contributing

We welcome contributions to InstaGeo. Please follow the [contribution guidelines](./CONTRIBUTING.md) for submitting pull requests and reporting issues to help us improve the package.

<!-- ## License -->
