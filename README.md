<picture>
  <source srcset="assets/logo-dark.png" media="(prefers-color-scheme: dark)">
  <img src="assets/logo.png" alt="Logo">
</picture>

## Overview

InstaGeo is geospatial deep learning Python package designed to facilitate geospatial machine learning using satellite imagery data from [Harmonized Landsat and Sentinel-2 (HLS)](https://hls.gsfc.nasa.gov/) Data Product and [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) geospatial foundational model. It consists of three core components: Data, Model, and Apps, each tailored to support various aspects of geospatial data retrieval, manipulation, preprocessing, model training, and inference serving.

### Components

1. [**Data**](./instageo/data/README.md): Focuses on retrieving, manipulating, and processing Harmonized Landsat Sentinel-2 (HLS) data for classification and segmentation tasks such as disaster mapping, crop classification, and breeding ground prediction.
2. [**Model**](./instageo/model/README.md): Centers around data loading, training, and evaluating models, particularly leveraging the Prithvi model for various modeling tasks. It includes a sliding-window feature that allows inference to be run on large inputs.
3. [**Apps**](./instageo/apps/README.md): Aims to operationalize models developed in the Model component for practical applications.

## Installation

To get started with InstaGeo, ensure you have Python installed on your system. Then, execute the following command in your terminal or command prompt to install InstaGeo:

```bash
pip install instageo
```
This command will download and install the latest version of InstaGeo along with its required dependencies.

## Running Tests
After installation, you may want to verify that InstaGeo has been correctly installed and is functioning as expected. To do this, run the included test suite with the following commands:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
pytest --verbose .
```
## Usage

### Data Component

- **HLS Data Retrieval**: InstaGeo efficiently searches for and download Sentinel-2 and Landsat 8/9 multi-spectral earth observation images from the HLS data product.

- **Create Chips and Segmentation Maps**: InstaGeo breaks down large satellite image tiles into smaller, manageable patches (referred to as "chips") suitable for deep learning model training. It also generate segmentation maps, which serve as targets for training, by categorizing each pixel in the chips.

### Model Component

- **Training Custom Models**: Utilize the Prithvi geospatial foundational model as a backbone to develop custom models tailored for precise geospatial applications. These applications include, but are not limited to, flood mapping for emergency response planning, crop classification for agricultural management, and locust breeding ground prediction to address food security.

- **Inference on Large-scale Geospatial Data**: Perform inference using the models that have been trained on 'chips' (typically measuring 224 x 224 pixels) on expansive HLS tiles, which measure 3660 x 3660 pixels.

### Apps Component

- **Operationalize Models**: Once data has been created and model trained, deploy model for use using the Apps components. HLS tile predictions can be overlaid and visualized on interactive maps.

### Putting It All Together - Locust Breeding Ground Prediction
See [InstGeo_Demo](notebooks/InstaGeo_Demo.ipynb) notebook for an end-to-end demo.
#### Download locust breeding ground observation records.
```bash
mkdir locust_breeding
gsutil -m cp -r gs://instageo/data/locust_breeding/records locust_breeding
```

#### Download HLS tiles, create chips and segmentation maps

**Note: Ensure that you have up to **1.5TB** free disk space**

Create output directory for each split
```bash
mkdir locust_breeding/train locust_breeding/val locust_breeding/test
```

- Train Split
```bash
python -m "instageo.data.chip_creator" \
    --dataframe_path="locust_breeding/records/train.csv" \
    --output_directory="locust_breeding/train" \
    --min_count=1 \
    --chip_size=224 \
    --no_data_value=-1 \
    --temporal_tolerance=3 \
    --temporal_step=30 \
    --mask_cloud=False \
    --num_steps=3
```

- Validation Split
```bash
python -m "instageo.data.chip_creator" \
    --dataframe_path="locust_breeding/records/val.csv" \
    --output_directory="locust_breeding/val" \
    --min_count=1 \
    --chip_size=224 \
    --no_data_value=-1 \
    --temporal_tolerance=3 \
    --temporal_step=30 \
    --mask_cloud=False \
    --num_steps=3
```

- Test Split
```bash
python -m "instageo.data.chip_creator" \
    --dataframe_path="locust_breeding/records/test.csv" \
    --output_directory="locust_breeding/test" \
    --min_count=1 \
    --chip_size=224 \
    --no_data_value=-1 \
    --temporal_tolerance=3 \
    --temporal_step=30 \
    --mask_cloud=False \
    --num_steps=3
```

#### Launch Training

Before launching training, modify the path to chips and segmentation maps in each split
```python
for split in ["train", "val", "test"]:
    root_dir = "locust_breeding"
    chips = [
        chip.replace("chip", f"{split}/chips/chip")
        for chip in os.listdir(os.path.join(root_dir, f"{split}/chips"))
    ]
    seg_maps = [
        chip.replace("chip", f"{split}/seg_maps/seg_map") for chip in chips_orig
    ]

    df = pd.DataFrame({"Input": chips, "Label": seg_maps})
    df.to_csv(os.path.join(root_dir, f"{split}.csv"))
```

```bash
python -m instageo.model.run --config-name=locust \
    root_dir='locust_breeding' \
    train_filepath="locust_breeding/train.csv" \
    valid_filepath="locust_breeding/val.csv"
```

#### Run Evaluation
```bash
python -m instageo.model.run --config-name=locust \
    root_dir='locust_breeding' \
    test_filepath="locust_breeding/test.csv" \
    train.batch_size=16 \
    checkpoint_path='instageo-data/outputs/2024-03-01/09-16-30/instageo_epoch-10-val_iou-0.70.ckpt' \
    mode=eval
```

#### Run Inference on Africa Continent
- Download HLS tiles
```bash
python -m "instageo.data.chip_creator" \
    --dataframe_path="gs://instageo/utils/africa_prediction_template.csv" \
    --output_directory="inference/20223-01" \
    --min_count=1 \
    --no_data_value=-1 \
    --temporal_tolerance=3 \
    --temporal_step=30 \
    --num_steps=3 \
    --download_only
```

- Inference
```bash
python -m instageo.model.run --config-name=locust \
    root_dir='inference/20223-01' \
    test_filepath='hls_dataset.json' \
    train.batch_size=16 \
    checkpoint_path='instageo-data/outputs/2024-03-01/09-16-30/instageo_epoch-10-val_iou-0.70.ckpt' \
    mode=predict
```

#### Visualize Predictions
- Run InstaGeo Serve
```bash
cd instageo/apps
streamlit run app.py
```
- Specify the directory containing the predictions.

![InstaGeo Serve](assets/instageo_serve.png)


## Contributing

We welcome contributions to InstaGeo. Please follow the [contribution guidelines](./CONTRIBUTING.md) for submitting pull requests and reporting issues to help us improve the package.

## License

This project is licensed under the [CC BY-NC-SA 4.0](LICENSE).
