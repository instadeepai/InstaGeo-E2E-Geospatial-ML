# InstaGeo - Model

## Overview

The InstaGeo Model component is designed for training, validation and inference using custom deep learning models having [Prithvi_100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) foundational model as backbone.

## Requirements

Install dependencies using uv (recommended):

CPU

```bash
uv sync --locked --extra model --extra cpu
```

GPU

```bash
uv sync --locked --extra model --extra gpu
```

## Features
- Supports both classification and regression tasks
- Accepts both temporal and non-temporal inputs
- Custom models with [Prithvi_100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) backbone
- Training, Validation and Inference runs
- Sliding window inference for inference on expansive tiles, which measure 3660 x 3660 pixels
- Reproducible training pipeline
- Command-line flags for easy configuration of training parameters.

## Usage

1. **Setting Up Run Arguments:** Configure the training parameters using command-line arguments.

    - `root_dir`: Root directory of the dataset.
    - `valid_filepath`: File path for validation data.
    - `train_filepath`: File path for training data.
    - `test_filepath`: File path for test data.
    - `checkpoint_path`: File path for model checkpoint.
    - `learning_rate`: Initial learning rate.
    - `num_epochs`: Number of training epochs.
    - `batch_size`: Batch size for training and validation.
    - `mode`: Select stats, train, eval, sliding_inference or chip_inference mode.

See `configs/config.yaml` for more.

2. **Dataset Preparation:** Prepare your geospatial data using the InstaGeo Chip Creator or similar and place it in the specified `root_dir`. Ensure that the csv file for each dataset has `Input` and `Label` columns corresponding to the path of the image and label relative to the `root_dir`. Additionally, ensure the data is compatible with `InstaGeoDataset`

3. **Training the Model:**

Run training with the necessary flags:

```bash
python -m instageo.model.run \
    root_dir=path/to/root valid_filepath=path/to/valdata \
    train_filepath=path/to/traindata \
    train.learning_rate=0.001 \
    train.num_epochs=100 \
    train.batch_size=4
```

4. **Prediction using:**

*a.* **Sliding Window Inference:**
For training we create chips from tiles, this is necessary because our model can only process an input of size 224 x 224. For the purpose of inference we have a sliding window inference feature that inputs tile and perform a sliding window inference on patches of size 224 x 224. This is useful because it skips the process of creating chips using the `instageo.data.chip_creator`, we only need to download tiles and directly runs inference on them. We can run inference using the following command:

```bash
python -m instageo.model.run --config-name=config.yaml \
    root_dir='path-to-root_dir-containing-dataset.json' \
    test_filepath='dataset.json' \
    train.batch_size=16 \
    test.stride=224 \
    checkpoint_path='path-to-checkpoint' \
    mode=sliding_inference
```

*b.* **Chip Inference:**
This mode performs efficient and optimized inference on geospatial image "chips" using a pre-trained model. It processes the data in batches, makes predictions, and saves the results as TIFF files with the appropriate geospatial metadata. It uses GPU (if available) and multithreading to save files faster.

*Note:* The image size used for training and chip inference must be the same (currently 224) due to preprocessing steps applied during training.

```bash
python -m instageo.model.run --config-name=config.yaml \
    root_dir='path-to-root_dir-containing-dataset.json' \
    test_filepath='dataset.json' \
    train.batch_size=16 \
    test.stride=224 \
    checkpoint_path='path-to-checkpoint' \
    mode=chip_inference
```



5. **Example (Flood Mapping):**
[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) is a geospatial dataset of 10m Sentinel-2 imagery for flood detection.
- Data: Download the Sen1Floods11 hand labelled Sentinel-2 chips as well as `train`, `validation` and `test` splits using the following command
```bash
mkdir sen1floods11
mkdir sen1floods11/S2Hand
mkdir sen1floods11/LabelHand

gsutil cp gs://instageo/data/sen1floods11/flood_train_data.csv sen1floods11
gsutil cp gs://instageo/data/sen1floods11/flood_test_data.csv sen1floods11
gsutil cp gs://instageo/data/sen1floods11/flood_valid_data.csv sen1floods11

gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand sen1floods11/S2Hand
gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand sen1floods11/LabelHand
```

- Model: Fine-tune Prithvi on the dataset by running the following command
```bash
python -m instageo.model.run --config-name=sen1floods11 \
    root_dir=sen1floods11 \
    train_filepath=sen1floods11/flood_train_data.csv \
    valid_filepath=sen1floods11/flood_valid_data.csv \
    train.num_epochs=100
```
After training you are expected to have a checkpoint with mIoU of ~ 89%

- Evaluate: Evaluate the fine-tuned model on test set using the following command.
Replace `path/to/checkpoint/checkpoint.ckpt` with the path to your model checkpoint.
```bash
python -m instageo.model.run --config-name=sen1floods11 \
    root_dir=sen1floods11 test_filepath=sen1floods11/flood_test_data.csv \
    checkpoint_path=path/to/checkpoint/checkpoint.ckpt \
    mode=eval
```
When the saved checkpoint is evaluated on the test set, you should have results comparable to the following

`Class based metrics:`
| Metric            | Class 0 (No Water)                 | Class 1 (Flood/Water)                 |
|-------------------|-----------------------| -----------------------|
| Accuracy          | 0.99    | 0.88
| Intersection over Union (IoU)          | 0.97    | 0.81

`Global metrics:`
| Metric    | Value |
|-----------|-------|
| Overall Accuracy | 0.98 |
| Mean IoU | 0.89 |
| Cross Entropy Loss | 0.11 |

6. **Example (Multi-Temporal Crop Classification):**
[Multi-Temporal Crop Classification](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification) contains Harmonized Landsat-Sentinel (HLS) imagery spanning various land cover and crop type classes throughout the Contiguous United States, captured during the year 2022. The classification labels used in this dataset are based on the Crop Data Layer (CDL) provided by the United States Department of Agriculture (USDA).

- Data: Download the Multi-Temporal Crop Classification data splits using the following command (~13GB)

```bash
gsutil -m cp -r gs://instageo/data/multi-temporal-crop-classification .
```

- Model: Fine-tune Prithvi on the dataset by running the following command

```bash
python -m instageo.model.run --config-name=multitemporal_crop_classification \
    root_dir='multi-temporal-crop-classification' \
    train_filepath='multi-temporal-crop-classification/training_data.csv' \
    valid_filepath='multi-temporal-crop-classification/validation_data.csv' \
    train.batch_size=16 \
    train.num_epochs=100 \
    train.learning_rate=1e-4
```
After training you are expected to have a checkpoint with mIoU of ~ 45%

- Evaluate: Evaluate the fine-tuned model on test set using the following command.
Replace `path/to/checkpoint/checkpoint.ckpt` with the path to your model checkpoint.

```bash
python -m instageo.model.run --config-name=multitemporal_crop_classification \
    root_dir='multi-temporal-crop-classification' \
    test_filepath='multi-temporal-crop-classification/validation_data.csv' \
    train.batch_size=16 \
    checkpoint_path=`path/to/checkpoint/checkpoint.ckpt` \
    mode=eval
```
When the saved checkpoint is evaluated on the test set, you should have results comparable to the following

`Class based metrics:`
| Metric                        | Accuracy | Intersection over Union (IoU)  |
|-------------------------------|-------|-------|
| Natural Vegetation                      | 0.44     | 0.50  |
| Forest | 0.53     | 0.76  |
| Corn                      | 0.62     | 0.73  |
| Soybeans | 0.60     | 0.75  |
| Wetlands                      | 0.45     | 0.59  |
| Developed/Barren | 0.42     | 0.66  |
| Open Water                      | 0.69     | 0.88  |
| Winter Wheat | 0.55     | 0.71  |
| Alfalfa                      | 0.37     | 0.67  |
| Fallow/Idle Cropland | 0.37     | 0.57  |
| Cotton                      | 0.37     | 0.67  |
| Sorghum | 0.36     | 0.65  |
| Other                      | 0.43     | 0.57  |

`Global metrics:`
| Metric    | Value |
|-----------|-------|
| Overall Accuracy | 0.67 |
| Mean IoU | 0.48 |
| Cross Entropy Loss | 0.93 |

7. **Example (Desert Locust Breeding Ground Prediction):**
Desert Locusts Breeding Ground Prediction using HLS dataset. Observation records of breeding grounds are sourced from [UN-FAO Locust Hub](https://locust-hub-hqfao.hub.arcgis.com/) and used to download HLS tiles used for creating chips and segmentation maps.

- Data: The resulting chips and segmentation maps created using `instageo.chip_creator` can be downloaded using the following command (~15GB)
```bash
gsutil -m cp -r gs://instageo/data/locust_breeding .
```

- Model: Fine-tune Prithvi on the dataset by running the following command

```bash
python -m instageo.model.run --config-name=locust \
    root_dir='locust_breeding' \
    train_filepath='locust_breeding/train.csv' \
    valid_filepath='locust_breeding/val.csv'
```
After training you are expected to have a checkpoint with mIoU of 70%

- Evaluate: Evaluate the fine-tuned model on test set using the following command.
Replace `path/to/checkpoint/checkpoint.ckpt` with the path to your model checkpoint.

```bash
python -m instageo.model.run --config-name=locust \
    root_dir='locust_breeding' \
    test_filepath='locust_breeding/test.csv' \
    checkpoint_path=`path/to/checkpoint/checkpoint.ckpt` \
    mode=eval
```
When the saved checkpoint is evaluated on the test set, you should have results comparable to the following

`Class based metrics:`
| Metric                         | Class 0 (Non-Breeding)   | Class 1 (Breeding)     |
|--------------------------------|--------------------------| -----------------------|
| Accuracy                       | 0.85                     | 0.85                   |
| Intersection over Union (IoU)  | 0.74                     | 0.74                   |

`Global metrics:`
| Metric    | Value |
|-----------|-------|
| Overall Accuracy | 0.85 |
| Mean IoU | 0.74 |
| Cross Entropy Loss | 0.40 |

## Advanced Model Features

### Training Custom Models
Utilize the Prithvi geospatial foundational model as a backbone to develop custom models tailored for precise geospatial applications. Supports both classification and regression tasks with temporal and non-temporal inputs.

### Advanced Inference Capabilities
- **Chip Inference**: Efficient batch processing with GPU acceleration and multithreading for TIFF output
- **Ray-based Model Serving**: Scalable model deployment with Ray Serve for production environments

### Model Registry System
Centralized model management with metadata tracking, version control, and Google Cloud Storage integration for model artifacts.

### Enhanced Metrics and Monitoring
Comprehensive evaluation metrics including streaming regression metrics, R² scores, Pearson correlation, and expected error calculations.

### Flexible Architecture
Support for model distillation, custom loss functions, and configurable training pipelines with Hydra configuration management.

## Model Registry Synchronization

InstaGeo provides a model registry system that allows you to download pre-trained models from Google Cloud Storage.

### Setting up Google Cloud Credentials

1. **Install Google Cloud SDK** (if not already installed):
   ```bash
   # For macOS
   brew install google-cloud-sdk

   # For Ubuntu/Debian
   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
   echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
   sudo apt-get update && sudo apt-get install google-cloud-sdk
   ```

2. **Authenticate with Google Cloud**:
   ```bash
   gcloud auth login
   ```

3. **Set up Application Default Credentials**:
   ```bash
   gcloud auth application-default login
   ```

4. **Verify your setup**:
   ```bash
   gcloud auth list
   gsutil ls gs://example/path/
   ```

### Running the Model Registry Sync Script

The `model_registry_sync.sh` script downloads pre-trained models and their configuration files from Google Cloud Storage to your local machine.

**Usage**:
```bash
./instageo/model/registry/model_registry_sync.sh <gs://path/to/registry_file.yaml> <MODELS_DESTINATION_PATH>
```

**Example**:
```bash
# Navigate to the root directory of the project
cd InstaGeo

# Create a directory for models configurations and checkpoints
mkdir -p /path/to/models/folder

# Run the sync script
cd instageo/model/registry && chmod +x model_registry_sync.sh
./model_registry_sync.sh "gs://path/to/registry/file.yaml" /path/to/models/folder
```

### Available Models
- **AOD Estimator**: Aerosol optical depth estimation from satellite imagery
- **Sen1Floods**: Flood area segmentation from Sentinel-1 imagery
- **Biomass Estimator**: Biomass estimation for environmental monitoring
- **Locust Prediction**: Locust breeding ground prediction
- **Crop Classification**: Crop type classification over agricultural regions

Each model may have multiple sizes (e.g., `tiny`, `student`, `teacher`, `normal`) with different parameter counts and performance characteristics.

## Customization

- Use the `stats` mode to compute the `mean`, and `std` lists of your dataset.
- Modify the `bands`, `mean`, and `std` lists in `configs/config.yaml` to match your dataset's characteristics and improve its normalization.
- Implement additional data augmentation strategies in `process_and_augment`.
