# InstaGeo - Model

## Overview

The **InstaGeo Model** component powers training, validation, and inference using custom deep learning architectures built on the [Prithvi](https://huggingface.co/ibm-nasa-geospatial) family of geospatial foundational models.
It is part of the InstaGeo ecosystem, a modular framework for large-scale geospatial machine learning and AI for Social Good applications.

📄 **Paper:** [InstaGeo: Compute-Efficient Geospatial Machine Learning from Data to Deployment](https://arxiv.org/abs/2510.05617)

<!-- 📦 **Dataset Access:** [Download Dataset Placeholder Link]()

💾 **Model Checkpoints:** [Download Checkpoints Placeholder Link]() -->

---

## Requirements

Install dependencies using [uv](https://docs.astral.sh/uv/) (recommended):

**CPU**

```bash
uv sync --locked --extra model --extra cpu
```

**GPU**

```bash
uv sync --locked --extra model --extra gpu
```

---

## Features

* Supports both **classification** and **regression** tasks
* Handles **temporal** and **non-temporal** inputs
* Uses **Prithvi (V1 or V2)** as backbone for transfer learning
* Reproducible training and evaluation
* CLI and YAML-based configuration (Hydra-compatible)

---

## Usage

### 1. Setting Up Run Arguments

Define training parameters using command-line arguments or YAML configs (see `instageo/model/configs/*.yaml` for templates).

### 2. Dataset Preparation

Prepare your data using the InstaGeo Chip Creator or similar tools.
Your dataset CSV should include `Input` and `Label` columns, with paths relative to `root_dir`.
Take a look at how you can prepare your dataset in the demo notebooks that cover useful data preparation examples (chip creation, cleaning, and splitting). You can use the following notebooks in `notebooks/` as references for reproducible data prepreparation:

- `chip_creator_demo.ipynb` – point-based chip creation
- `raster_chip_creator_demo.ipynb` – raster/bbox-based chip creation
- `data_cleaner_demo.ipynb` – cleaning of chips and segmentation maps dataset
- `data_splitter_demo.ipynb` – train/val/test splits with multiple strategies

 The commands in these notebooks produce CSVs with `Input` and `Label` paths compatible with the training commands below.

### 3. Training Example

```bash
python -m instageo.model.run \
    root_dir=path/to/root \
    valid_filepath=path/to/valdata \
    train_filepath=path/to/traindata \
    train.learning_rate=0.001 \
    train.num_epochs=100 \
    train.batch_size=4
```

### 4. Inference Example (Chip Inference)

```bash
python -m instageo.model.run --config-name=config.yaml \
    root_dir='path-to-root' \
    test_filepath='dataset.json' \
    checkpoint_path='path-to-checkpoint' \
    mode=chip_inference
```

---

## Benchmarks

### 🌊 Flood Mapping (Sen1Floods11)

* Original Dataset: [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)

Create Original Dataset using this command
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


| Model                          | Dataset                | GFM             | mIoU | Acc   | mF1 | ROC-AUC |
| ------------------------------ | ------------------- | --------------- | ---------- | ----- | --------- | ------------- |
| [Baseline](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11)                   | [Original](https://github.com/cloudtostreet/Sen1Floods11)        | Prithvi-V1-100M | 88.3 | --    | 97.3 | --            |
| [InstaGeo-Baseline](https://console.cloud.google.com/storage/browser/instageo/model/sen1floods11-baseline)          | [Original](https://github.com/cloudtostreet/Sen1Floods11)        | Prithvi-V1-100M | 88.53      | 97.24 | 93.71     | 99.16         |
| [InstaGeo-Replica (HLS)](https://console.cloud.google.com/storage/browser/instageo/model/sen1floods11-hls-replica)     | [Replica (HLS)](https://console.cloud.google.com/storage/browser/instageo/data/sen1floods-hls-replica)   | Prithvi-V1-100M | 85.40      | 96.39 | 91.78     | 97.15         |
| [InstaGeo-Replica (S2)](https://console.cloud.google.com/storage/browser/instageo/model/sen1floods11-s2-replica)      | [Replica (S2)](https://console.cloud.google.com/storage/browser/instageo/data/sen1floods-s2-replica)    | Prithvi-V1-100M | 87.80      | 97.07 | 93.26     | 97.61         |

---

### 🌾 Multi-Temporal Crop Classification (HLS 2022)


| Model                   | Dataset             | GFM             | mIoU | Acc   | mF1 | ROC-AUC |
| ----------------------- | ---------------- | --------------- | ---------- | ----- | --------- | ------------- |
| [Baseline](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification)            | [Original](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification)    | Prithvi-V1-100M | 42.7       | 60.7  | --        | --            |
| [InstaGeo-Baseline](https://console.cloud.google.com/storage/browser/instageo/model/multitemporal-crop-segmentation-2022-baseline)   | [Original](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification)     | Prithvi-V1-100M | 48.07      | 65.77 | 64.34     | 95.79         |
| [InstaGeo-Replica](https://console.cloud.google.com/storage/browser/instageo/model/multitemporal-crop-segmentation-2022-replica)    | [Replica](https://console.cloud.google.com/storage/browser/instageo/data/multitemporal-crop-classification-replica)      | Prithvi-V1-100M | 47.87      | 66.10 | 64.19     | 95.82         |

#### Expanded US CDL Variants (Prithvi-V2-300M)

|  Variant                          | Model                                                                                                      | Dataset                                                                                                               | GFM              | mIoU  | Acc   | mF1   | ROC-AUC |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------- | ----- | ----- | ----- | ------- |
| [Published baseline (Prithvi, 2022, 3.8k)](https://arxiv.org/pdf/2412.0273) | - | [Original](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification)                  | Prithvi-V2-300M  | 48.60 | 66.80 |  -    |   -     |
| Expanded 2022 CDL (InstaGeo, 2022, 14k)            | [Browse](https://console.cloud.google.com/storage/browser/instageo/model/multitemporal_crop_segmentation-US-CDL-2022) | [InstaGeo-US-CDL-2022-14k](https://console.cloud.google.com/storage/browser/instageo/data/multitemporal-crop-segmentation-US-CDL-2022) | Prithvi-V2-300M  | 60.65 | 83.02 | 73.46 | 97.99   |
| 2024 CDL (InstaGeo, 2024, 18k)                     | [Browse](https://console.cloud.google.com/storage/browser/instageo/model/multitemporal_crop_segmentation-US-CDL-2024) | [InstaGeo-US-CDL-2024-18k](https://console.cloud.google.com/storage/browser/instageo/data/multitemporal-crop-segmentation-US-CDL-2024)         | Prithvi-V2-300M  | 54.86 | 83.30 | 67.19 | 97.96   |

---

### 🦗 Desert Locust Breeding Ground Prediction

| Model                   | Dataset             | GFM             | mIoU | Acc   | mF1 | ROC-AUC |
| ----------------------- | ---------------- | --------------- | ---------- | ----- | --------- | ------------- |
| Baseline            | [Original](https://console.cloud.google.com/storage/browser/instageo/data/locust_breeding)     | Prithvi-V1-100M | --         | 83.03 | 81.53     | --            |
| [InstaGeo-Baseline](https://console.cloud.google.com/storage/browser/instageo/model/locust-baseline)   | [Original](https://console.cloud.google.com/storage/browser/instageo/data/locust_breeding)     | Prithvi-V1-100M | 71.51      | 83.39 | 83.39     | 86.74         |
| [InstaGeo-Replica](instageo/model/locust-replica)    | [Replica](https://console.cloud.google.com/storage/browser/instageo/data/locust-replica)      | Prithvi-V1-100M | 73.30      | 84.60 | 84.60     | 88.66         |

---


## Advanced Model Features

### 🧠 Training Custom Models

Leverage Prithvi as a geospatial backbone for custom tasks:

* Classification & Regression
* Temporal & Non-temporal input support
* Configurable architectures via Hydra

### ⚙️ Advanced Inference Capabilities

* **Chip Inference** for high-resolution geospatial predictions
* **Ray Serve Integration** for scalable deployment

### 🗂️ Model Registry System

Centralized model management with metadata tracking, version control, and Google Cloud Storage integration for model artifacts.

### 📊 Enhanced Metrics

Comprehensive metrics for both regression and classification:

* mIoU, Accuracy, F1-score
* R², Pearson correlation, Expected error

### 🏗️ Flexible Architecture
Support for model distillation, custom loss functions, and configurable training
pipelines with Hydra configuration management.

## 🗃️ Model Registry Synchronization

InstaGeo provides a model registry system that lets you easily download pre-trained models from Google Cloud Storage.

### ☁️ Setting up Google Cloud Credentials

1. **🛠️ Install Google Cloud SDK** (if not already installed):
   ```bash
   # 🍏 For macOS
   brew install google-cloud-sdk

   # 🐧 For Ubuntu/Debian
   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
   echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
   sudo apt-get update && sudo apt-get install google-cloud-sdk
   ```

2. **🔑 Authenticate with Google Cloud**:
   ```bash
   gcloud auth login
   ```

3. **🪪 Set up Application Default Credentials**:
   ```bash
   gcloud auth application-default login
   ```

4. **✅ Verify your setup**:
   ```bash
   gcloud auth list
   gsutil ls gs://example/path/
   ```

### 🔄 Running the Model Registry Sync Script

The `model_registry_sync.sh` script downloads pre-trained models and their configuration files from Google Cloud Storage to your local machine. 💾

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

---

## Citation

If you use InstaGeo in your research, please cite:

```bibtex
@article{yusuf2025instageo,
  title={InstaGeo: Compute-Efficient Geospatial Machine
Learning from Data to Deployment},
  author={Yusuf, Ibrahim and {et al.}},
  journal={arXiv preprint arXiv:2510.05617},
  year={2025}
}
```
