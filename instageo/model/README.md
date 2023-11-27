# InstaGeo Training Module

## Overview

The InstaGeo Training Module is designed to train deep learning models for geospatial data analysis using PyTorch and PyTorch Lightning frameworks. The module focuses on segmenting geospatial imagery using the [Prithvi_100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) foundational model. It includes features for dataset loading, preprocessing, and augmentation, along with a training loop setup.

## Requirements

See `requirements.txt`

## Features

- **Custom Dataset Handling:** Integration with `InstaGeoDataset` for handling geospatial datasets.
- **Data Augmentation:** Support for data augmentation and preprocessing.
- **Training and Validation:** Configurable training and validation data loaders.
- **Model Training:** Utilizes the `PrithviSeg` model for segmentation tasks.
- **Flexible Configuration:** Command-line flags for easy configuration of training parameters.

## Installation

Ensure you have the required libraries installed:

```bash
pip install -r requirements.txt
```

## Usage

1. **Setting Up Flags:** Configure the training parameters using command-line flags.

    - `root_dir`: Root directory of the dataset.
    - `valid_filepath`: File path for validation data.
    - `train_filepath`: File path for training data.
    - `test_filepath`: File path for test data.
    - `checkpoint_path`: File path for model checkpoint.
    - `learning_rate`: Initial learning rate.
    - `num_epochs`: Number of training epochs.
    - `batch_size`: Batch size for training and validation.
    - `mode`: Select one of training or evaluation mode.

2. **Dataset Preparation:** Prepare your geospatial data using the InstaGeo Chip Creator or similar and place it in the specified `root_dir`. Ensure that the csv file for each dataset has `Input` and `Label` columns corresponding to the path of the image and label relative to the `root_dir`. Additionally, ensure the data is compatible with `InstaGeoDataset`

3. **Training the Model:**

    Run the module with the necessary flags:

    ```bash
    python train.py --root_dir=path/to/root --valid_filepath=path/to/valdata --train_filepath=path/to/traindata --learning_rate=0.001 --num_epochs=100 --batch_size=4
    ```

4. **Example (Flood Mapping):**
[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) is a geospatial dataset of 10m Sentinel-2 imagery for flood detection.
- Data: Download the Sen1Floods11 hand labelled Sentinel-2 chips as well as `train`, `validation` and `test` splits using the following command
```
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
```
python train.py --root_dir sen1floods11 --train_filepath sen1floods11_mini/flood_train_data.csv --valid_filepath sen1floods11_mini/flood_valid_data.csv --num_epochs 10
```

- Evaluate: Evaluate the fine-tuned model on test set using the following command.
Replace `path/to/checkpoint/checkpoint.ckpt` with the path to your model checkpoint.
```
python train.py --root_dir sen1floods11_mini --test_filepath sen1floods11_mini/train_data.csv --checkpoint_path path/to/checkpoint/checkpoint.ckpt --mode eval
```



## Customization

- Modify the `BANDS`, `MEAN`, and `STD` lists to match your dataset's characteristics.
- Implement additional data augmentation strategies in `process_and_augment`.
