#!/bin/bash

mkdir data

gsutil -m cp -r gs://instageo/data/multitemporal-crop-classification-replica/* data
python experiments_dir/utils/create_dataset.py --root-dir=data/train --train-file=train.csv --name-regex 'chip_(.*).merged_(.*)' --seg-map-pattern 'chip_\1.mask_\2'
python experiments_dir/utils/create_dataset.py --root-dir=data/val --train-file=val.csv --name-regex 'chip_(.*).merged_(.*)' --seg-map-pattern 'chip_\1.mask_\2'

echo "Computing Data Statistics..."
stats_output=$(python -m instageo.model.run --config-name=multitemporal_crop_classification \
    root_dir='.' \
    train.batch_size=64 \
    dataloader.img_size=224 \
    "dataloader.replace_label=[-1, 0]" \
    mode=stats \
    train_filepath='train.csv'
)
json_line=$(echo "$stats_output" | tail -n 1)
echo $json_line
MEAN="$(echo $json_line | jq -r '.mean | join(",")')"
STD="$(echo $json_line | jq -r '.std | join(",")')"
MODEL="prithvi_eo_v1_100"
CLASS_WEIGHTS="$(echo $json_line | jq -r '.class_weights | join(",")')"
echo $stats_output
echo "[${MEAN}]"
echo "[${STD}]"

echo "Create a single Neptune experiment"
NEPTUNE_EXPERIMENT=$(python -c "
import os
from instageo.model.neptune_logger import set_neptune_api_token
import neptune
logger = neptune.init_run(
    api_token=set_neptune_api_token(),
    project=os.environ['NEPTUNE_PROJECT'],
)
logger['gcs_bucket'] = '${GCS_BUCKET}'
print(logger._id)
" | grep -o 'IN[0-9]\+-[0-9]\+' | head -n 1)

echo "Start Training..."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir instageo_exp
python -m instageo.model.run --config-name=multitemporal_crop_classification \
    hydra.run.dir="instageo_exp" \
    root_dir='.' \
    "dataloader.mean=[${MEAN}]" \
    "dataloader.std=[${STD}]" \
    "dataloader.replace_label=[-1, 0]" \
    dataloader.img_size=224 \
    model.freeze_backbone=False \
    train.weight_decay=0.01 \
    "train.class_weights=[${CLASS_WEIGHTS}]" \
    train.batch_size=64 \
    train.num_epochs=100 \
    model.model_name=$MODEL \
    train_filepath='train.csv' \
    valid_filepath='val.csv' \
    +neptune_experiment_id="${NEPTUNE_EXPERIMENT}" 2>&1 | tee instageo_exp/train.log

echo "Evaluating Val Split..."
python -m instageo.model.run --config-path $(pwd)/instageo_exp/.hydra --config-name=config \
    root_dir="." \
    test_filepath="val.csv" \
    train.batch_size=64 \
    checkpoint_path='instageo_exp/instageo_best_checkpoint.ckpt' \
    mode=eval 2>&1 | tee instageo_exp/eval.log

gsutil -m cp -r instageo_exp gs://instageo/model/multitemporal-crop-segmentation-replica
