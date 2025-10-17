#!/bin/bash

DATETIME=$(date +"%Y-%m-%dT%H:%M:%S")
GCS_BUCKET="gs://instageo-internal/model/locust_breeding-${DATETIME}"
mkdir data

gsutil -m cp -r gs://instageo-internal/data/locust-replica-2025-04-17T10:20:35/* data/

python experiments_dir/utils/create_dataset.py --root-dir=data/train --train-file=train.csv
python experiments_dir/utils/create_dataset.py --root-dir=data/val --train-file=val.csv
python experiments_dir/utils/create_dataset.py --root-dir=data/test --train-file=test.csv

python -c "import glob, os, numpy as np, rasterio; from collections import Counter; label_counts = Counter(); [label_counts.update(dict(zip(*np.unique(rasterio.open(f).read(1), return_counts=True)))) for f in glob.glob('data/train/seg_maps/*.tif')]; print('Label counts:'); [print(f'Label {k}: {v} pixels') for k, v in sorted(label_counts.items())]"
python -c "import glob, os, numpy as np, rasterio; from collections import Counter; label_counts = Counter(); [label_counts.update(dict(zip(*np.unique(rasterio.open(f).read(1), return_counts=True)))) for f in glob.glob('data/val/seg_maps/*.tif')]; print('Label counts:'); [print(f'Label {k}: {v} pixels') for k, v in sorted(label_counts.items())]"
python -c "import glob, os, numpy as np, rasterio; from collections import Counter; label_counts = Counter(); [label_counts.update(dict(zip(*np.unique(rasterio.open(f).read(1), return_counts=True)))) for f in glob.glob('data/test/seg_maps/*.tif')]; print('Label counts:'); [print(f'Label {k}: {v} pixels') for k, v in sorted(label_counts.items())]"

echo "Computing Data Statistics..."
stats_output=$(python -m instageo.model.run --config-name=locust \
    root_dir='.' \
    train.batch_size=64 \
    dataloader.img_size=224 \
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
python -m instageo.model.run --config-name=locust \
    hydra.run.dir="instageo_exp" \
    root_dir='.' \
    "dataloader.mean=[${MEAN}]" \
    "dataloader.std=[${STD}]" \
    train.batch_size=8 \
    train.num_epochs=30 \
    model.model_name=$MODEL \
    train.weight_decay=0.3 \
    train_filepath='train.csv' \
    valid_filepath='val.csv' \
    +neptune_experiment_id="${NEPTUNE_EXPERIMENT}" 2>&1 | tee instageo_exp/train.log

echo "Evaluating Val Split..."
python -m instageo.model.run --config-path $(pwd)/instageo_exp/.hydra --config-name=config \
    root_dir='.' \
    test_filepath='val.csv' \
    train.batch_size=64 \
    checkpoint_path='instageo_exp/instageo_best_checkpoint.ckpt' \
    mode=eval 2>&1 | tee instageo_exp/eval.log

echo "Evaluating Test Split..."
python -m instageo.model.run --config-path $(pwd)/instageo_exp/.hydra --config-name=config \
    root_dir='.' \
    test_filepath='test.csv' \
    train.batch_size=64 \
    checkpoint_path='instageo_exp/instageo_best_checkpoint.ckpt' \
    mode=eval 2>&1 | tee instageo_exp/test.log

gsutil -m cp -r instageo_exp ${GCS_BUCKET}
