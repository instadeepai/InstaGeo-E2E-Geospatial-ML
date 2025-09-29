#!/bin/bash

mkdir -p data/us_cdl_2022
mkdir -p data/train
mkdir -p data/val
mkdir intermediate_labels

GCS_BUCKET="ENTER_GCS_BUCKET_PATH"
if ["$GCS_BUCKET" = "ENTER_GCS_BUCKET_PATH"]; then
    echo "GCS_BUCKET is not set"
    exit 1
fi

gsutil -m cp -r gs://instageo/data/multi-temporal-crop-classification .

python -m "instageo.data.raster_chip_creator" \
    --raster_path multi-temporal-crop-classification/validation_chips \
    --records_file data/multi-temporal-crop-classification-val-records.gpkg \
    --chip_size=224 \
    --temporal_step=50 \
    --num_steps=3 \
    --temporal_tolerance=20 \
    --cloud_coverage=30 \
    --output_directory data/val \
    --masking_strategy=any \
    --mask_types=cloud,near_cloud_or_shadow,cloud_shadow \
    --src_crs=5070 --spatial_resolution=30 \
    --daytime_only=false --qa_check=true 2>&1 | tee data/val.log &

python -m "instageo.data.raster_chip_creator" \
    --raster_path multi-temporal-crop-classification/training_chips \
    --records_file data/multi-temporal-crop-classification-train-records.gpkg \
    --chip_size=224 \
    --temporal_step=50 \
    --num_steps=3 \
    --temporal_tolerance=20 \
    --cloud_coverage=30 \
    --output_directory data/train \
    --masking_strategy=any \
    --mask_types=cloud,near_cloud_or_shadow,cloud_shadow \
    --src_crs=5070 --spatial_resolution=30 \
    --daytime_only=false --qa_check=true 2>&1 | tee data/train.log

gsutil -m cp -r data ${GCS_BUCKET}/data/multitemporal-crop-classification_replica
