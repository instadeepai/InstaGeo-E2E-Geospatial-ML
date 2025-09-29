#!/bin/bash

GCS_BUCKET="ENTER_GCS_BUCKET_PATH"
if ["$GCS_BUCKET" = "ENTER_GCS_BUCKET_PATH"]; then
    echo "GCS_BUCKET is not set"
    exit 1
fi

mkdir -p data

gsutil cp -r gs://instageo/data/observation_records/sen1floods/records/* data

python  -m "instageo.data.raster_chip_creator" \
    --raster_path data/sen1floods11/LabelHand \
    --records_file data/sen1floods-val-records.gpkg \
    --chip_size=512 \
    --temporal_step=0 \
    --num_steps=1 \
    --temporal_tolerance=2 \
    --cloud_coverage=100 \
    --output_directory data/val \
    --data_source=S2 \
    --src_crs=4326 --spatial_resolution=8.983152841195215e-05 \
    --daytime_only=false --qa_check=false 2>&1 | tee data/val.log

python -m "instageo.data.raster_chip_creator" \
    --raster_path data/sen1floods11/LabelHand \
    --records_file data/sen1floods-train-records.gpkg \
    --chip_size=512 \
    --temporal_step=0 \
    --num_steps=1 \
    --temporal_tolerance=2 \
    --cloud_coverage=100 \
    --output_directory data/train \
    --data_source=S2 \
    --src_crs=4326 --spatial_resolution=8.983152841195215e-05 \
    --daytime_only=false --qa_check=false 2>&1 | tee data/train.log

python -m "instageo.data.raster_chip_creator" \
    --raster_path data/sen1floods11/LabelHand \
    --records_file data/sen1floods-test-records.gpkg \
    --chip_size=512 \
    --temporal_step=0 \
    --num_steps=1 \
    --temporal_tolerance=2 \
    --cloud_coverage=100 \
    --output_directory data/test \
    --data_source=S2 \
    --src_crs=4326 --spatial_resolution=8.983152841195215e-05 \
    --daytime_only=false --qa_check=false 2>&1 | tee data/test.log

gsutil -m cp -r data ${GCS_BUCKET}/data/sen1floods_replica
