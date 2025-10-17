#!/bin/bash

mkdir -p data
mkdir intermediate_labels

GCS_BUCKET="ENTER_GCS_BUCKET_PATH"
if ["$GCS_BUCKET" = "ENTER_GCS_BUCKET_PATH"]; then
    echo "GCS_BUCKET is not set"
    exit 1
fi

# # download 2022 us cdl raster
# gsutil cp gs://instageo/data/observation_records/us_cdls/2022_30m_cdls.tif data/us_cdl_raster.tif

# download 2024 us cdl raster
gsutil cp gs://instageo/data/observation_records/us_cdls/2024_30m_cdls.tif data/us_cdl_raster.tif

python experiments_dir/utils/create_instageo_us_cdl_records.py \
    --raster-path 'data/us_cdl_raster.tif' \
    --records-file data/us_cdl_records.gpkg \
    --date 2024-09-01 \
    --output-path intermediate_labels

export DATAPIPELINESETTINGS_BATCH_SIZE=10

python -m "instageo.data.raster_chip_creator" \
    --raster_path intermediate_labels \
    --records_file data/us_cdl_records.gpkg \
    --temporal_step=50 \
    --num_steps=3 \
    --temporal_tolerance=20 \
    --cloud_coverage=30 \
    --output_directory data \
    --masking_strategy=any \
    --mask_types=cloud,near_cloud_or_shadow,cloud_shadow \
    --src_crs=5070 --spatial_resolution=30 \
    --daytime_only=false --qa_check=true 2>&1 | tee data/us_cdl_2022.log

gsutil -m cp -r data ${GCS_BUCKET}/data/multitemporal-crop-classification-instageo
