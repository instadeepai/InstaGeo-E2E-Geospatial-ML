#!/bin/bash

mkdir -p data
gsutil -m cp gs://instageo/data/locust_breeding/records/* data

python -m "instageo.data.chip_creator" \
    --dataframe_path="data/train.csv" \
    --output_directory="data/train" \
    --min_count=1 \
    --chip_size=224 \
    --temporal_tolerance=5 \
    --temporal_step=30 \
    --num_steps=3 \
    --masking_strategy=each \
    --mask_types=cloud \
    --data_source=HLS \
    --cloud_coverage=100 \
    --processing_method=cog 2>&1 | tee data/locust_replica_train.log

python -m "instageo.data.chip_creator" \
    --dataframe_path="data/val.csv" \
    --output_directory="data/val" \
    --min_count=1 \
    --chip_size=224 \
    --temporal_tolerance=5 \
    --temporal_step=30 \
    --num_steps=3 \
    --masking_strategy=each \
    --mask_types=cloud \
    --data_source=HLS \
    --cloud_coverage=100 \
    --processing_method=cog 2>&1 | tee data/locust_replica_val.log

python -m "instageo.data.chip_creator" \
    --dataframe_path="data/test.csv" \
    --output_directory="data/test" \
    --min_count=1 \
    --chip_size=224 \
    --temporal_tolerance=5 \
    --temporal_step=30 \
    --num_steps=3 \
    --masking_strategy=each \
    --mask_types=cloud \
    --data_source=HLS \
    --cloud_coverage=100 \
    --processing_method=cog 2>&1 | tee data/locust_replica_test.log

gsutil -m cp -r data gs://instageo/data/locust-replica
