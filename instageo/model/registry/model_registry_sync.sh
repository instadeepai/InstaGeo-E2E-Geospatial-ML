#!/usr/bin/env bash

# Example usage:
# ./model_registry_sync.sh gs://path/to/models_registry.yaml /path/to/checkpoint_dir
# This will sync the checkpoint and .hydra directory from the GCS folder to the local directory

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <gs://path/to/models_registry.yaml> <CHECKPOINT_DIR>"
  exit 1
fi

# Get the registry file from the GCS bucket
gsutil cp "$1" models_registry.yaml

MODELS_YAML="models_registry.yaml"
CHECKPOINT_DIR="$2"

yq -r '
  .models
  | to_entries[]
  | .key as $model
  | .value.sizes
  | to_entries[]
  | [$model, .key, .value.gcs_folder]
  | @tsv
' "$MODELS_YAML" | while IFS=$'\t' read -r MODEL_KEY SIZE GCS_FOLDER; do
  if [[ -z "${GCS_FOLDER:-}" || "${GCS_FOLDER}" == "null" ]]; then
    echo "Skipping ${MODEL_KEY}/${SIZE}: missing gcs_folder"
    continue
  fi

  DEST_DIR="${CHECKPOINT_DIR}/${MODEL_KEY}/${SIZE}"
  echo "Syncing ${GCS_FOLDER} -> ${DEST_DIR}"
  mkdir -p "${DEST_DIR}"

  gsutil -m cp "${GCS_FOLDER%/}/instageo_best_checkpoint.ckpt" "${DEST_DIR}/" || {
    echo "Warning: checkpoint missing for ${MODEL_KEY}/${SIZE}"
  }

  gsutil -m cp -r "${GCS_FOLDER%/}/.hydra" "${DEST_DIR}/" || {
    echo "Warning: .hydra missing for ${MODEL_KEY}/${SIZE}"
  }
done

echo "Sync completed."
