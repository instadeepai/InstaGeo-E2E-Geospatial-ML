# ------------------------------------------------------------------------------
# Install filter-repo
# brew install git-filter-repo # macOS
# or
# sudo apt-get install git-filter-repo # Linux
# or
# pip install git-filter-repo # Python
# ------------------------------------------------------------------------------

#!/usr/bin/env bash
set -euo pipefail

PRIVATE_REMOTE="git@github.com:instadeepai/InstaGeo.git"
PRIVATE_BRANCH="develop"

PUBLIC_REMOTE="git@github.com:instadeepai/InstaGeo-E2E-Geospatial-ML.git"
PUBLIC_BRANCH="develop"   # change to the branch (new/existing) you want to publish

# Fresh clone (non-mirror, so we can isolate a single branch)
rm -rf tmp-publish
git clone --origin private --branch "$PRIVATE_BRANCH" "$PRIVATE_REMOTE" tmp-publish
cd tmp-publish

# Make sure we are on the branch we want
git switch "$PRIVATE_BRANCH"

# Run filter-repo against HEAD only
# list all private files and directories to exclude from public repo
git filter-repo \
  --force \
  --invert-paths \
  --path experiments/ \
  --path manifest.yaml \
  --path push_to_public_repo.sh \
  --path experiments_dir/data/aod \
  --path experiments_dir/data/biomass \
  --path experiments_dir/data/locusts/locust_new_pseudo_absence.sh \
  --path experiments_dir/data/locusts/locust_new_true_absence.sh \
  --path experiments_dir/train/aod \
  --path experiments_dir/train/biomass \
  --path experiments_dir/train/locusts/locust_pseudo_absence.sh \
  --path experiments_dir/train/locusts/locust_true_absence.sh \
  --path experiments_dir/utils/ \
  --path instageo/data/datasets \



# Strip large blobs (e.g., >20 MB)
git filter-repo --strip-blobs-bigger-than 20M --force

# Reset remotes and push to the public repo
git remote add origin "$PUBLIC_REMOTE"
git push -u origin HEAD:"$PUBLIC_BRANCH" --force

# Clean up
cd ..
rm -rf tmp-publish
