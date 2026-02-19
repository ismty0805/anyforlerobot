#!/bin/bash
#SBATCH --job-name=postprocess_agibot_v2
#SBATCH --output=slurm_out/%x_%A.out
#SBATCH --error=slurm_out/%x_%A.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=360G
#SBATCH --partition=cpu
#SBATCH --time=72:00:00

set -e
source /fsx/ubuntu/miniconda3/bin/activate convert

# Args: DATASET_NAME
DATASET_NAME="agibot_v2"
INTERMEDIATE_DIR="/fsx/ubuntu/taeyoung/data/processing/intermediate/agibot_v2"
LEROBOT_V3_ROOT="/fsx/ubuntu/taeyoung/data/processing/lerobot_v3_new"
FINAL_V3_DIR="${LEROBOT_V3_ROOT}/${DATASET_NAME}"
EXPECTED_EPISODES=59137

echo "Step 1: Merging legacy shards to final v2.1 format"
rm -rf "$FINAL_V3_DIR"
python merge_legacy_to_v21.py \
    --source-dir "$INTERMEDIATE_DIR" \
    --output-dir "$LEROBOT_V3_ROOT" \
    --dataset-name "$DATASET_NAME" \
    --fps 30

echo "Step 2: Verification"
# Verification for legacy (per-episode files in data/chunk-000)
FOUND=$(ls ${FINAL_V3_DIR}/data/chunk-*/episode_*.parquet 2>/dev/null | wc -l)
echo "Found $FOUND episodes."
if [ "$FOUND" -lt 55000 ]; then 
    echo "❌ ERROR: Expected ~$EXPECTED_EPISODES episodes, but only found $FOUND"
    exit 1 
fi

echo "🎉 Agibot Post-processing completed successfully!"
