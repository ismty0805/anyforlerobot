#!/bin/bash
#SBATCH --job-name=postprocess_openx
#SBATCH --output=slurm_out/%x_%A.out
#SBATCH --error=slurm_out/%x_%A.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=360G
#SBATCH --partition=cpu
#SBATCH --time=72:00:00

set -e
source /fsx/ubuntu/miniconda3/bin/activate convert

# Args: DATASET_NAME INTERMEDIATE_DIR FPS
DATASET_NAME=$1
INTERMEDIATE_DIR=$2
FPS=${3:-10} # Default 10Hz for OpenX if not specified
LEROBOT_V3_ROOT="/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/openx_lerobot"

if [ -z "$DATASET_NAME" ] || [ -z "$INTERMEDIATE_DIR" ]; then
    echo "Usage: sbatch postprocess_openx.sh DATASET_NAME INTERMEDIATE_DIR [FPS]"
    exit 1
fi

echo "Post-processing $DATASET_NAME from $INTERMEDIATE_DIR"

python merge_legacy_to_v21.py \
    --source-dir "$INTERMEDIATE_DIR" \
    --output-dir "$LEROBOT_V3_ROOT" \
    --dataset-name "$DATASET_NAME" \
    --fps $FPS

echo "Post-processing finished for $DATASET_NAME"
