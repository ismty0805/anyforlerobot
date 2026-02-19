#!/bin/bash
#SBATCH --job-name=postprocess_agibot_dexhand
#SBATCH --output=slurm_out/%x_%A.out
#SBATCH --error=slurm_out/%x_%A.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=360G
#SBATCH --partition=cpu
#SBATCH --time=72:00:00

set -e
source /fsx/ubuntu/miniconda3/bin/activate convert

SOURCE_DIR="/fsx/ubuntu/taeyoung/data/processing/intermediate/agibot_dexhand"
OUTPUT_DIR="/fsx/ubuntu/taeyoung/data/processing/lerobot_v3_new"
DATASET_NAME="agibot_dexhand"

# Merge to LeRobot (v2.1 legacy format)
python merge_legacy_to_v21.py \
    --source-dir "$SOURCE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-name "$DATASET_NAME" \
    --fps 30

echo "Post-processing finished."
