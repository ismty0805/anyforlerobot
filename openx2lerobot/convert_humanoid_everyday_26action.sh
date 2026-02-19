#!/bin/bash
#SBATCH --job-name=convert_humanoid_everyday_26action
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_out/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --time=12:00:00

# Usage: sbatch --array=0-1023 convert_humanoid_everyday_26action.sh
set -e
source /fsx/ubuntu/miniconda3/bin/activate convert

OFFSET=${1:-0}
REAL_JOB_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))

DATASET_NAME="humanoid_everyday_26action"
RAW_DIR="/fsx/ubuntu/taeyoung/data/rlds/humanoid_everyday_26action/1.0.0"
LOCAL_DIR="/fsx/ubuntu/taeyoung/data/processing/intermediate/humanoid_everyday_26action"
TOTAL_SHARDS=256

python others_native_worker.py \
    --raw-dir "$RAW_DIR" \
    --local-dir "$LOCAL_DIR" \
    --use-videos \
    --job-id $REAL_JOB_ID \
    --num-slurm_jobs $TOTAL_SHARDS \
    --total-physical-shards $TOTAL_SHARDS \
    --mode legacy \
    --dataset-name "$DATASET_NAME"
