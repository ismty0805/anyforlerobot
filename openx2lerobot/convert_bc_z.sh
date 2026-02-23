#!/bin/bash
#SBATCH --job-name=convert_bc_z
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_out/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --time=04:00:00

# Usage: sbatch --array=0-999 convert_bc_z.sh [OFFSET]
set -e
source /fsx/ubuntu/miniconda3/bin/activate convert

OFFSET=${1:-0}
REAL_JOB_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))

DATASET_NAME="bc_z"
RAW_DIR="/fsx/ubuntu/taeyoung/data/rlds/open-x-embodiment/bc_z/0.1.0"
LOCAL_DIR="/fsx/ubuntu/taeyoung/data/processing/intermediate/bc_z"
TOTAL_SHARDS=1088

python openx_native_worker.py \
    --raw-dir "$RAW_DIR" \
    --local-dir "$LOCAL_DIR" \
    --use-videos \
    --job-id $REAL_JOB_ID \
    --num-slurm_jobs $TOTAL_SHARDS \
    --total-physical-shards $TOTAL_SHARDS \
    --mode legacy
