#!/bin/bash
#SBATCH --job-name=convert_agibot_v2
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_out/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --time=04:00:00

# Usage: sbatch --array=0-999 convert_agibot_v2.sh [OFFSET]
set -e
source /fsx/ubuntu/miniconda3/bin/activate convert

OFFSET=${1:-0}
REAL_JOB_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))

DATASET_NAME="agibot_gripper_0_part1"
RAW_DIR="/fsx/ubuntu/taeyoung/data/rlds/agibot/agibot_gripper_0_part1/1.0.0"
LOCAL_DIR="/fsx/ubuntu/taeyoung/data/processing/intermediate/agibot_v2"
TOTAL_SHARDS=1024

python others_native_worker.py \
    --raw-dir "$RAW_DIR" \
    --local-dir "$LOCAL_DIR" \
    --use-videos \
    --job-id $REAL_JOB_ID \
    --num-slurm_jobs $TOTAL_SHARDS \
    --total-physical-shards $TOTAL_SHARDS \
    --mode legacy
