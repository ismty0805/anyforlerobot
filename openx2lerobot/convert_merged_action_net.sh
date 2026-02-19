#!/bin/bash
#SBATCH --job-name=convert_merged_action_net
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_out/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --time=12:00:00

# Usage: sbatch --array=0-1023 convert_merged_action_net.sh
set -e
source /fsx/ubuntu/miniconda3/bin/activate convert

OFFSET=${1:-0}
REAL_JOB_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))

DATASET_NAME="merged_action_net"
RAW_DIR="/fsx/ubuntu/taeyoung/data/rlds/merged_action_net/merged_action_net/1.0.0"
LOCAL_DIR="/fsx/ubuntu/taeyoung/data/processing/intermediate/merged_action_net"
TOTAL_SHARDS=1024

python others_native_worker.py \
    --raw-dir "$RAW_DIR" \
    --local-dir "$LOCAL_DIR" \
    --use-videos \
    --job-id $REAL_JOB_ID \
    --num-slurm_jobs $TOTAL_SHARDS \
    --total-physical-shards $TOTAL_SHARDS \
    --mode legacy \
    --dataset-name "$DATASET_NAME"
