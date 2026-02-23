#!/bin/bash
#SBATCH --job-name=convert_iamlab_cmu
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_out/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --time=04:00:00

# Usage: sbatch --array=0-63 convert_iamlab_cmu.sh
set -e
source /fsx/ubuntu/miniconda3/bin/activate convert

REAL_JOB_ID=$SLURM_ARRAY_TASK_ID

DATASET_NAME="iamlab_cmu_pickup_insert_converted_externally_to_rlds"
RAW_DIR="/fsx/ubuntu/taeyoung/data/rlds/open-x-embodiment/iamlab_cmu_pickup_insert_converted_externally_to_rlds/0.1.0"
LOCAL_DIR="/fsx/ubuntu/taeyoung/data/processing/intermediate/iamlab_cmu"
TOTAL_SHARDS=64

python openx_native_worker.py \
    --raw-dir "$RAW_DIR" \
    --local-dir "$LOCAL_DIR" \
    --use-videos \
    --job-id $REAL_JOB_ID \
    --num-slurm_jobs $TOTAL_SHARDS \
    --total-physical-shards $TOTAL_SHARDS \
    --mode legacy
