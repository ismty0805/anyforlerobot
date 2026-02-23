#!/bin/bash
#SBATCH --job-name=convert_fractal_test
#SBATCH --output=slurm_out/%x_%A.out
#SBATCH --error=slurm_out/%x_%A.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --time=00:30:00

set -e
source /fsx/ubuntu/miniconda3/bin/activate convert

DATASET_NAME="fractal20220817_data"
RAW_DIR="/fsx/ubuntu/taeyoung/data/rlds/open-x-embodiment/fractal20220817_data/0.1.0"
LOCAL_DIR="/fsx/ubuntu/taeyoung/data/processing/intermediate/fractal_test"
TOTAL_SHARDS=1024

python openx_native_worker.py \
    --raw-dir "$RAW_DIR" \
    --local-dir "$LOCAL_DIR" \
    --use-videos \
    --job-id 0 \
    --num-slurm_jobs $TOTAL_SHARDS \
    --total-physical-shards $TOTAL_SHARDS \
    --mode legacy
