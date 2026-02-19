#!/bin/bash
#SBATCH --job-name=bridge_native
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_out/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G               # SVT-AV1을 위해 넉넉히 32G
#SBATCH --partition=cpu
#SBATCH --time=24:00:00         # SVT-AV1은 느리므로 시간을 넉넉히
#SBATCH --array=0-127

source /fsx/ubuntu/miniconda3/bin/activate convert

# 로그 폴더 생성
mkdir -p slurm_out

# ==========================================================
# [설정]
DATA_NAME=bridge_orig
DATA_VER=1.0.0
VERSION_DIR="/fsx/ubuntu/taeyoung/data/rlds/open-x-embodiment/$DATA_NAME/$DATA_VER"
LOCAL_DIR="/fsx/ubuntu/taeyoung/data/processing/intermediate/openx_lerobot_bridge"

# DROID는 파일명이 of-02048로 끝나므로 2048개입니다.
TOTAL_PHYSICAL_SHARDS=128
# ==========================================================

echo "Job: $SLURM_ARRAY_TASK_ID / 128"

srun python openx_native_worker.py \
    --raw-dir "$VERSION_DIR" \
    --local-dir "$LOCAL_DIR" \
    --use-videos \
    --job-id $SLURM_ARRAY_TASK_ID \
    --num-slurm_jobs 128 \
    --total-physical-shards $TOTAL_PHYSICAL_SHARDS

echo "Job $SLURM_ARRAY_TASK_ID finished."

