#!/bin/bash
#SBATCH --job-name=reformat_bridge_droid
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_out/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --array=0-1

source /fsx/ubuntu/miniconda3/bin/activate convert

# Array: 0=bridge, 1=droid
DATASETS=("bridge_openx" "droid_100")
INTERMEDIATE_DIRS=("openx_lerobot_bridge" "openx_lerobot_droid_800")

DATASET_NAME="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
INTERMEDIATE_DIR="${INTERMEDIATE_DIRS[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID: Reformatting $DATASET_NAME"
echo "=========================================="

python reformat_to_lerobot_v3.py \
    --source-dir "/fsx/ubuntu/taeyoung/data/processing/intermediate/${INTERMEDIATE_DIR}" \
    --output-dir "/fsx/ubuntu/taeyoung/data/processing/lerobot_v3" \
    --dataset-name "${DATASET_NAME}"

echo "Task $SLURM_ARRAY_TASK_ID finished."
