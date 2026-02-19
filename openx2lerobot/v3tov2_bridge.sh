#!/bin/bash
#SBATCH --job-name="v3tov2_bridge"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --partition=cpu
#SBATCH --output=slurm_out/%A_bridge_v3tov2.out
#SBATCH --error=slurm_out/%A_bridge_v3tov2.err
#SBATCH --dependency=afterok:2080_0

source /fsx/ubuntu/miniconda3/bin/activate convert

mkdir -p slurm_out

DATASET_PATH="/fsx/ubuntu/taeyoung/data/processing/lerobot_v3/bridge_openx"

echo "=========================================="
echo "Converting bridge_openx from v3 to v2.1"
echo "Input: $DATASET_PATH"
echo "Output: ${DATASET_PATH}_converted_v21"
echo "=========================================="

python convert_v3_to_v2.py \
    --path "$DATASET_PATH" \
    --force-conversion

echo "Bridge conversion finished."
