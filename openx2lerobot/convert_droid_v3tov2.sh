#!/bin/bash
#SBATCH --job-name="Convert_droid_v3tov2"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --partition=cpu
#SBATCH --output=slurm_out/%A_droid_v3tov2.out
#SBATCH --error=slurm_out/%A_droid_v3tov2.err

source /fsx/ubuntu/miniconda3/bin/activate convert

# 로그 폴더 생성
mkdir -p slurm_out

LOCAL_DIR="/fsx/ubuntu/taeyoung/data/processing/lerobot_v3"

echo "=========================================="
echo "Processing Dataset: $LOCAL_DIR/droid_100"
echo "Output will be: $LOCAL_DIR/droid_100_converted_v21"
echo "=========================================="

python convert_v3_to_v2.py \
    --path "$LOCAL_DIR/droid_100" \
    --force-conversion

echo "DROID v3 to v2 conversion finished."
