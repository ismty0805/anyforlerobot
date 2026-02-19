#!/bin/bash
#SBATCH --job-name=reformat_lerobot_v3
#SBATCH --output=slurm_out/%A_%a_reformat.out
#SBATCH --error=slurm_out/%A_%a_reformat.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --time=6:00:00
#SBATCH --array=0-3

source /fsx/ubuntu/miniconda3/bin/activate convert

# 로그 폴더 생성
mkdir -p slurm_out

# 데이터셋 정의
DATASETS=(
    "openx_lerobot_bridge:bridge_openx"
    "openx_lerobot_droid_800:droid_100"
    "openx_lerobot_kuka:kuka"
    "openx_lerobot_langtable:language_table"
)

# 현재 작업에 해당하는 데이터셋 선택
DATASET_INFO=${DATASETS[$SLURM_ARRAY_TASK_ID]}
SOURCE_NAME=$(echo $DATASET_INFO | cut -d':' -f1)
OUTPUT_NAME=$(echo $DATASET_INFO | cut -d':' -f2)

SOURCE_DIR="/fsx/ubuntu/taeyoung/data/processing/intermediate/$SOURCE_NAME"
OUTPUT_DIR="/fsx/ubuntu/taeyoung/data/processing/lerobot_v3"

echo "=========================================="
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Source: $SOURCE_DIR"
echo "Output: $OUTPUT_DIR/$OUTPUT_NAME"
echo "=========================================="

# 리포매팅 실행
python reformat_to_lerobot_v3.py \
    --source-dir "$SOURCE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-name "$OUTPUT_NAME"

echo "Task $SLURM_ARRAY_TASK_ID finished."
