#!/bin/bash
#SBATCH --job-name="Convert_all_v3tov2"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --partition=cpu
#SBATCH --output=slurm_out/%A_%a_v3tov2.out
#SBATCH --error=slurm_out/%A_%a_v3tov2.err
#SBATCH --array=0-3
#SBATCH --dependency=afterok:2080

source /fsx/ubuntu/miniconda3/bin/activate convert

# 로그 폴더 생성
mkdir -p slurm_out

# 0. 기본 경로 설정
LOCAL_DIR="/fsx/ubuntu/taeyoung/data/processing/lerobot_v3"

# 1. 데이터셋 정의 (v3 디렉토리에 있는 실제 이름)
DATASETS=(
"bridge_openx"
"droid_100"
"kuka"
"language_table"
)

# 2. 현재 Task ID에 해당하는 데이터셋 선택
DATASET_NAME="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
DATASET_PATH="$LOCAL_DIR/$DATASET_NAME"

echo "=========================================="
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Dataset: $DATASET_PATH"
echo "Output will be: ${DATASET_PATH}_converted_v21"
echo "=========================================="

# 3. v3 to v2 변환 실행
python convert_v3_to_v2.py \
    --path "$DATASET_PATH" \
    --force-conversion

echo "Task $SLURM_ARRAY_TASK_ID finished."
