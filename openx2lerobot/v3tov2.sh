#!/bin/bash
#SBATCH --job-name="Convert_v3tov2"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu-dy-dynamic-1
#SBATCH --output=slurm_out/%A_%a_v3tov2.out  # %A=Job ID, %a=Array Task ID
#SBATCH --error=slurm_out/%A_%a_v3tov2.err
#SBATCH --array=0-3

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

# 2. 현재 작업에 해당하는 데이터셋 선택
DATA_NAME=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Dataset: $LOCAL_DIR/$DATA_NAME"
echo "Output will be: $LOCAL_DIR/${DATA_NAME}_converted_v21"
echo "=========================================="

# 3. 변환 스크립트 실행
python convert_v3_to_v2.py \
    --path "$LOCAL_DIR/$DATA_NAME" \
    --force-conversion

echo "Task $SLURM_ARRAY_TASK_ID finished."