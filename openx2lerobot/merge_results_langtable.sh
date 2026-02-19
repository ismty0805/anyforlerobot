#!/bin/bash
#SBATCH --job-name="Convert langtable to lerobot"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96       # 요청하신 코어 수
#SBATCH --mem=200G               # 요청하신 15G보다 약간 여유있게 16G 설정
#SBATCH --partition=cpu # 파티션 이름
#SBATCH --time=12:00:00         # 제한 시간
#SBATCH --output=slurm_out/%A_%a_langtable.out  # %A=Job ID, %a=Array Task ID
#SBATCH --error=slurm_out/%A_%a_langtable.err

source /fsx/ubuntu/miniconda3/bin/activate convert
LOCAL_DIR="/fsx/ubuntu/taeyoung/data/processing/intermediate/openx_lerobot_langtable"



python merge_results.py --local-dir $LOCAL_DIR