#!/bin/bash
#SBATCH --job-name="upload_info_json"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --output=slurm_out/upload_info_json_%j.out
#SBATCH --error=slurm_out/upload_info_json_%j.err

# Fast upload of specific files
AWS_PROFILE=alin /fsx/ubuntu/miniconda3/bin/s5cmd run /fsx/ubuntu/taeyoung/workspace/any4lerobot/openx2lerobot/s5cmd_info_json_tasks.txt
