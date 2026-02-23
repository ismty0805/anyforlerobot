#!/bin/bash
#SBATCH --job-name=final_merge_openx
#SBATCH --output=slurm_out/final_merge_%j.out
#SBATCH --error=slurm_out/final_merge_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --time=08:00:00

source /fsx/ubuntu/miniconda3/bin/activate convert

BASE_SRC="/fsx/ubuntu/taeyoung/data/processing/intermediate"
BASE_DST="/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/openx_lerobot"

echo "Starting Final Merge..."

echo "1. Merging bc_z..."
python merge_legacy_to_v21.py --source-dir $BASE_SRC/bc_z --output-dir $BASE_DST --dataset-name bc_z --fps 10 --robot-type "Google Robot"

echo "2. Merging fmb_dataset..."
python merge_legacy_to_v21.py --source-dir $BASE_SRC/fmb_dataset --output-dir $BASE_DST --dataset-name fmb_dataset --fps 10 --robot-type "Franka"

echo "3. Merging fractal20220817_data..."
python merge_legacy_to_v21.py --source-dir $BASE_SRC/fractal20220817_data --output-dir $BASE_DST --dataset-name fractal20220817_data --fps 3 --robot-type "Google Robot"

echo "4. Merging furniture_bench..."
python merge_legacy_to_v21.py --source-dir $BASE_SRC/furniture_bench --output-dir $BASE_DST --dataset-name furniture_bench --fps 10 --robot-type "Franka"

echo "5. Merging iamlab_cmu..."
python merge_legacy_to_v21.py --source-dir $BASE_SRC/iamlab_cmu --output-dir $BASE_DST --dataset-name iamlab_cmu --fps 20 --robot-type "Franka"

echo "All Merges Complete!"
