#!/bin/bash
#SBATCH --job-name=final_merge_all
#SBATCH --output=slurm_out/final_merge_all_%j.out
#SBATCH --error=slurm_out/final_merge_all_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --time=12:00:00

set -e
source /fsx/ubuntu/miniconda3/bin/activate convert

BASE_SRC="/fsx/ubuntu/taeyoung/data/processing/intermediate"
BASE_VLA="/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset"
OPENX_DST="$BASE_VLA/openx_lerobot"

echo "Starting Massive Final Merge..."

# 1. Humanoids & Action Net
echo "Merging Humanoid H1..."
python merge_legacy_to_v21.py --source-dir $BASE_SRC/humanoid_everyday_26action --output-dir $BASE_VLA --dataset-name humanoid_everyday_h1 --fps 30 --robot-type "Humanoid H1"

echo "Merging Humanoid G1..."
python merge_legacy_to_v21.py --source-dir $BASE_SRC/humanoid_everyday_28action --output-dir $BASE_VLA --dataset-name humanoid_everyday_g1 --fps 30 --robot-type "Humanoid G1"

echo "Merging Action Net..."
python merge_legacy_to_v21.py --source-dir $BASE_SRC/merged_action_net --output-dir $BASE_VLA --dataset-name action_net --fps 30 --robot-type "ActionNet"

# 2. Agibot Dexhand
echo "Merging Agibot Dexhand..."
python merge_legacy_to_v21.py --source-dir $BASE_SRC/agibot_dexhand --output-dir $BASE_VLA/agibot --dataset-name agibot_dexhand --fps 30 --robot-type "Agibot"

# 3. OpenX Datasets
echo "Merging bc_z..."
python merge_legacy_to_v21.py --source-dir $BASE_SRC/bc_z --output-dir $OPENX_DST --dataset-name bc_z --fps 10 --robot-type "Google Robot"

echo "Merging fmb_dataset..."
python merge_legacy_to_v21.py --source-dir $BASE_SRC/fmb_dataset --output-dir $OPENX_DST --dataset-name fmb_dataset --fps 10 --robot-type "Franka"

echo "Merging fractal20220817_data..."
python merge_legacy_to_v21.py --source-dir $BASE_SRC/fractal20220817_data --output-dir $OPENX_DST --dataset-name fractal20220817_data --fps 3 --robot-type "Google Robot"

echo "Merging furniture_bench..."
python merge_legacy_to_v21.py --source-dir $BASE_SRC/furniture_bench --output-dir $OPENX_DST --dataset-name furniture_bench --fps 10 --robot-type "Franka"

echo "Merging iamlab_cmu..."
python merge_legacy_to_v21.py --source-dir $BASE_SRC/iamlab_cmu --output-dir $OPENX_DST --dataset-name iamlab_cmu --fps 20 --robot-type "Franka"

echo "All Merges Complete!"
