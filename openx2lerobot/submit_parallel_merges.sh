#!/bin/bash

# Configuration
BASE_SRC="/fsx/ubuntu/taeyoung/data/processing/intermediate"
BASE_VLA="/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset"
OPENX_DST="$BASE_VLA/openx_lerobot"

# Function to submit a merge job
submit_merge() {
    local dataset_name=$1
    local source=$2
    local output_dir=$3
    local fps=$4
    local robot_type=$5
    local job_name="merge_$dataset_name"

    echo "Submitting merge for $dataset_name..."
    
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=slurm_out/${job_name}_%j.out
#SBATCH --error=slurm_out/${job_name}_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --time=04:00:00

set -e
source /fsx/ubuntu/miniconda3/bin/activate convert

python merge_legacy_to_v21.py \
    --source-dir "$source" \
    --output-dir "$output_dir" \
    --dataset-name "$dataset_name" \
    --fps $fps \
    --robot-type "$robot_type"
EOT
}

# 1. Humanoids & Action Net
submit_merge "humanoid_everyday_h1" "$BASE_SRC/humanoid_everyday_26action" "$BASE_VLA" 30 "Humanoid H1"
submit_merge "humanoid_everyday_g1" "$BASE_SRC/humanoid_everyday_28action" "$BASE_VLA" 30 "Humanoid G1"
submit_merge "action_net" "$BASE_SRC/merged_action_net" "$BASE_VLA" 30 "ActionNet"

# 2. Agibot Dexhand
submit_merge "agibot_dexhand" "$BASE_SRC/agibot_dexhand" "$BASE_VLA/agibot" 30 "Agibot"

# 3. OpenX Datasets
submit_merge "bc_z" "$BASE_SRC/bc_z" "$OPENX_DST" 10 "Google Robot"
submit_merge "fmb_dataset" "$BASE_SRC/fmb_dataset" "$OPENX_DST" 10 "Franka"
submit_merge "fractal20220817_data" "$BASE_SRC/fractal20220817_data" "$OPENX_DST" 3 "Google Robot"
submit_merge "furniture_bench" "$BASE_SRC/furniture_bench" "$OPENX_DST" 10 "Franka"
submit_merge "iamlab_cmu" "$BASE_SRC/iamlab_cmu" "$OPENX_DST" 20 "Franka"

echo "All parallel merge jobs submitted!"
