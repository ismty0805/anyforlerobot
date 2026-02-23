#!/bin/bash
# Submit one slurm job per dataset for parallel task index fixing

BASE="/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset"
SCRIPT="/fsx/ubuntu/taeyoung/workspace/any4lerobot/openx2lerobot/fix_parquet_task_index.py"

mkdir -p slurm_out

datasets_to_fix=(
    "$BASE/action_net"
    "$BASE/agibot/agibot_dexhand"
    "$BASE/agibot/agibot_gripper0_part1"
    "$BASE/agibot/agibot_gripper0_part2"
    "$BASE/agibot/agibot_gripper0_part3"
    "$BASE/agibot/agibot_gripper0_part4"
    "$BASE/galaxea/galaxea_part1"
    "$BASE/galaxea/galaxea_part2"
    "$BASE/galaxea/galaxea_part3"
    "$BASE/galaxea/galaxea_part4"
    "$BASE/galaxea/galaxea_part5"
    "$BASE/humanoid_everyday_g1"
    "$BASE/humanoid_everyday_h1"
    "$BASE/openx_lerobot/bc_z"
    "$BASE/openx_lerobot/fmb_dataset"
    "$BASE/openx_lerobot/fractal20220817_data"
    "$BASE/openx_lerobot/furniture_bench_dataset_converted_externally_to_rlds"
    "$BASE/openx_lerobot/iamlab_cmu_pickup_insert_converted_externally_to_rlds"
)

for ds_path in "${datasets_to_fix[@]}"; do
    if [ -d "$ds_path" ]; then
        ds_name=$(basename "$ds_path")
        sbatch \
            --job-name="fix_${ds_name:0:20}" \
            --output="slurm_out/fix_${ds_name}_%j.out" \
            --error="slurm_out/fix_${ds_name}_%j.err" \
            --nodes=1 --cpus-per-task=8 --mem=32G \
            --partition=cpu --time=02:00:00 \
            --wrap="/fsx/ubuntu/miniconda3/bin/conda run -n convert python $SCRIPT \"$ds_path\""
    else
        echo "Warning: Directory not found - $ds_path"
    fi
done

echo "All jobs submitted. Monitor with squeue --me"
