#!/bin/bash
# Submit one slurm job per dataset for parallel integrity checking

BASE="/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset"
RESULTS_DIR="/fsx/ubuntu/taeyoung/workspace/any4lerobot/openx2lerobot/integrity_results"
SCRIPT="/fsx/ubuntu/taeyoung/workspace/any4lerobot/openx2lerobot/check_single_dataset.py"

mkdir -p "$RESULTS_DIR"
mkdir -p slurm_out

job_ids=()

submit_check() {
    local ds_path=$1
    local ds_name=$(basename "$ds_path")
    local result_file="$RESULTS_DIR/${ds_name}.json"

    job_id=$(sbatch --parsable \
        --job-name="chk_${ds_name:0:20}" \
        --output="slurm_out/chk_${ds_name}_%j.out" \
        --error="slurm_out/chk_${ds_name}_%j.err" \
        --nodes=1 --cpus-per-task=4 --mem=16G \
        --partition=cpu --time=01:00:00 \
        --wrap="/fsx/ubuntu/miniconda3/bin/conda run -n convert python $SCRIPT \"$ds_path\" \"$result_file\"")

    echo "Submitted job $job_id for $ds_name"
    job_ids+=($job_id)
}

# Top-level datasets
for ds in "$BASE"/humanoid_everyday_h1 "$BASE"/humanoid_everyday_g1 "$BASE"/action_net; do
    [ -d "$ds/meta" ] && submit_check "$ds"
done

# Agibot
for ds in "$BASE"/agibot/*/; do
    [ -f "${ds}meta/info.json" ] && submit_check "${ds%/}"
done

# Galaxea
for ds in "$BASE"/galaxea/*/; do
    [ -f "${ds}meta/info.json" ] && submit_check "${ds%/}"
done

# Neural Traj
for ds in "$BASE"/neural_traj/*/; do
    [ -f "${ds}meta/info.json" ] && submit_check "${ds%/}"
done

# OpenX
for ds in "$BASE"/openx_lerobot/*/; do
    [ -f "${ds}meta/info.json" ] && submit_check "${ds%/}"
done

echo ""
echo "All jobs submitted: ${#job_ids[@]} total"
echo "Job IDs: ${job_ids[*]}"
echo ""
echo "Monitor with: squeue --me"
echo "Collect results with: python collect_integrity_results.py"
