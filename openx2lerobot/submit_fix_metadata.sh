#!/bin/bash

# Configuration
BASE_VLA="/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset"

repair() {
    local target=$1
    local fps=$2
    local robot=$3
    echo "Submitting repair for $(basename $target)..."
    
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=repair_$(basename $target)
#SBATCH --output=slurm_out/repair_$(basename $target)_%j.out
#SBATCH --error=slurm_out/repair_$(basename $target)_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --time=04:00:00

set -e
source /fsx/ubuntu/miniconda3/bin/activate convert

python repair_lerobot_dataset.py \
    --dataset-dir "$target" \
    --fps $fps \
    --robot-type "$robot"
EOT
}

# 1. Agibot Gripper (30 FPS)
for i in {1..4}; do
    repair "$BASE_VLA/agibot/agibot_gripper0_part$i" 30 "Agibot"
done

# 2. Galaxea (15 FPS)
for i in {1..5}; do
    repair "$BASE_VLA/galaxea/galaxea_part$i" 15 "Galaxea"
done

# 3. Neural Traj (Likely 30 FPS based on GR1)
repair "$BASE_VLA/neural_traj/neural_robocurate_v1" 30 "GR1"
repair "$BASE_VLA/neural_traj/neural_robocurate_v2" 30 "GR1"

echo "All repair jobs submitted!"
