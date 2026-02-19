#!/bin/bash
#SBATCH --job-name=others_to_lerobot
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_out/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --time=24:00:00

# Usage: sbatch --array=0-N convert_others_to_lerobot.sh <DATASET_TYPE> <RLDS_DATA_DIR> <OUTPUT_NAME> <NUM_JOBS>
# Example: sbatch --array=0-99 convert_others_to_lerobot.sh agibot_gripper /fsx/ubuntu/taeyoung/data/rlds/agibot/agibot_gripper_0_part1/1.0.0 agibot_gripper 100

set -e  # Exit on error

# ============================================================
# Parse arguments
# ============================================================
if [ $# -lt 4 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch --array=0-N $0 <DATASET_TYPE> <RLDS_DATA_DIR> <OUTPUT_NAME> <NUM_JOBS>"
    echo ""
    echo "Arguments:"
    echo "  DATASET_TYPE  : Dataset type (agibot_gripper, agibot_dexhand, galaxea, humanoid_everyday_g1, humanoid_everyday_h1, action_net, neural_gr1)"
    echo "  RLDS_DATA_DIR : Path to RLDS dataset directory (e.g., /path/to/agibot/agibot_gripper_0_part1/1.0.0)"
    echo "  OUTPUT_NAME   : Output dataset name (e.g., agibot_gripper)"
    echo "  NUM_JOBS      : Number of parallel jobs (should match array size)"
    echo ""
    echo "Example:"
    echo "  sbatch --array=0-99 $0 agibot_gripper /fsx/ubuntu/taeyoung/data/rlds/agibot/agibot_gripper_0_part1/1.0.0 agibot_gripper 100"
    exit 1
fi

DATASET_TYPE="$1"
RLDS_DATA_DIR="$2"
OUTPUT_NAME="$3"
NUM_JOBS="$4"

# ============================================================
# Configuration
# ============================================================
source /fsx/ubuntu/miniconda3/bin/activate convert

INTERMEDIATE_DIR="/fsx/ubuntu/taeyoung/data/processing/intermediate/others_lerobot_${OUTPUT_NAME}"

# Create log directory
mkdir -p slurm_out

# ============================================================
# Auto-detect total physical shards
# ============================================================
echo "=========================================="
echo "Job: $SLURM_ARRAY_TASK_ID / $NUM_JOBS"
echo "Dataset Type: $DATASET_TYPE"
echo "RLDS Data Directory: $RLDS_DATA_DIR"
echo "Output Name: $OUTPUT_NAME"
echo "Intermediate Directory: $INTERMEDIATE_DIR"
echo "=========================================="

# Count total physical shards by finding files matching pattern *-of-XXXXX*
SAMPLE_FILE=$(find "$RLDS_DATA_DIR" -type f -name "*-of-*" | head -1)

if [ -z "$SAMPLE_FILE" ]; then
    echo "Error: No shard files found in $RLDS_DATA_DIR"
    echo "Looking for files matching pattern: *-of-*"
    exit 1
fi

# Extract total shards from filename (e.g., file-00000-of-02048 -> 2048)
TOTAL_PHYSICAL_SHARDS=$(basename "$SAMPLE_FILE" | grep -oP 'of-\K\d+' | head -1)

if [ -z "$TOTAL_PHYSICAL_SHARDS" ]; then
    echo "Error: Could not extract total shards from filename: $SAMPLE_FILE"
    exit 1
fi

echo "Auto-detected total physical shards: $TOTAL_PHYSICAL_SHARDS"
echo "Processing with $NUM_JOBS parallel jobs"
echo "Each job processes approximately $((TOTAL_PHYSICAL_SHARDS / NUM_JOBS)) shards"
echo ""

# ============================================================
# Run conversion
# ============================================================
srun python others_native_worker.py \
    --raw-dir "$RLDS_DATA_DIR" \
    --local-dir "$INTERMEDIATE_DIR" \
    --use-videos \
    --job-id $SLURM_ARRAY_TASK_ID \
    --num-slurm_jobs $NUM_JOBS \
    --total-physical-shards $TOTAL_PHYSICAL_SHARDS

echo "Job $SLURM_ARRAY_TASK_ID finished successfully."
