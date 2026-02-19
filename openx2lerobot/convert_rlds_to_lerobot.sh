#!/bin/bash
#SBATCH --job-name=rlds_to_lerobot
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_out/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --time=24:00:00

# Usage: sbatch --array=0-N convert_rlds_to_lerobot.sh <RLDS_DATA_DIR> <OUTPUT_NAME> <NUM_JOBS>
# Example: sbatch --array=0-799 convert_rlds_to_lerobot.sh /path/to/droid/1.4.0 droid_100 800

set -e  # Exit on error

# ============================================================
# Parse arguments
# ============================================================
if [ $# -lt 3 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch --array=0-N $0 <RLDS_DATA_DIR> <OUTPUT_NAME> <NUM_JOBS>"
    echo ""
    echo "Arguments:"
    echo "  RLDS_DATA_DIR : Path to RLDS dataset directory (e.g., /path/to/droid/1.4.0)"
    echo "  OUTPUT_NAME   : Output dataset name (e.g., droid_100)"
    echo "  NUM_JOBS      : Number of parallel jobs (should match array size)"
    echo ""
    echo "Example:"
    echo "  sbatch --array=0-799 $0 /fsx/ubuntu/taeyoung/data/rlds/open-x-embodiment/droid/1.4.0 droid_100 800"
    exit 1
fi

RLDS_DATA_DIR="$1"
OUTPUT_NAME="$2"
NUM_JOBS="$3"

# ============================================================
# Configuration
# ============================================================
source /fsx/ubuntu/miniconda3/bin/activate convert

INTERMEDIATE_DIR="/fsx/ubuntu/taeyoung/data/processing/intermediate/openx_lerobot_${OUTPUT_NAME}"

# Create log directory
mkdir -p slurm_out

# ============================================================
# Auto-detect total physical shards
# ============================================================
echo "=========================================="
echo "Job: $SLURM_ARRAY_TASK_ID / $NUM_JOBS"
echo "RLDS Data Directory: $RLDS_DATA_DIR"
echo "Output Name: $OUTPUT_NAME"
echo "Intermediate Directory: $INTERMEDIATE_DIR"
echo "=========================================="

# Count total physical shards by finding files matching pattern *-of-XXXXX*
# This works for both tfrecord and other formats
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
srun python openx_native_worker.py \
    --raw-dir "$RLDS_DATA_DIR" \
    --local-dir "$INTERMEDIATE_DIR" \
    --use-videos \
    --job-id $SLURM_ARRAY_TASK_ID \
    --num-slurm_jobs $NUM_JOBS \
    --total-physical-shards $TOTAL_PHYSICAL_SHARDS

echo "Job $SLURM_ARRAY_TASK_ID finished successfully."
