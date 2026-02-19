#!/bin/bash
#SBATCH --job-name=postprocess_others
#SBATCH --output=slurm_out/%x_%A.out
#SBATCH --error=slurm_out/%x_%A.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=200G
#SBATCH --partition=cpu
#SBATCH --time=12:00:00

# Usage: sbatch --dependency=afterok:<CONVERSION_JOB_ID> postprocess_others.sh <OUTPUT_NAME>
# Example: sbatch --dependency=afterok:2100 postprocess_others.sh agibot_gripper

set -e  # Exit on error

# ============================================================
# Parse arguments
# ============================================================
if [ $# -lt 1 ]; then
    echo "Error: Missing required argument"
    echo "Usage: sbatch --dependency=afterok:<JOB_ID> $0 <OUTPUT_NAME>"
    echo ""
    echo "Arguments:"
    echo "  OUTPUT_NAME : Dataset name (e.g., agibot_gripper, galaxea, humanoid_everyday_g1)"
    echo ""
    echo "Example:"
    echo "  sbatch --dependency=afterok:2100 $0 agibot_gripper"
    exit 1
fi

OUTPUT_NAME="$1"

# ============================================================
# Configuration
# ============================================================
source /fsx/ubuntu/miniconda3/bin/activate convert

INTERMEDIATE_DIR="/fsx/ubuntu/taeyoung/data/processing/intermediate/others_lerobot_${OUTPUT_NAME}"
LEROBOT_V3_DIR="/fsx/ubuntu/taeyoung/data/processing/lerobot_v3"
FINAL_V3_DIR="${LEROBOT_V3_DIR}/${OUTPUT_NAME}"
FINAL_V2_DIR="${FINAL_V3_DIR}_converted_v21"

# Create directories
mkdir -p slurm_out
mkdir -p "$LEROBOT_V3_DIR"

echo "=========================================="
echo "Post-processing Pipeline for: $OUTPUT_NAME"
echo "=========================================="
echo "Intermediate Directory: $INTERMEDIATE_DIR"
echo "LeRobot v3 Directory: $FINAL_V3_DIR"
echo "LeRobot v2.1 Directory: $FINAL_V2_DIR"
echo ""

# ============================================================
# Step 1: Merge intermediate results
# ============================================================
echo "=========================================="
echo "Step 1/3: Merging intermediate results"
echo "=========================================="

if [ ! -d "$INTERMEDIATE_DIR/_temp_shards" ]; then
    echo "Error: Intermediate directory not found: $INTERMEDIATE_DIR/_temp_shards"
    echo "Make sure the conversion job completed successfully."
    exit 1
fi

python merge_results.py --local-dir "$INTERMEDIATE_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Merge failed"
    exit 1
fi

echo "✓ Merge completed successfully"
echo ""

# ============================================================
# Step 2: Reformat to LeRobot v3
# ============================================================
echo "=========================================="
echo "Step 2/3: Reformatting to LeRobot v3"
echo "=========================================="

# Remove old v3 directory if exists
if [ -d "$FINAL_V3_DIR" ]; then
    echo "Removing existing v3 directory: $FINAL_V3_DIR"
    rm -rf "$FINAL_V3_DIR"
fi

python reformat_to_lerobot_v3.py \
    --source-dir "$INTERMEDIATE_DIR" \
    --output-dir "$LEROBOT_V3_DIR" \
    --dataset-name "$OUTPUT_NAME"

if [ $? -ne 0 ]; then
    echo "Error: v3 reformatting failed"
    exit 1
fi

echo "✓ LeRobot v3 reformatting completed successfully"
echo ""

# ============================================================
# Step 3: Convert v3 to v2.1
# ============================================================
echo "=========================================="
echo "Step 3/3: Converting v3 to v2.1"
echo "=========================================="

# Remove old v2 directory if exists
if [ -d "$FINAL_V2_DIR" ]; then
    echo "Removing existing v2.1 directory: $FINAL_V2_DIR"
    rm -rf "$FINAL_V2_DIR"
fi

python convert_v3_to_v2.py \
    --path "$FINAL_V3_DIR" \
    --force-conversion

if [ $? -ne 0 ]; then
    echo "Error: v3 to v2.1 conversion failed"
    exit 1
fi

echo "✓ v3 to v2.1 conversion completed successfully"
echo ""

# ============================================================
# Step 4: Verification
# ============================================================
echo "=========================================="
echo "Step 4/4: Verification"
echo "=========================================="

# Check v3 directory
echo "Checking LeRobot v3 directory..."
if [ ! -d "$FINAL_V3_DIR" ]; then
    echo "✗ Error: v3 directory not found: $FINAL_V3_DIR"
    exit 1
fi

V3_DATA_FILES=$(find "$FINAL_V3_DIR/data" -name "*.parquet" 2>/dev/null | wc -l)
V3_EPISODE_FILES=$(find "$FINAL_V3_DIR/meta/episodes" -name "*.parquet" 2>/dev/null | wc -l)
V3_VIDEO_FILES=$(find "$FINAL_V3_DIR/videos" -name "*.mp4" 2>/dev/null | wc -l)

echo "  Data files: $V3_DATA_FILES"
echo "  Episode files: $V3_EPISODE_FILES"
echo "  Video files: $V3_VIDEO_FILES"

# Check v2.1 directory
echo ""
echo "Checking LeRobot v2.1 directory..."
if [ ! -d "$FINAL_V2_DIR" ]; then
    echo "✗ Error: v2.1 directory not found: $FINAL_V2_DIR"
    exit 1
fi

V2_DATA_FILES=$(find "$FINAL_V2_DIR/data" -name "episode_*.parquet" 2>/dev/null | wc -l)
V2_VIDEO_FILES=$(find "$FINAL_V2_DIR/videos" -name "episode_*.mp4" 2>/dev/null | wc -l)

# Check for required metadata files
V2_HAS_INFO=$([ -f "$FINAL_V2_DIR/meta/info.json" ] && echo "✓" || echo "✗")
V2_HAS_EPISODES=$([ -f "$FINAL_V2_DIR/meta/episodes.jsonl" ] && echo "✓" || echo "✗")
V2_HAS_STATS=$([ -f "$FINAL_V2_DIR/meta/episodes_stats.jsonl" ] && echo "✓" || echo "✗")
V2_HAS_TASKS=$([ -f "$FINAL_V2_DIR/meta/tasks.jsonl" ] && echo "✓" || echo "✗")

echo "  Data files: $V2_DATA_FILES"
echo "  Video files: $V2_VIDEO_FILES"
echo "  Metadata files:"
echo "    info.json: $V2_HAS_INFO"
echo "    episodes.jsonl: $V2_HAS_EPISODES"
echo "    episodes_stats.jsonl: $V2_HAS_STATS"
echo "    tasks.jsonl: $V2_HAS_TASKS"

# Final check
echo ""
if [ "$V2_HAS_INFO" = "✓" ] && [ "$V2_HAS_EPISODES" = "✓" ] && [ "$V2_HAS_STATS" = "✓" ] && [ "$V2_DATA_FILES" -gt 0 ]; then
    echo "=========================================="
    echo "✓ ALL STEPS COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo ""
    echo "Final outputs:"
    echo "  LeRobot v3: $FINAL_V3_DIR"
    echo "  LeRobot v2.1: $FINAL_V2_DIR"
    echo ""
    echo "You can now use the v2.1 dataset for training."
else
    echo "=========================================="
    echo "✗ VERIFICATION FAILED"
    echo "=========================================="
    echo "Some required files are missing. Please check the logs."
    exit 1
fi
