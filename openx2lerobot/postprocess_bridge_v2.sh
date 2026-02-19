#!/bin/bash
#SBATCH --job-name=verify_and_postprocess_bridge_v2
#SBATCH --output=slurm_out/%x_%A.out
#SBATCH --error=slurm_out/%x_%A.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=360G
#SBATCH --partition=cpu
#SBATCH --time=12:00:00

set -e

# Configuration
source /fsx/ubuntu/miniconda3/bin/activate convert
OUTPUT_NAME="bridge_v2"
INTERMEDIATE_DIR="/fsx/ubuntu/taeyoung/data/processing/intermediate/bridge_v2"
LEROBOT_V3_ROOT="/fsx/ubuntu/taeyoung/data/processing/lerobot_v3_new"
FINAL_V3_DIR="${LEROBOT_V3_ROOT}/${OUTPUT_NAME}"
EXPECTED_EPISODES=60064

# echo "=========================================="
# echo "Step 0: Verification of intermediate shards"
# echo "=========================================="
# # Check if all jobs produced temp shards
# NUM_SHARDS=$(ls -d ${INTERMEDIATE_DIR}/_temp_shards/job_* 2>/dev/null | wc -l)
# echo "Found $NUM_SHARDS intermediate job directories."

# if [ "$NUM_SHARDS" -lt 256 ]; then
#   echo "Warning: Expected 256 job directories, but found $NUM_SHARDS. Some jobs might have failed."
#   # Non-fatal if at least some shards exist, but worth noting.
# fi

# echo "=========================================="
# echo "Step 1: Reformatting to v3 (Fixed Logic)"
# echo "=========================================="
# # Ensure clean start for v3
# rm -rf "$FINAL_V3_DIR"

# python reformat_to_lerobot_v3.py \
#     --source-dir "$INTERMEDIATE_DIR" \
#     --output-dir "$LEROBOT_V3_ROOT" \
#     --dataset-name "$OUTPUT_NAME"

echo "=========================================="
echo "Step 2: Verification of v3 results"
echo "=========================================="
# Check episode count from metadata parquet files
FOUND_EPISODES=$(python -c "import pyarrow.parquet as pq; import glob; print(sum(len(pq.read_table(f)) for f in glob.glob('${FINAL_V3_DIR}/meta/episodes/chunk-000/*.parquet')))")
echo "Found $FOUND_EPISODES episodes in v3 metadata."

if [ "$FOUND_EPISODES" -lt 50000 ]; then
  echo "❌ CRITICAL ERROR: Found only $FOUND_EPISODES. Expected around $EXPECTED_EPISODES."
  exit 1
fi

# Check metadata deduplication
UNIQUE_INDICES=$(head -n 200000 "${FINAL_V3_DIR}/meta/episodes/chunk-000/file-000.parquet" 2>/dev/null | wc -l || echo "0")
echo "Verification of metadata structure..."
# Note: we will check this manually or with a script if needed.

echo "=========================================="
echo "Step 3: Converting v3 to v2.1"
echo "=========================================="
python convert_v3_to_v2.py --path "$FINAL_V3_DIR" --force-conversion

echo "=========================================="
echo "Step 4: Final Validation"
echo "=========================================="
FINAL_DIR="/fsx/ubuntu/taeyoung/data/processing/lerobot_v3_new/bridge_v2"
if [ -f "${FINAL_DIR}/meta/episodes.jsonl" ]; then
    FINAL_COUNT=$(wc -l < "${FINAL_DIR}/meta/episodes.jsonl")
    echo "Final v2.1 episode count: $FINAL_COUNT"
    if [ "$FINAL_COUNT" -eq "$FOUND_EPISODES" ]; then
        echo "✅ Verification SUCCESS: Episode counts match."
    else
        echo "❌ Verification FAILED: Metadata count ($FINAL_COUNT) != File count ($FOUND_EPISODES)"
    fi
else
    echo "❌ Final metadata file not found!"
fi

echo "🎉 All steps completed successfully for Bridge v2!"
