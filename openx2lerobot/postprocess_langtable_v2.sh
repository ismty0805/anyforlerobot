#!/bin/bash
#SBATCH --job-name=postprocess_langtable_v2
#SBATCH --output=slurm_out/%x_%A.out
#SBATCH --error=slurm_out/%x_%A.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=360G
#SBATCH --partition=cpu
#SBATCH --time=12:00:00

set -e
source /fsx/ubuntu/miniconda3/bin/activate convert

# Args: DATASET_NAME
DATASET_NAME="language_table_v2"
INTERMEDIATE_DIR="/fsx/ubuntu/taeyoung/data/processing/intermediate/language_table_v2"
LEROBOT_V3_ROOT="/fsx/ubuntu/taeyoung/data/processing/lerobot_v3_new"
FINAL_V3_DIR="${LEROBOT_V3_ROOT}/${DATASET_NAME}"
EXPECTED_EPISODES=2475

# echo "Step 1: Reformatting to v3 (Re-indexing applied)"
# rm -rf "$FINAL_V3_DIR"
# python reformat_to_lerobot_v3.py --source-dir "$INTERMEDIATE_DIR" --output-dir "$LEROBOT_V3_ROOT" --dataset-name "$DATASET_NAME"

# echo "Step 2: Verification"
# FOUND=$(python -c "import pyarrow.parquet as pq; import glob; print(sum(len(pq.read_table(f)) for f in glob.glob('${FINAL_V3_DIR}/meta/episodes/chunk-000/*.parquet')))")
# echo "Found $FOUND episodes."
# if [ "$FOUND" -lt 400000 ]; then exit 1; fi

echo "Step 3: Converting v3 to v2.1"
python convert_v3_to_v2.py --path "$FINAL_V3_DIR" --force-conversion
