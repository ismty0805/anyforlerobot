# OpenX to LeRobot Conversion Pipeline

This directory contains scripts to convert OpenX RLDS datasets to LeRobot format (both v3 and v2.1).

## Pipeline Overview

The conversion pipeline consists of two main scripts:

1. **`convert_rlds_to_lerobot.sh`** - Parallel RLDS to intermediate format conversion
2. **`postprocess_lerobot.sh`** - Merge, reformat to v3, convert to v2.1, and verify

## Prerequisites

- ffmpeg must be installed in the `convert` conda environment
- The `convert` conda environment must be activated

## Usage

### Step 1: Convert RLDS to LeRobot (Parallel)

```bash
sbatch --array=0-N convert_rlds_to_lerobot.sh <RLDS_DATA_DIR> <OUTPUT_NAME> <NUM_JOBS>
```

**Arguments:**
- `RLDS_DATA_DIR`: Path to RLDS dataset directory (e.g., `/fsx/ubuntu/taeyoung/data/rlds/open-x-embodiment/droid/1.4.0`)
- `OUTPUT_NAME`: Output dataset name (e.g., `droid_100`)
- `NUM_JOBS`: Number of parallel jobs (should match array size N+1)

**Example:**
```bash
# Convert DROID dataset with 800 parallel jobs
sbatch --array=0-799 convert_rlds_to_lerobot.sh \
    /fsx/ubuntu/taeyoung/data/rlds/open-x-embodiment/droid/1.4.0 \
    droid_100 \
    800
```

This will:
- Auto-detect the total number of physical shards in the RLDS dataset
- Process the data in parallel using 800 jobs
- Save intermediate results to `/fsx/ubuntu/taeyoung/data/processing/intermediate/openx_lerobot_droid_100`

**Note the Job ID** from the output (e.g., `Submitted batch job 1235`)

### Step 2: Post-process (Merge, Reformat, Convert, Verify)

```bash
sbatch --dependency=afterok:<JOB_ID> postprocess_lerobot.sh <OUTPUT_NAME>
```

**Arguments:**
- `JOB_ID`: Job ID from Step 1 (ensures this runs after conversion completes)
- `OUTPUT_NAME`: Same dataset name used in Step 1

**Example:**
```bash
# Post-process DROID dataset after job 1235 completes
sbatch --dependency=afterok:1235 postprocess_lerobot.sh droid_100
```

This will:
1. **Merge** intermediate results from all parallel jobs
2. **Reformat** to LeRobot v3 format
3. **Convert** v3 to v2.1 format (including video splitting)
4. **Verify** all required files are present

### Output Locations

After successful completion, you'll have:

- **LeRobot v3**: `/fsx/ubuntu/taeyoung/data/processing/lerobot_v3/<OUTPUT_NAME>/`
- **LeRobot v2.1**: `/fsx/ubuntu/taeyoung/data/processing/lerobot_v3/<OUTPUT_NAME>_converted_v21/`

The v2.1 format is ready for training with LeRobot.

## Complete Example Workflow

```bash
# 1. Convert Bridge dataset (200 parallel jobs)
sbatch --array=0-199 convert_rlds_to_lerobot.sh \
    /fsx/ubuntu/taeyoung/data/rlds/open-x-embodiment/bridge_dataset/0.1.0 \
    bridge_openx \
    200

# Note the job ID (e.g., 1234)

# 2. Post-process after conversion completes
sbatch --dependency=afterok:1234 postprocess_lerobot.sh bridge_openx

# 3. Check status
squeue -u $USER

# 4. View logs
tail -f slurm_out/postprocess_lerobot_*.out
```

## Monitoring Progress

### Check conversion progress:
```bash
# View running jobs
squeue -u $USER

# Check conversion logs
tail -f slurm_out/rlds_to_lerobot_*_0.out

# Count completed jobs
ls slurm_out/rlds_to_lerobot_*_*.out | wc -l
```

### Check post-processing progress:
```bash
# View post-processing log
tail -f slurm_out/postprocess_lerobot_*.out

# Check intermediate files
ls /fsx/ubuntu/taeyoung/data/processing/intermediate/openx_lerobot_<NAME>/_temp_shards/
```

## Troubleshooting

### Issue: "No shard files found"
- Check that the RLDS_DATA_DIR path is correct
- Ensure the directory contains files matching pattern `*-of-*`

### Issue: "ffmpeg executable not found"
- Install ffmpeg in the convert environment:
  ```bash
  conda activate convert
  conda install -y -c conda-forge ffmpeg
  ```

### Issue: "Merge failed"
- Check that all conversion jobs completed successfully
- Look for errors in individual job logs: `slurm_out/rlds_to_lerobot_*_*.err`

### Issue: Missing episodes.jsonl in v2.1
- This usually means video conversion failed
- Check that ffmpeg is installed and accessible
- Review the conversion error log

## Dataset-Specific Examples

### DROID (2048 shards, 800 jobs)
```bash
sbatch --array=0-799 convert_rlds_to_lerobot.sh \
    /fsx/ubuntu/taeyoung/data/rlds/open-x-embodiment/droid/1.4.0 \
    droid_100 800
sbatch --dependency=afterok:<JOB_ID> postprocess_lerobot.sh droid_100
```

### Bridge (400 shards, 200 jobs)
```bash
sbatch --array=0-199 convert_rlds_to_lerobot.sh \
    /fsx/ubuntu/taeyoung/data/rlds/open-x-embodiment/bridge_dataset/0.1.0 \
    bridge_openx 200
sbatch --dependency=afterok:<JOB_ID> postprocess_lerobot.sh bridge_openx
```

### KUKA (400 shards, 200 jobs)
```bash
sbatch --array=0-199 convert_rlds_to_lerobot.sh \
    /fsx/ubuntu/taeyoung/data/rlds/open-x-embodiment/kuka/0.1.0 \
    kuka 200
sbatch --dependency=afterok:<JOB_ID> postprocess_lerobot.sh kuka
```

### Language Table (400 shards, 200 jobs)
```bash
sbatch --array=0-199 convert_rlds_to_lerobot.sh \
    /fsx/ubuntu/taeyoung/data/rlds/open-x-embodiment/language_table/0.1.0 \
    language_table 200
sbatch --dependency=afterok:<JOB_ID> postprocess_lerobot.sh language_table
```

## Notes

- The script auto-detects the total number of physical shards, so you don't need to count them manually
- More parallel jobs = faster conversion, but ensure you have enough compute resources
- The postprocess step runs on a single node with high CPU/memory for efficient merging
- All scripts use the `cpu-dy-dynamic-1` node where ffmpeg is installed
