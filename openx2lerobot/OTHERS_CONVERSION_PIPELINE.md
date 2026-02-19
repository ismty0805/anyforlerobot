# Others Dataset Conversion Pipeline (Agibot, Galaxea, Humanoid Everyday)

This pipeline converts non-OpenX datasets (agibot_gripper, agibot_dexhand, galaxea, humanoid_everyday_g1, humanoid_everyday_h1, action_net, neural_gr1) to LeRobot format.

## Supported Datasets

- **agibot_gripper**: Agibot with gripper end-effector (34-dim state/action)
- **agibot_dexhand**: Agibot with dexterous hand (44-dim state/action)
- **galaxea**: Galaxea humanoid robot (18-dim state, 26-dim action)
- **humanoid_everyday_g1**: Humanoid Everyday G1 (28-dim state/action)
- **humanoid_everyday_h1**: Humanoid Everyday H1 (26-dim state/action)
- **action_net**: Action Net dataset (44-dim state/action)
- **neural_gr1**: Neural GR1 dataset

## Pipeline Overview

The conversion pipeline consists of two main scripts:

1. **`convert_others_to_lerobot.sh`** - Parallel RLDS to intermediate format conversion
2. **`postprocess_others.sh`** - Merge, reformat to v3, convert to v2.1, and verify

## Prerequisites

- ffmpeg installed in the `convert` conda environment (now available on all nodes)
- The `convert` conda environment must be activated
- Custom transforms and configs in `others_utils/` directory

## Usage

### Step 1: Convert RLDS to LeRobot (Parallel)

```bash
sbatch --array=0-N convert_others_to_lerobot.sh <DATASET_TYPE> <RLDS_DATA_DIR> <OUTPUT_NAME> <NUM_JOBS>
```

**Arguments:**
- `DATASET_TYPE`: Dataset type (agibot_gripper, agibot_dexhand, galaxea, humanoid_everyday_g1, etc.)
- `RLDS_DATA_DIR`: Path to RLDS dataset directory
- `OUTPUT_NAME`: Output dataset name
- `NUM_JOBS`: Number of parallel jobs (should match array size N+1)

**Example for agibot_gripper:**
```bash
# Convert agibot_gripper with 100 parallel jobs
sbatch --array=0-99 convert_others_to_lerobot.sh \
    agibot_gripper \
    /fsx/ubuntu/taeyoung/data/rlds/agibot/agibot_gripper_0_part1/1.0.0 \
    agibot_gripper \
    100
```

This will:
- Auto-detect the total number of physical shards in the RLDS dataset
- Process the data in parallel using 100 jobs
- Save intermediate results to `/fsx/ubuntu/taeyoung/data/processing/intermediate/others_lerobot_agibot_gripper`

**Note the Job ID** from the output (e.g., `Submitted batch job 2100`)

### Step 2: Post-process (Merge, Reformat, Convert, Verify)

```bash
sbatch --dependency=afterok:<JOB_ID> postprocess_others.sh <OUTPUT_NAME>
```

**Arguments:**
- `JOB_ID`: Job ID from Step 1 (ensures this runs after conversion completes)
- `OUTPUT_NAME`: Same dataset name used in Step 1

**Example:**
```bash
# Post-process agibot_gripper after job 2100 completes
sbatch --dependency=afterok:2100 postprocess_others.sh agibot_gripper
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

## Complete Example Workflows

### Agibot Gripper

```bash
# 1. Convert agibot_gripper dataset (100 parallel jobs)
sbatch --array=0-99 convert_others_to_lerobot.sh \
    agibot_gripper \
    /fsx/ubuntu/taeyoung/data/rlds/agibot/agibot_gripper_0_part1/1.0.0 \
    agibot_gripper \
    100

# Note the job ID (e.g., 2100)

# 2. Post-process after conversion completes
sbatch --dependency=afterok:2100 postprocess_others.sh agibot_gripper

# 3. Check status
squeue -u $USER

# 4. View logs
tail -f slurm_out/postprocess_others_*.out
```

### Agibot Dexhand

```bash
# 1. Convert agibot_dexhand dataset
sbatch --array=0-99 convert_others_to_lerobot.sh \
    agibot_dexhand \
    /fsx/ubuntu/taeyoung/data/rlds/agibot/agibot_dexhand/1.0.0 \
    agibot_dexhand \
    100

# 2. Post-process
sbatch --dependency=afterok:<JOB_ID> postprocess_others.sh agibot_dexhand
```

### Galaxea

```bash
# 1. Convert galaxea dataset
sbatch --array=0-99 convert_others_to_lerobot.sh \
    galaxea \
    /fsx/ubuntu/taeyoung/data/rlds/galaxea/1.0.0 \
    galaxea \
    100

# 2. Post-process
sbatch --dependency=afterok:<JOB_ID> postprocess_others.sh galaxea
```

### Humanoid Everyday G1

```bash
# 1. Convert humanoid_everyday_g1 dataset
sbatch --array=0-99 convert_others_to_lerobot.sh \
    humanoid_everyday_g1 \
    /fsx/ubuntu/taeyoung/data/rlds/humanoid_everyday/g1/1.0.0 \
    humanoid_everyday_g1 \
    100

# 2. Post-process
sbatch --dependency=afterok:<JOB_ID> postprocess_others.sh humanoid_everyday_g1
```

## Monitoring Progress

### Check conversion progress:
```bash
# View running jobs
squeue -u $USER

# Check conversion logs
tail -f slurm_out/others_to_lerobot_*_0.out

# Count completed jobs
ls slurm_out/others_to_lerobot_*_*.out | wc -l
```

### Check post-processing progress:
```bash
# View post-processing log
tail -f slurm_out/postprocess_others_*.out

# Check intermediate files
ls /fsx/ubuntu/taeyoung/data/processing/intermediate/others_lerobot_<NAME>/_temp_shards/
```

## Troubleshooting

### Issue: "No shard files found"
- Check that the RLDS_DATA_DIR path is correct
- Ensure the directory contains files matching pattern `*-of-*`

### Issue: "Module not found: others_utils"
- Make sure `others_utils/configs.py` and `others_utils/transforms.py` exist
- Check that the `convert` environment can import from `others_utils`

### Issue: "Merge failed"
- Check that all conversion jobs completed successfully
- Look for errors in individual job logs: `slurm_out/others_to_lerobot_*_*.err`

### Issue: Missing episodes.jsonl in v2.1
- This usually means video conversion failed
- Check that ffmpeg is installed and accessible on all nodes
- Review the conversion error log

## Key Differences from OpenX Pipeline

1. **Different transforms**: Uses `OTHERS_STANDARDIZATION_TRANSFORMS` instead of `OXE_STANDARDIZATION_TRANSFORMS`
2. **Different configs**: Uses `OTHERS_DATASET_CONFIGS` instead of `OXE_DATASET_CONFIGS`
3. **Custom state/action dimensions**: Supports variable-dimension state/action spaces (not just 8-dim)
4. **Absolute actions**: For humanoid robots, actions are already absolute joint positions (no relative computation needed)
5. **Different worker**: Uses `others_native_worker.py` instead of `openx_native_worker.py`

## State/Action Dimensions by Dataset

| Dataset | State Dim | Action Dim | Description |
|---------|-----------|------------|-------------|
| agibot_gripper | 34 | 34 | Left Arm (7) + Left Gripper (1) + Left Leg (6) + Neck (3) + Right Arm (7) + Right Gripper (1) + Right Leg (6) + Waist (3) |
| agibot_dexhand | 44 | 44 | Left Arm (7) + Left Hand (6) + Left Leg (6) + Neck (3) + Right Arm (7) + Right Hand (6) + Right Leg (6) + Waist (3) |
| galaxea | 18 | 26 | Bimanual arms + torso + chassis |
| humanoid_everyday_g1 | 28 | 28 | Bimanual arms (7+7) + hands (7+7) |
| humanoid_everyday_h1 | 26 | 26 | Bimanual arms (7+7) + hands (6+6) |
| action_net | 44 | 44 | Full humanoid state |
| neural_gr1 | 8 | 8 | Default dimensions |

## Notes

- The script auto-detects the total number of physical shards, so you don't need to count them manually
- More parallel jobs = faster conversion, but ensure you have enough compute resources
- The postprocess step runs on a single node with high CPU/memory for efficient merging
- All scripts now work on any node (ffmpeg is installed everywhere)
- Actions for humanoid robots are absolute joint positions, not deltas
