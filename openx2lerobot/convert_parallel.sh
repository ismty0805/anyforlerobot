#!/bin/bash
#SBATCH --job-name="Convert droid to lerobot"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62259M
#SBATCH --partition=cpu-compute
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err


DATA_NAME=droid
DATA_VER=1.4.0

/fsx/ec2-user/miniconda3/bin/activate convert
python openx_sharded.py \
    --raw-dir /fsx/ec2-user/data/open-x-embodiment/$DATA_NAME/$DATA_VER \
    --local-dir /fsx/ec2-user/data/openx_lerobot_par \
    --num-workers 32


