#!/bin/bash
#SBATCH --job-name=mass_repair_others
#SBATCH --output=slurm_out/mass_repair_%j.out
#SBATCH --error=slurm_out/mass_repair_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --time=12:00:00

source /fsx/ubuntu/miniconda3/bin/activate convert
python run_repairs.py
