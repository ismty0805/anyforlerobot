#!/bin/bash
#SBATCH --job-name="install_ffmpeg"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --output=slurm_out/%A_install_ffmpeg.out
#SBATCH --error=slurm_out/%A_install_ffmpeg.err

source /fsx/ubuntu/miniconda3/bin/activate convert

echo "Installing ffmpeg in convert environment..."
conda install -y -c conda-forge ffmpeg

echo "Verifying installation..."
which ffmpeg
ffmpeg -version

echo "ffmpeg installation complete!"
