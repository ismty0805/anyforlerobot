#!/bin/bash
#SBATCH --job-name=check_droid
#SBATCH --output=check_droid_status.out
#SBATCH --error=check_droid_status.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=cpu
#SBATCH --time=00:10:00

echo "=== DROID Job Status Check ==="
echo "Total output files:"
ls -1 slurm_out/droid_native_*.out 2>/dev/null | wc -l

echo ""
echo "Jobs with 'finished' message:"
grep -l "Job .* finished" slurm_out/droid_native_*.out 2>/dev/null | wc -l

echo ""
echo "Error files with errors:"
grep -l "error\|Error\|ERROR\|failed\|Failed\|FAILED" slurm_out/droid_native_*.err 2>/dev/null | wc -l

echo ""
echo "Parquet files created:"
ls /fsx/ubuntu/taeyoung/data/processing/intermediate/openx_lerobot_droid/*.parquet 2>/dev/null | wc -l

echo ""
echo "Expected: 200 jobs (array 0-199), processing 2048 physical shards"
echo "Each job should process: 2048/200 = ~10.24 shards"

echo ""
echo "Sample of last 10 output files:"
ls -1 slurm_out/droid_native_*.out 2>/dev/null | tail -10

echo ""
echo "Checking a few output files for completion:"
for f in slurm_out/droid_native_*_{0,50,100,150,199}.out; do
  if [ -f "$f" ]; then
    echo "--- $f ---"
    tail -3 "$f"
  fi
done
