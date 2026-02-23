#!/bin/bash
datasets=(
    "humanoid_everyday_26action"
    "humanoid_everyday_28action"
    "merged_action_net"
    "agibot_dexhand"
    "bc_z"
    "fmb_dataset"
    "fractal20220817_data"
    "furniture_bench"
    "iamlab_cmu"
)

for ds in "${datasets[@]}"; do
    path="/fsx/ubuntu/taeyoung/data/processing/intermediate/$ds"
    if [ -d "$path" ]; then
        count=$(find "$path/_temp_shards" -name "*.parquet" 2>/dev/null | wc -l)
        echo "$ds: $count"
    else
        echo "$ds: NOT FOUND"
    fi
done
