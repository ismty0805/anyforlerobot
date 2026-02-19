#!/bin/bash
RLDS_BASE="/fsx/ubuntu/taeyoung/data/rlds/open-x-embodiment"
LEROBOT_BASE="/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/openx2lerobot/vla_pretrain_dataset/openx_lerobot"
# Fix path based on the previous ls output which was /fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/openx_lerobot/
LEROBOT_BASE="/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/openx_lerobot"

datasets=(
    "austin_buds_dataset_converted_externally_to_rlds"
    "berkeley_autolab_ur5"
    "berkeley_cable_routing"
    "berkeley_fanuc_manipulation"
    "cmu_stretch"
    "dobbe"
    "jaco_play"
    "roboturk"
    "taco_play"
    "toto"
    "viola"
)

printf "%-40s | %-15s | %-15s | %-10s\n" "Dataset" "RLDS Ep" "LeRobot Ep" "Status"
printf -- "---------------------------------------------------------------------------------------------\n"

for ds in "${datasets[@]}"; do
    # RLDS Count
    rlds_info=$(find "$RLDS_BASE/$ds" -name "dataset_info.json" -print -quit)
    if [ -n "$rlds_info" ]; then
        rlds_count=$(grep "\"num_examples\"" "$rlds_info" | awk -F': ' '{print $2}' | tr -d ', ' | head -n 1)
    else
        rlds_count="N/A"
    fi

    # LeRobot Count
    lr_ep_file="$LEROBOT_BASE/$ds/meta/episodes.jsonl"
    lr_info_file="$LEROBOT_BASE/$ds/meta/info.json"
    
    if [ -f "$lr_ep_file" ]; then
        lr_count=$(wc -l < "$lr_ep_file")
    elif [ -f "$lr_info_file" ]; then
        lr_count=$(grep "\"total_episodes\"" "$lr_info_file" | awk -F': ' '{print $2}' | tr -d ', ')
    else
        lr_count="Missing"
    fi

    status="Mismatch"
    if [ "$rlds_count" == "$lr_count" ]; then
        status="OK"
    fi
    if [ "$lr_count" == "Missing" ]; then
        status="Missing"
    fi

    printf "%-40s | %-15s | %-15s | %-10s\n" "$ds" "$rlds_count" "$lr_count" "$status"
done
