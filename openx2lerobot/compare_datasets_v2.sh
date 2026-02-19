#!/bin/bash
RLDS_BASE="/fsx/ubuntu/taeyoung/data/rlds/open-x-embodiment"
LEROBOT_BASE="/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/openx_lerobot"

datasets=(
    "austin_buds_dataset_converted_externally_to_rlds"
    "austin_sailor_dataset_converted_externally_to_rlds"
    "austin_sirius_dataset_converted_externally_to_rlds"
    "berkeley_autolab_ur5"
    "berkeley_cable_routing"
    "berkeley_fanuc_manipulation"
    "cmu_stretch"
    "dlr_edan_shared_control_converted_externally_to_rlds"
    "dobbe"
    "jaco_play"
    "nyu_franka_play_dataset_converted_externally_to_rlds"
    "roboturk"
    "stanford_hydra_dataset_converted_externally_to_rlds"
    "taco_play"
    "toto"
    "ucsd_kitchen_dataset_converted_externally_to_rlds"
    "utaustin_mutex"
    "viola"
)

printf "%-50s | %-10s | %-10s | %-8s\n" "Dataset" "RLDS" "LeRobot" "Status"
printf -- "----------------------------------------------------------------------------------------\n"

for ds in "${datasets[@]}"; do
    # RLDS Count using jq to sum shardLengths
    # First find the version directory
    v_dir=$(ls -d "$RLDS_BASE/$ds"/*/ 2>/dev/null | head -n 1)
    if [ -d "$v_dir" ]; then
        rlds_info="$v_dir/dataset_info.json"
        if [ -f "$rlds_info" ]; then
             # Sum up shardLengths across all splits
             rlds_count=$(jq '[.splits[].shardLengths[] | tonumber] | add' "$rlds_info" 2>/dev/null)
        else
             rlds_count="NoInfo"
        fi
    else
        rlds_count="NoDir"
    fi

    # LeRobot Count
    lr_ep_file="$LEROBOT_BASE/$ds/meta/episodes.jsonl"
    lr_info_file="$LEROBOT_BASE/$ds/meta/info.json"
    
    if [ -f "$lr_ep_file" ]; then
        lr_count=$(wc -l < "$lr_ep_file")
    elif [ -f "$lr_info_file" ]; then
        lr_count=$(grep "\"total_episodes\"" "$lr_info_file" | awk -F': ' '{print $2}' | tr -d ', ')
    else
        # Check subdirectories if it's nested (sometimes it is)
        lr_count="Missing"
    fi

    status="Mismatch"
    if [[ "$rlds_count" == "$lr_count" ]]; then
        status="OK"
    elif [[ "$rlds_count" == "null" && "$lr_count" == "60064" ]]; then
        # Specialized case if needed
        status="Check"
    fi

    printf "%-50s | %-10s | %-10s | %-8s\n" "$ds" "$rlds_count" "$lr_count" "$status"
done
