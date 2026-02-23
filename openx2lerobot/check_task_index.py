import json
import pandas as pd
from pathlib import Path
import os
import io
import sys

def get_datasets(base_dirs):
    datasets = []
    for base in base_dirs:
        base_path = Path(base)
        if not base_path.exists():
            continue
        for item in base_path.iterdir():
            if not item.is_dir():
                continue
            if (item / "meta/info.json").exists():
                datasets.append(item)
            else:
                for sub in item.iterdir():
                    if sub.is_dir() and (sub / "meta/info.json").exists():
                        datasets.append(sub)
    return sorted(list(set(datasets)))

def check_task_index(ds_path):
    print(f"\n[{ds_path.name}] Checking task_index...")
    
    # Check info.json
    info_path = ds_path / "meta/info.json"
    has_task_index_in_info = False
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
            has_task_index_in_info = "task_index" in info.get("features", {})
            print(f"  - info.json features has task_index: {has_task_index_in_info}")
            
    # Check episodes.jsonl
    episodes_path = ds_path / "meta/episodes.jsonl"
    task_indices_jsonl = set()
    if episodes_path.exists():
        with open(episodes_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if "tasks" in data:
                    task_indices_jsonl.update(data["tasks"])
                # some datasets might have it directly
                elif "task_index" in data:
                    task_indices_jsonl.add(data["task_index"])
        print(f"  - episodes.jsonl has tasks: {bool(task_indices_jsonl)}")
        if task_indices_jsonl:
            print(f"    - Unique tasks indices found: {len(task_indices_jsonl)}")

    # Check first parquet file
    first_parquet = next(ds_path.glob("data/**/*.parquet"), None)
    if first_parquet:
        try:
            df = pd.read_parquet(first_parquet, columns=["task_index"])
            has_task_index_in_parquet = "task_index" in df.columns
            print(f"  - First parquet has task_index column: {has_task_index_in_parquet}")
            if has_task_index_in_parquet:
                unique_indices = df["task_index"].unique()
                print(f"    - Unique task indices in first parquet: {len(unique_indices)}")
        except Exception as e:
            # Maybe the column doesn't exist at all
            if "Column 'task_index' not found" in str(e):
                print(f"  - First parquet has task_index column: False")
            else:
                print(f"  - Error reading parquet: {e}")
    else:
        print("  - No parquet files found.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Check specific dataset if provided
        check_task_index(Path(sys.argv[1]))
    else:
        # Check all datasets
        base_dirs = [
            "/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset",
            "/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/openx_lerobot"
        ]
        datasets = get_datasets(base_dirs)
        for ds in datasets:
            check_task_index(ds)
