import jsonlines
import os
from pathlib import Path
from tqdm import tqdm

base_path = Path("/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/openx_lerobot")

def check_tasks(dataset_path):
    meta_path = dataset_path / "meta"
    episodes_file = meta_path / "episodes.jsonl"
    tasks_file = meta_path / "tasks.jsonl"
    
    if not episodes_file.exists():
        return None
    
    unique_tasks = set()
    try:
        with jsonlines.open(episodes_file) as reader:
            for obj in reader:
                if "tasks" in obj and obj["tasks"]:
                    unique_tasks.update(obj["tasks"])
                elif "task" in obj:
                    unique_tasks.add(obj["task"])
    except Exception as e:
        print(f"Error reading {episodes_file}: {e}")
        return None

    tasks_count = 0
    if tasks_file.exists():
        try:
            with jsonlines.open(tasks_file) as reader:
                for _ in reader:
                    tasks_count += 1
        except:
            pass
            
    return len(unique_tasks), tasks_count, list(unique_tasks)

def fix_tasks(dataset_path, unique_tasks):
    tasks_file = dataset_path / "meta" / "tasks.jsonl"
    task_list = sorted(list(unique_tasks))
    with jsonlines.open(tasks_file, mode='w') as writer:
        for i, task in enumerate(task_list):
            writer.write({"task_index": i, "task": task})
    print(f"Fixed tasks.jsonl for {dataset_path.name}: {len(task_list)} tasks written.")

if __name__ == "__main__":
    datasets = sorted([d for d in base_path.iterdir() if d.is_dir()])
    for ds in datasets:
        print(f"Checking {ds.name}...")
        result = check_tasks(ds)
        if result:
            actual_count, meta_count, task_list = result
            print(f"  - Unique Tasks in episodes: {actual_count}")
            print(f"  - Count in tasks.jsonl: {meta_count}")
            if actual_count != meta_count:
                print(f"  -> DISCREPANCY DETECTED. Fixing...")
                fix_tasks(ds, set(task_list))
        else:
            print(f"  - Skip: No episodes.jsonl found.")
