import jsonlines
import os
from pathlib import Path
from tqdm import tqdm

base_path = Path("/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/openx_lerobot")

def fix_dataset_tasks(dataset_path):
    meta_path = dataset_path / "meta"
    episodes_file = meta_path / "episodes.jsonl"
    tasks_file = meta_path / "tasks.jsonl"
    
    if not episodes_file.exists():
        return f"Skip: {dataset_path.name} (No episodes.jsonl)"
    
    # 1. Collect all unique task strings from episodes.jsonl
    unique_tasks = set()
    try:
        with jsonlines.open(episodes_file) as reader:
            for obj in reader:
                tasks = obj.get("tasks", [])
                if not isinstance(tasks, list):
                    tasks = [tasks]
                
                # Check for single 'task' key too
                if not tasks and "task" in obj:
                    tasks = [obj["task"]]
                
                for t in tasks:
                    if t is not None:
                        unique_tasks.add(str(t).strip())
    except Exception as e:
        return f"Error reading episodes for {dataset_path.name}: {e}"

    if not unique_tasks:
        return f"Warning: {dataset_path.name} has NO tasks mentioned in episodes.jsonl"

    # 2. Sort and create new task list
    new_task_list = sorted(list(unique_tasks))
    
    # 3. Read existing tasks for comparison
    old_tasks = set()
    if tasks_file.exists():
        try:
            with jsonlines.open(tasks_file) as reader:
                for obj in reader:
                    old_tasks.add(obj["task"].strip())
        except:
            pass
            
    # 4. Overwrite tasks.jsonl if different or force overwrite
    # We always overwrite to ensure indices are consistent and strings are clean
    with jsonlines.open(tasks_file, mode='w') as writer:
        for i, task in enumerate(new_task_list):
            writer.write({"task_index": i, "task": task})
            
    diff = len(unique_tasks) - len(old_tasks)
    return f"Fixed {dataset_path.name}: {len(new_task_list)} unique tasks. (Diff from old tasks.jsonl: {diff})"

if __name__ == "__main__":
    datasets = sorted([d for d in base_path.iterdir() if d.is_dir()])
    results = []
    for ds in datasets:
        print(f"Processing {ds.name}...")
        results.append(fix_dataset_tasks(ds))
    
    print("\n--- Summary ---")
    for res in results:
        print(res)
