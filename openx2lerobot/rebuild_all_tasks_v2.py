import jsonlines
import os
from pathlib import Path
from tqdm import tqdm

base_path = Path("/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset")

def fix_dataset_tasks(dataset_path):
    # Search for meta dir recursively up to 2 levels (to handle subfolders like openx_lerobot/droid)
    meta_candidates = list(dataset_path.glob("meta")) + list(dataset_path.glob("*/meta"))
    
    results = []
    for meta_dir in meta_candidates:
        episodes_file = meta_dir / "episodes.jsonl"
        tasks_file = meta_dir / "tasks.jsonl"
        
        if not episodes_file.exists():
            continue
            
        ds_name = f"{dataset_path.name}/{meta_dir.parent.name}" if meta_dir.parent != dataset_path else dataset_path.name
        
        # 1. Collect all unique task strings
        unique_tasks = set()
        try:
            with jsonlines.open(episodes_file) as reader:
                for obj in reader:
                    tasks = obj.get("tasks", [])
                    if not isinstance(tasks, list):
                        tasks = [tasks]
                    if not tasks and "task" in obj:
                        tasks = [obj["task"]]
                    
                    for t in tasks:
                        if t is not None:
                            unique_tasks.add(str(t).strip())
        except Exception as e:
            results.append(f"Error reading {ds_name}: {e}")
            continue

        if not unique_tasks:
            results.append(f"Warning: {ds_name} has NO tasks.")
            continue

        # 2. Sort
        new_task_list = sorted(list(unique_tasks))
        
        # 3. Read old
        old_tasks_count = 0
        if tasks_file.exists():
            try:
                with jsonlines.open(tasks_file) as reader:
                    for _ in reader: old_tasks_count += 1
            except: pass

        # 4. Overwrite
        with jsonlines.open(tasks_file, mode='w') as writer:
            for i, task in enumerate(new_task_list):
                writer.write({"task_index": i, "task": task})
        
        results.append(f"Fixed {ds_name}: {len(new_task_list)} tasks. (Previously had {old_tasks_count})")
        
    return results

if __name__ == "__main__":
    top_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    all_results = []
    for d in top_dirs:
        print(f"Checking {d.name}...")
        all_results.extend(fix_dataset_tasks(d))
    
    print("\n--- Summary ---")
    for res in all_results:
        print(res)
