import json
import logging
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import shutil
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fix_single_parquet(pq_file, ep_to_task):
    try:
        # Read original table
        table = pq.read_table(pq_file)
        
        if "task_index" in table.schema.names:
            return f"task_index already in {pq_file.name}"
            
        # Build the task_index array for the table
        ep_indices = table.column("episode_index").to_pylist()
        task_indices = [ep_to_task.get(ep, 0) for ep in ep_indices]
        
        # Append column
        task_index_array = pa.array(task_indices, type=pa.int64())
        new_table = table.append_column("task_index", task_index_array)
        
        # Drop the original 'task' column if it exists
        if "task" in new_table.schema.names:
            new_table = new_table.drop(["task"])
            
        # Write to a temp file, then replace
        temp_file = pq_file.with_suffix('.parquet.tmp')
        pq.write_table(new_table, temp_file)
        shutil.move(temp_file, pq_file)
        
        return f"Fixed {pq_file.name}"
        
    except Exception as e:
        return f"Error processing {pq_file.name}: {e}"

def fix_parquet_task_index(ds_path):
    ds_path = Path(ds_path)
    parquet_files = list(ds_path.glob("data/**/*.parquet"))
    if not parquet_files:
        logging.warning(f"No parquet files found in {ds_path}")
        return
        
    logging.info(f"Processing {len(parquet_files)} parquet files in {ds_path.name}")
    
    # Check if task_index is missing in the first file
    try:
        table = pq.read_table(parquet_files[0])
        if "task_index" in table.schema.names:
            logging.info(f"task_index already present in {ds_path.name}")
            return
    except Exception as e:
        logging.error(f"Error reading {parquet_files[0]}: {e}")
        return
        
    # Load task mapping from tasks.jsonl
    tasks_jsonl = ds_path / "meta" / "tasks.jsonl"
    task_str_to_idx = {}
    if tasks_jsonl.exists():
        with open(tasks_jsonl, 'r') as f:
            for line in f:
                data = json.loads(line)
                task_str_to_idx[data["task"]] = data["task_index"]
                
    # Build mapping from episodes.jsonl
    episodes_jsonl = ds_path / "meta" / "episodes.jsonl"
    ep_to_task = {}
    if episodes_jsonl.exists():
        with open(episodes_jsonl, 'r') as f:
            for line in f:
                data = json.loads(line)
                ep_idx = data["episode_index"]
                
                task_val = 0
                if "tasks" in data and len(data["tasks"]) > 0:
                    task_val = data["tasks"][0]
                elif "task_index" in data:
                    task_val = data["task_index"]
                    
                if isinstance(task_val, str):
                    task_val = task_str_to_idx.get(task_val, 0)
                    
                ep_to_task[ep_idx] = task_val
    
    # Run in parallel per file
    success_count = 0
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fix_single_parquet, pq_file, ep_to_task): pq_file for pq_file in parquet_files}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if "Fixed" in result:
                success_count += 1
            if "Error" in result:
                logging.error(result)
            
    logging.info(f"Finished fixing {success_count}/{len(parquet_files)} files in {ds_path.name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_parquet_task_index.py <dataset_path>")
        sys.exit(1)
        
    ds_path = sys.argv[1]
    fix_parquet_task_index(ds_path)
