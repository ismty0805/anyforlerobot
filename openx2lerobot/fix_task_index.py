import json
import logging
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        
    for pq_file in parquet_files:
        try:
            # Read original table
            table = pq.read_table(pq_file)
            
            if "task_index" in table.schema.names:
                continue
                
            # Create task_index column (all zeros for single-task, or map from something else?)
            # Wait, usually episodes.jsonl has the tasks array mapping. 
            # In lerobot v2.1, episodes.jsonl has tasks: [task_index]. 
            # Parquet files might just have "task_index" = 0 for all rows if it's a single task,
            # or we need to extract episode_index and map.
            
            # Let's map episode_index -> task_index from episodes.jsonl
            episodes_jsonl = ds_path / "meta" / "episodes.jsonl"
            ep_to_task = {}
            with open(episodes_jsonl, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    ep_idx = data["episode_index"]
                    # Usually "tasks": [task_idx]
                    if "tasks" in data and len(data["tasks"]) > 0:
                        ep_to_task[ep_idx] = data["tasks"][0]
                    elif "task_index" in data:
                        ep_to_task[ep_idx] = data["task_index"]
                    else:
                        ep_to_task[ep_idx] = 0 # default
            
            # Now build the task_index array for the table
            ep_indices = table.column("episode_index").to_pylist()
            task_indices = [ep_to_task.get(ep, 0) for ep in ep_indices]
            
            # Append column
            task_index_array = pa.array(task_indices, type=pa.int64())
            new_table = table.append_column("task_index", task_index_array)
            
            # Write to a temp file, then replace
            temp_file = pq_file.with_suffix('.parquet.tmp')
            pq.write_table(new_table, temp_file)
            shutil.move(temp_file, pq_file)
            
            logging.info(f"Fixed {pq_file.name}")
            
        except Exception as e:
            logging.error(f"Error processing {pq_file}: {e}")
            
    logging.info(f"Finished fixing {ds_path.name}")

if __name__ == "__main__":
    datasets_to_fix = [
        "action_net", "agibot_dexhand", "agibot_gripper0_part1", "agibot_gripper0_part2",
        "agibot_gripper0_part3", "agibot_gripper0_part4", "galaxea_part1", "galaxea_part2",
        "galaxea_part3", "galaxea_part4", "galaxea_part5", "humanoid_everyday_g1", "humanoid_everyday_h1",
        "bc_z", "fmb_dataset", "fractal20220817_data", "furniture_bench_dataset_converted_externally_to_rlds",
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds"
    ]
    
    base_dirs = [
        Path("/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset"),
        Path("/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/agibot"),
        Path("/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/galaxea"),
        Path("/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/openx_lerobot")
    ]
    
    for ds_name in datasets_to_fix:
        # find path
        ds_path = None
        for base in base_dirs:
            p = base / ds_name
            if p.exists():
                ds_path = p
                break
        
        if ds_path:
            fix_parquet_task_index(ds_path)
        else:
            logging.error(f"Could not find dataset path for {ds_name}")
