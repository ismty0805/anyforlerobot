import sys
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

datasets_to_check = [
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

for ds_name in datasets_to_check:
    ds_path = None
    for base in base_dirs:
        p = base / ds_name
        if p.exists():
            ds_path = p
            break
            
    if ds_path:
        first_parquet = next(ds_path.glob("data/**/*.parquet"), None)
        if first_parquet:
            try:
                table = pq.read_table(first_parquet)
                schema_names = table.schema.names
                has_task_index = "task_index" in schema_names
                has_task = "task" in schema_names
                
                status_parts = []
                if has_task_index:
                    task_indices = table.column("task_index").unique().to_pylist()
                    status_parts.append(f"✅ task_index={task_indices}")
                else:
                    status_parts.append("❌ NO task_index")
                    
                if has_task:
                    status_parts.append("⚠️ STILL HAS task (string)")
                    
                print(f"{ds_name:55} : {' | '.join(status_parts)}")
                
            except Exception as e:
                print(f"{ds_name:55} : ❌ Error reading parquet - {e}")
        else:
            print(f"{ds_name:55} : ❌ No parquet files")
