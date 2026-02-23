import os
from pathlib import Path

def main():
    base_dirs = [
        "/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset",
        "/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/openx_lerobot"
    ]
    
    datasets = []
    for base in base_dirs:
        base = Path(base)
        if not base.exists():
            continue
        for item in base.iterdir():
            if not item.is_dir():
                continue
            if (item / "meta/info.json").exists():
                datasets.append(item)
            else:
                for sub in item.iterdir():
                    if sub.is_dir() and (sub / "meta/info.json").exists():
                        datasets.append(sub)
    
    # Remove duplicates if any
    datasets = sorted(list(set(datasets)))
    
    output_file = "/fsx/ubuntu/taeyoung/workspace/any4lerobot/openx2lerobot/s5cmd_info_json_tasks.txt"
    commands = []
    
    local_base_path = "/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/"
    s3_base_path = "s3://alinvla/vla_pretrain_dataset/"
    
    for ds_p in datasets:
        info_path = ds_p / "meta/info.json"
        
        # Calculate relative path to construct S3 path
        rel_path = os.path.relpath(info_path, local_base_path)
        s3_path = os.path.join(s3_base_path, rel_path)
        
        # s5cmd cp command
        commands.append(f"cp {info_path} {s3_path}")
        
    with open(output_file, "w") as f:
        f.write("\n".join(commands))
        
    print(f"Generated {len(commands)} upload commands to {output_file}")

if __name__ == "__main__":
    main()
