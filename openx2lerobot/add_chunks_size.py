import json
from pathlib import Path

def collect_datasets(base_dirs):
    results = []
    for base in base_dirs:
        base = Path(base)
        if not base.exists():
            continue
        for item in base.iterdir():
            if not item.is_dir():
                continue
            if (item / "meta/info.json").exists():
                results.append(item)
            else:
                # Check subdirectories
                for sub in item.iterdir():
                    if sub.is_dir() and (sub / "meta/info.json").exists():
                        results.append(sub)
    return sorted(list(set(results)))

def main():
    base_dirs = [
        "/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset",
        "/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/openx_lerobot"
    ]
    
    datasets = collect_datasets(base_dirs)
    print(f"Found {len(datasets)} datasets.")
    
    updated_count = 0
    for ds_p in datasets:
        info_path = ds_p / "meta/info.json"
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
            
            if "chunks_size" not in info:
                # Default to 1000 as used in other scripts
                info["chunks_size"] = 1000
                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=4)
                print(f"  Added chunks_size: 1000 to {ds_p.name}")
                updated_count += 1
        except Exception as e:
            print(f"  Error processing {ds_p.name}: {e}")

    print(f"\nDone. Updated {updated_count} datasets.")

if __name__ == "__main__":
    main()
