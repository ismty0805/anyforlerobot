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
    
    for ds_p in datasets:
        info_path = ds_p / "meta/info.json"
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
            
            updated = False
            
            # 1. total_chunks Calculation
            # total_chunks = ceil(total_episodes / chunks_size)
            total_episodes = info.get("total_episodes", 0)
            chunks_size = info.get("chunks_size", 1000)
            
            if total_episodes > 0:
                calculated_total_chunks = (total_episodes + chunks_size - 1) // chunks_size
                if info.get("total_chunks") != calculated_total_chunks:
                    info["total_chunks"] = calculated_total_chunks
                    updated = True

            # 2. total_videos Calculation
            # total_videos = total_episodes * number_of_video_features
            video_features = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
            calculated_total_videos = total_episodes * len(video_features)
            
            if info.get("total_videos") != calculated_total_videos:
                info["total_videos"] = calculated_total_videos
                updated = True
            
            if updated:
                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=4)
                print(f"  Updated {ds_p.name}: total_chunks={info.get('total_chunks')}, total_videos={info.get('total_videos')}")
                
        except Exception as e:
            print(f"  Error processing {ds_p.name}: {e}")

    print(f"\nDone updating total_chunks and total_videos.")

if __name__ == "__main__":
    main()
