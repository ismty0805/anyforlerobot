#!/usr/bin/env python
"""
Fix info.json for datasets where total_episodes/total_frames don't match actual data.
Recalculates from episodes.jsonl and parquet files.
"""
import json
from pathlib import Path

BASE = Path("/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/openx_lerobot")
TARGETS = ["bridge_orig", "droid", "kuka", "language_table"]

def fix_info(ds_path: Path):
    info_path = ds_path / "meta/info.json"
    ep_path   = ds_path / "meta/episodes.jsonl"

    with open(info_path) as f:
        info = json.load(f)

    # Count actual episodes from episodes.jsonl
    episodes = []
    with open(ep_path) as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    actual_ep = len(episodes)

    # Count total frames from episodes.jsonl length fields
    actual_frames = sum(ep.get("length", 0) for ep in episodes)

    # Count parquet files
    pq_count = len(list(ds_path.rglob("data/**/*.parquet")))

    # Count chunks
    chunk_size = info.get("chunks_size", 1000)
    total_chunks = (actual_ep + chunk_size - 1) // chunk_size

    # Count videos per episode
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    total_videos = actual_ep * len(video_keys)

    print(f"\n{'='*60}")
    print(f"Dataset: {ds_path.name}")
    print(f"  info.json:     total_episodes={info.get('total_episodes')}, total_frames={info.get('total_frames')}")
    print(f"  episodes.jsonl actual:  episodes={actual_ep}, frames={actual_frames}")
    print(f"  parquet count:          {pq_count}")
    print(f"  video_keys:             {video_keys}")

    # Update
    info["total_episodes"] = actual_ep
    info["total_frames"]   = actual_frames
    info["total_chunks"]   = total_chunks
    info["total_videos"]   = total_videos

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"  ✅ Fixed: total_episodes={actual_ep}, total_frames={actual_frames}, total_chunks={total_chunks}, total_videos={total_videos}")

if __name__ == "__main__":
    for name in TARGETS:
        ds_path = BASE / name
        if not (ds_path / "meta/info.json").exists():
            print(f"⚠️  Skipping {name}: no info.json")
            continue
        fix_info(ds_path)
    print("\nDone fixing info.json for all targets.")
