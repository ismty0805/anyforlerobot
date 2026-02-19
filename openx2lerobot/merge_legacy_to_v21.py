#!/usr/bin/env python
import argparse
import shutil
import json
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
import pandas as pd
import jsonlines
import numpy as np

def merge_legacy_to_v21(source_dir, output_dir, dataset_name, fps):
    source_path = Path(source_dir)
    output_path = Path(output_dir) / dataset_name
    
    if output_path.exists():
        print(f"Removing existing output dir: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    
    # 1. Collect all jobs
    job_dirs = sorted(list((source_path / "_temp_shards").glob("job_*")))
    print(f"Found {len(job_dirs)} job directories")
    
    global_episode_idx = 0
    episodes_records = []
    
    # Track stats for info.json and episodes_stats.jsonl
    # (Simplified stats for now)
    
    data_chunk_size = 1000
    
    for job_dir in tqdm(job_dirs, desc="Merging jobs"):
        meta_file = job_dir / "metadata.jsonl"
        if not meta_file.exists():
            continue
            
        with jsonlines.open(meta_file) as reader:
            for rec in reader:
                local_id = rec["local_id"]
                
                # Move Parquet
                dest_chunk = global_episode_idx // data_chunk_size
                dest_data_dir = output_path / f"data/chunk-{dest_chunk:03d}"
                dest_data_dir.mkdir(parents=True, exist_ok=True)
                
                src_pq = job_dir / "data" / f"episode_{local_id:06d}.parquet"
                if src_pq.exists():
                    shutil.move(src_pq, dest_data_dir / f"episode_{global_episode_idx:06d}.parquet")
                
                # Move Videos
                for vid_cam_dir in (job_dir / "videos").glob("observation.images.*"):
                    cam_key = vid_cam_dir.name # e.g. observation.images.primary
                    dest_vid_dir = output_path / f"videos/chunk-{dest_chunk:03d}" / cam_key
                    dest_vid_dir.mkdir(parents=True, exist_ok=True)
                    
                    src_vid = vid_cam_dir / f"episode_{local_id:06d}.mp4"
                    if src_vid.exists():
                        shutil.move(src_vid, dest_vid_dir / f"episode_{global_episode_idx:06d}.mp4")
                
                # Update record
                ep_rec = {
                    "episode_index": global_episode_idx,
                    "length": rec["length"],
                    "task": rec["task"],
                }
                episodes_records.append(ep_rec)
                global_episode_idx += 1

    print(f"Merged {global_episode_idx} episodes.")
    
    # 2. Create meta directory
    meta_dir = output_path / "meta"
    meta_dir.mkdir(exist_ok=True)
    
    # Save episodes.jsonl
    with jsonlines.open(meta_dir / "episodes.jsonl", mode="w") as writer:
        for rec in episodes_records:
            writer.write(rec)
            
    # Save tasks.jsonl (minimal)
    tasks = sorted(list(set(r["task"] for r in episodes_records)))
    with jsonlines.open(meta_dir / "tasks.jsonl", mode="w") as writer:
        for i, t in enumerate(tasks):
            writer.write({"task_index": i, "task": t})
            
    # Save info.json (Legacy v2.1 format)
    # We'll try to guess dimensions from the first parquet found
    first_pq = list(output_path.glob("data/chunk-000/*.parquet"))[0]
    table = pq.read_table(first_pq)
    
    info = {
        "codebase_version": "v2.1",
        "fps": fps,
        "video_fps": fps,
        "total_episodes": global_episode_idx,
        "total_frames": sum(r["length"] for r in episodes_records),
    }
    
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
        
    print(f"Successfully finalized legacy dataset at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    
    merge_legacy_to_v21(args.source_dir, args.output_dir, args.dataset_name, args.fps)
