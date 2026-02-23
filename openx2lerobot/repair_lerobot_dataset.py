#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
import pandas as pd
import jsonlines
import numpy as np

def repair_dataset(dataset_dir, fps, robot_type):
    dataset_path = Path(dataset_dir)
    print(f"Repairing dataset at: {dataset_path}")
    
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(exist_ok=True)
    
    # 1. Load existing episodes.jsonl
    episodes_old_file = meta_dir / "episodes.jsonl"
    if not episodes_old_file.exists():
        print(f"Error: {episodes_old_file} not found. Cannot repair.")
        return

    episodes_records = []
    with jsonlines.open(episodes_old_file) as reader:
        for rec in reader:
            # Fix task -> tasks [list]
            if "task" in rec and "tasks" not in rec:
                rec["tasks"] = [rec["task"]]
                del rec["task"]
            episodes_records.append(rec)
            
    print(f"Repairing {len(episodes_records)} episodes...")
    
    global_frame_idx = 0
    episodes_stats = []
    global_stats_tracker = {}
    features_info = {}
    all_tasks = set()
    
    # 2. Process Episodes
    for rec in tqdm(episodes_records, desc="Processing episodes"):
        ep_idx = rec["episode_index"]
        chunk_idx = ep_idx // 1000
        pq_path = dataset_path / f"data/chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
        
        if not pq_path.exists():
            # Try flat structure if chunked doesn't exist
            pq_path = dataset_path / "data" / f"episode_{ep_idx:06d}.parquet"
            if not pq_path.exists():
                # print(f"Warning: Missing {pq_path}")
                continue
            
        try:
            df = pd.read_parquet(pq_path)
        except Exception as e:
            print(f"Error reading {pq_path}: {e}")
            continue

        num_frames = len(df)
        
        # Normalize tasks
        task_str = rec["tasks"][0]
        all_tasks.add(task_str)
        
        # Add Standard Indices if missing or to ensure correctness
        df["episode_index"] = ep_idx
        df["frame_index"] = np.arange(num_frames)
        df["timestamp"] = (df["frame_index"] / fps).astype(np.float32)
        df["index"] = np.arange(global_frame_idx, global_frame_idx + num_frames)
        
        # Stats computation
        ep_stats_dict = {}
        for col in df.columns:
            if col in ["task", "tasks", "episode_index", "frame_index", "timestamp", "index"]:
                continue
            
            try:
                # Convert to float32 for computation
                data = np.stack(df[col].values).astype(np.float32)
            except:
                continue
            
            if col not in features_info:
                features_info[col] = {
                    "dtype": "float32",
                    "shape": [data.shape[1]] if len(data.shape) > 1 else [1]
                }
            
            # Episode Stats (Matching DROID style)
            ep_stats_dict[col] = {
                "min": np.min(data, axis=0).tolist(),
                "max": np.max(data, axis=0).tolist(),
                "mean": np.mean(data, axis=0).tolist(),
                "std": np.std(data, axis=0).tolist(),
                "count": [num_frames],
                "q01": np.percentile(data, 1, axis=0).tolist(),
                "q10": np.percentile(data, 10, axis=0).tolist(),
                "q50": np.percentile(data, 50, axis=0).tolist(),
                "q90": np.percentile(data, 90, axis=0).tolist(),
                "q99": np.percentile(data, 99, axis=0).tolist(),
            }
            
            # Update global tracker
            if col not in global_stats_tracker:
                global_stats_tracker[col] = {"min": np.min(data, axis=0), "max": np.max(data, axis=0), "sum": np.sum(data, axis=0), "sum_sq": np.sum(data**2, axis=0), "count": num_frames}
            else:
                global_stats_tracker[col]["min"] = np.minimum(global_stats_tracker[col]["min"], np.min(data, axis=0))
                global_stats_tracker[col]["max"] = np.maximum(global_stats_tracker[col]["max"], np.max(data, axis=0))
                global_stats_tracker[col]["sum"] += np.sum(data, axis=0)
                global_stats_tracker[col]["sum_sq"] += np.sum(data**2, axis=0)
                global_stats_tracker[col]["count"] += num_frames

        # Overwrite Parquet with new columns
        df.to_parquet(pq_path)
        
        # Detect video cameras from filesystem if not in stats
        vid_base = dataset_path / f"videos/chunk-{chunk_idx:03d}"
        if vid_base.exists():
            for cam_dir in vid_base.glob("observation.images.*"):
                cam_key = cam_dir.name
                if cam_key not in features_info:
                    features_info[cam_key] = {"dtype": "video", "shape": [None, None, 3]}
                if cam_key not in ep_stats_dict:
                    ep_stats_dict[cam_key] = {"min": [[[0.0]]*3], "max": [[[1.0]]*3], "mean": [[[0.5]]*3], "std": [[[0.5]]*3], "count": [num_frames]}

        episodes_stats.append({"episode_index": ep_idx, "stats": ep_stats_dict, "length": num_frames})
        global_frame_idx += num_frames

    # 3. Save Fixed Meta Files
    with jsonlines.open(meta_dir / "episodes.jsonl", mode="w") as writer:
        for rec in episodes_records: writer.write(rec)
            
    with jsonlines.open(meta_dir / "episodes_stats.jsonl", mode="w") as writer:
        for rec in episodes_stats: writer.write(rec)
            
    task_list = sorted(list(all_tasks))
    with jsonlines.open(meta_dir / "tasks.jsonl", mode="w") as writer:
        for i, t in enumerate(task_list): writer.write({"task_index": i, "task": t})
            
    final_global_stats = {}
    for col, trk in global_stats_tracker.items():
        total_count = trk["count"]
        mean = trk["sum"] / total_count
        var = (trk["sum_sq"] / total_count) - (mean**2)
        final_global_stats[col] = {"min": trk["min"].tolist(), "max": trk["max"].tolist(), "mean": mean.tolist(), "std": np.sqrt(np.maximum(var, 0)).tolist(), "count": [total_count]}
    
    for feat in features_info:
        if features_info[feat]["dtype"] == "video":
            final_global_stats[feat] = {"min": [[[0.0]]*3], "max": [[[1.0]]*3], "mean": [[[0.5]]*3], "std": [[[0.5]]*3], "count": [global_frame_idx]}
            
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(final_global_stats, f, indent=4)
        
    # Re-generate info.json
    feat_dict = {}
    for feat, info in features_info.items():
        if info["dtype"] == "video":
            feat_dict[feat] = {"dtype": "video", "shape": info["shape"], "names": ["height", "width", "rgb"], "info": {"video.codec": "h264", "video.pix_fmt": "yuv420p", "video.fps": fps, "video.channels": 3}}
        else:
            dim = info["shape"][0]
            motors = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"] if dim == 7 else ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"] if dim == 8 else [f"m{i}" for i in range(dim)]
            feat_dict[feat] = {"dtype": info["dtype"], "shape": info["shape"], "names": {"motors": motors} if dim > 1 else None}

    for idxf in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
         feat_dict[idxf] = {"dtype": "int64" if "index" in idxf else "float32", "shape": [1], "names": None}

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": len(episodes_records),
        "total_frames": global_frame_idx,
        "total_tasks": len(task_list),
        "fps": fps,
        "splits": {"train": f"0:{len(episodes_records)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": feat_dict,
    }
    with open(meta_dir / "info.json", "w") as f: json.dump(info, f, indent=4)
    print(f"Repair Complete for {dataset_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--robot-type", type=str, default="unknown")
    args = parser.parse_args()
    repair_dataset(args.dataset_dir, args.fps, args.robot_type)
