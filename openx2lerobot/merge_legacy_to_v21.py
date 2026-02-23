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

def merge_legacy_to_v21(source_dir, output_dir, dataset_name, fps, robot_type):
    source_path = Path(source_dir)
    output_path = Path(output_dir) / dataset_name
    
    print(f"Starting merge for {dataset_name}...")
    if output_path.exists():
        print(f"Removing existing output dir: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    
    temp_shards_dir = source_path / "_temp_shards"
    job_dirs = sorted(list(temp_shards_dir.glob("job_*")))
    print(f"Found {len(job_dirs)} job directories")
    
    global_episode_idx = 0
    global_frame_idx = 0
    episodes_records = []
    episodes_stats = []
    global_stats_tracker = {}
    features_info = {}
    all_tasks = set()
    data_chunk_size = 1000

    for job_dir in tqdm(job_dirs, desc="Merging jobs"):
        meta_file = job_dir / "metadata.jsonl"
        if not meta_file.exists(): continue
            
        with jsonlines.open(meta_file) as reader:
            for rec in reader:
                local_id = rec["local_id"]
                task_str = rec["task"]
                all_tasks.add(task_str)
                
                dest_chunk = global_episode_idx // data_chunk_size
                dest_data_dir = output_path / f"data/chunk-{dest_chunk:03d}"
                dest_data_dir.mkdir(parents=True, exist_ok=True)
                
                src_pq = job_dir / "data" / f"episode_{local_id:06d}.parquet"
                if not src_pq.exists(): continue
                
                df = pd.read_parquet(src_pq)
                num_frames = len(df)
                
                # Add Standard Indices
                df["episode_index"] = global_episode_idx
                df["frame_index"] = np.arange(num_frames)
                df["timestamp"] = df["frame_index"] / fps
                df["index"] = np.arange(global_frame_idx, global_frame_idx + num_frames)
                
                # Compute Stats
                ep_stats_dict = {}
                for col in df.columns:
                    if col in ["task", "episode_index", "frame_index", "timestamp", "index"]:
                        continue
                    
                    try:
                        data = np.stack(df[col].values).astype(np.float32)
                    except:
                        continue
                    
                    if col not in features_info:
                        features_info[col] = {"dtype": "float32", "shape": [data.shape[1]] if len(data.shape) > 1 else [1]}
                    
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
                    
                    # Global Tracker (Approximate global percentiles aren't mapped here, but min/max/sum/sum_sq are)
                    if col not in global_stats_tracker:
                        global_stats_tracker[col] = {"min": np.min(data, axis=0), "max": np.max(data, axis=0), "sum": np.sum(data, axis=0), "sum_sq": np.sum(data**2, axis=0), "count": num_frames}
                    else:
                        global_stats_tracker[col]["min"] = np.minimum(global_stats_tracker[col]["min"], np.min(data, axis=0))
                        global_stats_tracker[col]["max"] = np.maximum(global_stats_tracker[col]["max"], np.max(data, axis=0))
                        global_stats_tracker[col]["sum"] += np.sum(data, axis=0)
                        global_stats_tracker[col]["sum_sq"] += np.sum(data**2, axis=0)
                        global_stats_tracker[col]["count"] += num_frames

                # Save Fixed Parquet
                dest_pq = dest_data_dir / f"episode_{global_episode_idx:06d}.parquet"
                df.to_parquet(dest_pq)
                
                # Video Handling (Copy)
                for vid_cam_dir in (job_dir / "videos").glob("observation.images.*"):
                    cam_key = vid_cam_dir.name
                    dest_vid_dir = output_path / f"videos/chunk-{dest_chunk:03d}" / cam_key
                    dest_vid_dir.mkdir(parents=True, exist_ok=True)
                    src_vid = vid_cam_dir / f"episode_{local_id:06d}.mp4"
                    if src_vid.exists():
                        shutil.copy(src_vid, dest_vid_dir / f"episode_{global_episode_idx:06d}.mp4")
                    
                    if cam_key not in features_info:
                        features_info[cam_key] = {"dtype": "video", "shape": [None, None, 3], "names": ["height", "width", "rgb"]}

                episodes_records.append({"episode_index": global_episode_idx, "tasks": [task_str], "length": num_frames})
                episodes_stats.append({"episode_index": global_episode_idx, "stats": ep_stats_dict, "length": num_frames})
                global_episode_idx += 1
                global_frame_idx += num_frames

    # Save Meta
    meta_dir = output_path / "meta"
    meta_dir.mkdir(exist_ok=True)
    with jsonlines.open(meta_dir / "episodes.jsonl", mode="w") as writer:
        for r in episodes_records: writer.write(r)
    with jsonlines.open(meta_dir / "episodes_stats.jsonl", mode="w") as writer:
        for r in episodes_stats:
            # Add Dummy Video Stats for episodes_stats.jsonl
            for feat in features_info:
                if features_info[feat]["dtype"] == "video" and feat not in r["stats"]:
                    r["stats"][feat] = {"min": [[[0.0]]*3], "max": [[[1.0]]*3], "mean": [[[0.5]]*3], "std": [[[0.5]]*3], "count": [r["length"]]}
            writer.write(r)
            
    task_list = sorted(list(all_tasks))
    with jsonlines.open(meta_dir / "tasks.jsonl", mode="w") as writer:
        for i, t in enumerate(task_list): writer.write({"task_index": i, "task": t})
            
    # Final Global Stats
    final_global_stats = {}
    for col, trk in global_stats_tracker.items():
        mean = trk["sum"] / trk["count"]
        var = (trk["sum_sq"] / trk["count"]) - (mean**2)
        final_global_stats[col] = {"min": trk["min"].tolist(), "max": trk["max"].tolist(), "mean": mean.tolist(), "std": np.sqrt(np.maximum(var, 0)).tolist(), "count": [trk["count"]]}
    
    for feat in features_info:
        if features_info[feat]["dtype"] == "video":
            final_global_stats[feat] = {"min": [[[0.0]]*3], "max": [[[1.0]]*3], "mean": [[[0.5]]*3], "std": [[[0.5]]*3], "count": [global_frame_idx]}
    
    with open(meta_dir / "stats.json", "w") as f: json.dump(final_global_stats, f, indent=4)

    # Info.json
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
        "total_episodes": global_episode_idx, "total_frames": global_frame_idx, "total_tasks": len(task_list), "fps": fps,
        "splits": {"train": f"0:{global_episode_idx}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": feat_dict,
    }
    with open(meta_dir / "info.json", "w") as f: json.dump(info, f, indent=4)
    print(f"Merge Complete: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--robot-type", type=str, default="unknown")
    args = parser.parse_args()
    merge_legacy_to_v21(args.source_dir, args.output_dir, args.dataset_name, args.fps, args.robot_type)
