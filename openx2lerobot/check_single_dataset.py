#!/usr/bin/env python
"""
Single dataset integrity checker.
Usage: python check_single_dataset.py <dataset_path> <output_json>
"""
import sys
import json
import os
from pathlib import Path
from datetime import datetime


def check_dataset(ds_path: Path) -> dict:
    issues = []
    info_dict = {}

    # 1. info.json
    info_path = ds_path / "meta/info.json"
    try:
        with open(info_path) as f:
            info = json.load(f)
        total_ep = info.get("total_episodes", 0)
        total_frames = info.get("total_frames", 0)
        fps = info.get("fps", 0)
        features = info.get("features", {})
        robot_type = info.get("robot_type", "unknown")
        codebase_version = info.get("codebase_version", "?")
        video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
        data_keys = [k for k, v in features.items() if v.get("dtype") not in ("video",)
                     and k not in ("timestamp", "frame_index", "episode_index", "index", "task_index")]
        info_dict = {
            "total_episodes": total_ep,
            "total_frames": total_frames,
            "fps": fps,
            "robot_type": robot_type,
            "codebase_version": codebase_version,
            "data_features": data_keys,
            "video_keys": video_keys,
        }
        if total_ep == 0:
            issues.append("total_episodes=0 in info.json")
        if total_frames == 0:
            issues.append("total_frames=0 in info.json")
        if not features:
            issues.append("No features in info.json")
    except Exception as e:
        return {"name": ds_path.name, "ok": False, "issues": [f"info.json error: {e}"], "info": {}}

    # 2. episodes.jsonl
    ep_path = ds_path / "meta/episodes.jsonl"
    if not ep_path.exists():
        issues.append("episodes.jsonl MISSING")
    else:
        ep_count = sum(1 for _ in open(ep_path))
        info_dict["episodes_jsonl_count"] = ep_count
        if ep_count == 0:
            issues.append("episodes.jsonl EMPTY")
        elif ep_count != total_ep:
            issues.append(f"episodes.jsonl count={ep_count} != total_episodes={total_ep}")

    # 3. stats.json
    stats_path = ds_path / "meta/stats.json"
    if not stats_path.exists():
        issues.append("stats.json MISSING")
    else:
        try:
            with open(stats_path) as f:
                stats = json.load(f)
            if not stats:
                issues.append("stats.json EMPTY")
            else:
                info_dict["stats_keys"] = list(stats.keys())[:5]
        except Exception as e:
            issues.append(f"stats.json error: {e}")

    # 4. tasks.jsonl
    tasks_path = ds_path / "meta/tasks.jsonl"
    if not tasks_path.exists():
        issues.append("tasks.jsonl MISSING")
    else:
        task_count = sum(1 for _ in open(tasks_path))
        info_dict["tasks_count"] = task_count
        if task_count == 0:
            issues.append("tasks.jsonl EMPTY")

    # 5. Parquet count
    pq_files = sorted(ds_path.rglob("data/**/*.parquet"))
    pq_count = len(pq_files)
    info_dict["parquet_count"] = pq_count
    if pq_count == 0:
        issues.append("No parquet files in data/")
    elif pq_count != total_ep:
        issues.append(f"parquet count={pq_count} != total_episodes={total_ep}")

    # 6. Read first parquet - verify columns
    if pq_count > 0:
        try:
            import pandas as pd
            df = pd.read_parquet(pq_files[0])
            cols = df.columns.tolist()
            info_dict["parquet_columns"] = cols
            info_dict["first_parquet_rows"] = len(df)
            for key in data_keys:
                if key not in cols:
                    issues.append(f"Feature '{key}' missing from parquet")
        except Exception as e:
            issues.append(f"Parquet read error: {e}")

    # 7. Videos
    if video_keys:
        mp4_count = sum(1 for _ in ds_path.rglob("videos/**/*.mp4"))
        info_dict["mp4_count"] = mp4_count
        expected_mp4 = total_ep * len(video_keys)
        if mp4_count == 0:
            issues.append(f"No .mp4 files but {len(video_keys)} video keys in info.json")
        elif mp4_count != expected_mp4:
            issues.append(f"mp4 count={mp4_count} != expected {expected_mp4} ({total_ep} eps × {len(video_keys)} cams)")

    ok = len(issues) == 0
    return {
        "name": ds_path.name,
        "path": str(ds_path),
        "ok": ok,
        "issues": issues,
        "info": info_dict,
        "checked_at": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    ds_path = Path(sys.argv[1])
    output_json = sys.argv[2]

    result = check_dataset(ds_path)
    status = "✅ OK" if result["ok"] else "❌ FAIL"
    print(f"{status} | {ds_path.name}")
    if result["issues"]:
        for issue in result["issues"]:
            print(f"   ⚠️  {issue}")
    else:
        info = result["info"]
        print(f"   Episodes: {info.get('total_episodes')}, Frames: {info.get('total_frames')}, "
              f"Parquets: {info.get('parquet_count')}, Videos: {info.get('mp4_count', 'n/a')}")

    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
