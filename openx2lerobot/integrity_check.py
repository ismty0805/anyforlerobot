#!/usr/bin/env python
"""
Integrity check for all datasets in vla_pretrain_dataset.
Since our data uses v2.1 format (episode_*.parquet) and lerobot v0.3.3 also expects v2.1,
we directly validate:
  1. meta/info.json: valid JSON, has total_episodes, total_frames, fps, features
  2. meta/episodes.jsonl: non-empty, count matches total_episodes in info.json
  3. meta/stats.json: exists and non-empty
  4. meta/tasks.jsonl: exists and count > 0
  5. data/*.parquet: count matches total_episodes
  6. Parquet file: first parquet is readable and has expected feature columns
  7. Videos: if video features in info.json, at least one .mp4 exists
  8. LeRobotDataset load: attempt to load meta only with lerobot
"""
import json
import sys
import os
import traceback
from pathlib import Path
from datetime import datetime

BASE = Path("/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset")

def collect_datasets(base):
    results = []
    for top in sorted(base.iterdir()):
        if not top.is_dir():
            continue
        if (top / "meta/info.json").exists():
            results.append(top)
        else:
            for sub in sorted(top.iterdir()):
                if sub.is_dir() and (sub / "meta/info.json").exists():
                    results.append(sub)
    return results

def check_dataset(ds_path):
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
        data_keys = [k for k, v in features.items() if v.get("dtype") not in ("video",) and k not in ("timestamp","frame_index","episode_index","index","task_index")]
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
        issues.append(f"info.json error: {e}")
        return False, info_dict, issues

    # 2. episodes.jsonl
    ep_path = ds_path / "meta/episodes.jsonl"
    if not ep_path.exists():
        issues.append("episodes.jsonl MISSING")
    else:
        ep_count = sum(1 for _ in open(ep_path))
        if ep_count == 0:
            issues.append("episodes.jsonl EMPTY")
        elif ep_count != total_ep:
            issues.append(f"episodes.jsonl count={ep_count} != info.json total_episodes={total_ep}")
        info_dict["episodes_jsonl_count"] = ep_count

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
        except Exception as e:
            issues.append(f"stats.json read error: {e}")

    # 4. tasks.jsonl
    tasks_path = ds_path / "meta/tasks.jsonl"
    if not tasks_path.exists():
        issues.append("tasks.jsonl MISSING")
    else:
        task_count = sum(1 for _ in open(tasks_path))
        if task_count == 0:
            issues.append("tasks.jsonl EMPTY")
        info_dict["tasks_count"] = task_count

    # 5. Parquet count
    pq_files = sorted(ds_path.rglob("data/**/*.parquet"))
    pq_count = len(pq_files)
    info_dict["parquet_count"] = pq_count
    if pq_count == 0:
        issues.append("No parquet files in data/")
    elif pq_count != total_ep:
        issues.append(f"parquet count={pq_count} != total_episodes={total_ep}")

    # 6. Read first parquet
    if pq_count > 0:
        try:
            import pandas as pd
            df = pd.read_parquet(pq_files[0])
            cols = df.columns.tolist()
            info_dict["parquet_columns"] = cols
            # Check that key features are present
            for key in data_keys:
                if key not in cols:
                    issues.append(f"Feature '{key}' missing from parquet columns")
        except Exception as e:
            issues.append(f"Parquet read error: {e}")

    # 7. Videos
    if video_keys:
        mp4_count = sum(1 for _ in ds_path.rglob("videos/**/*.mp4"))
        info_dict["mp4_count"] = mp4_count
        if mp4_count == 0:
            issues.append(f"No .mp4 files but {len(video_keys)} video keys in info.json")

    # 8. LeRobotDataset load (metadata only approach)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        meta = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
        meta.repo_id = ds_path.name
        meta.root = ds_path
        meta.revision = "v2.1"
        meta.load_metadata()
        info_dict["lerobot_meta_load"] = "OK"
        info_dict["lerobot_total_episodes"] = meta.total_episodes
    except Exception as e:
        err_str = str(e)[:150]
        # Only flag as issue if it's not just a HF Hub check
        if "huggingface.co" not in err_str and "offline" not in err_str.lower():
            issues.append(f"LeRobotDataset meta load error: {err_str}")
        info_dict["lerobot_meta_load"] = f"FAIL: {err_str}"

    ok = len(issues) == 0
    return ok, info_dict, issues


if __name__ == "__main__":
    datasets = collect_datasets(BASE)
    print(f"Found {len(datasets)} datasets to check.\n")
    print(f"Check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = []
    ok_count = 0
    warn_count = 0
    fail_count = 0

    for ds_path in datasets:
        ok, info_dict, issues = check_dataset(ds_path)
        rel_name = f"{ds_path.parent.name}/{ds_path.name}"

        status = "✅ OK" if ok else "❌ FAIL"
        if ok:
            ok_count += 1
        else:
            fail_count += 1

        print(f"{status} | {rel_name}")
        print(f"   codebase={info_dict.get('codebase_version','?')} | fps={info_dict.get('fps','?')} | robot={info_dict.get('robot_type','?')}")
        print(f"   Episodes: {info_dict.get('total_episodes',0)} | Frames: {info_dict.get('total_frames',0)} | Parquets: {info_dict.get('parquet_count',0)} | Videos: {info_dict.get('mp4_count','n/a')}")
        print(f"   Data features: {info_dict.get('data_features', [])} | Video keys: {info_dict.get('video_keys', [])}")
        print(f"   Tasks: {info_dict.get('tasks_count', 0)} | LeRobot meta: {info_dict.get('lerobot_meta_load','?')}")
        if issues:
            for issue in issues:
                print(f"   ⚠️  {issue}")
        print()

        results.append({
            "name": rel_name,
            "status": status,
            "ok": ok,
            "info": info_dict,
            "issues": issues,
        })

    with open("integrity_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== SUMMARY ===")
    print(f"Total: {len(results)} | ✅ OK: {ok_count} | ❌ FAIL: {fail_count}")
    print("Results saved to integrity_results.json")
