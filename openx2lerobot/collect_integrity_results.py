#!/usr/bin/env python
"""
Collect all per-dataset integrity results and generate README summary.
Run after all submit_integrity_checks.sh jobs complete.
"""
import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("/fsx/ubuntu/taeyoung/workspace/any4lerobot/openx2lerobot/integrity_results")
README_PATH = Path("/fsx/ubuntu/taeyoung/workspace/any4lerobot/openx2lerobot/README.md")

def collect_results():
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(f) as fp:
                results.append(json.load(fp))
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return results

def format_number(n):
    if n is None:
        return "N/A"
    return f"{n:,}"

def build_readme(results):
    ok = [r for r in results if r.get("ok")]
    fail = [r for r in results if not r.get("ok")]

    # Group by category
    def cat(r):
        p = r.get("path", "")
        if "openx_lerobot" in p:
            return "OpenX"
        elif "agibot" in p:
            return "Agibot"
        elif "galaxea" in p:
            return "Galaxea"
        elif "neural_traj" in p:
            return "NeuralTraj"
        elif "humanoid" in p:
            return "Humanoid"
        elif "action_net" in p:
            return "ActionNet"
        return "Other"

    categories = {}
    total_eps = 0
    total_frames = 0
    total_videos = 0

    for r in results:
        c = cat(r)
        if c not in categories:
            categories[c] = []
        categories[c].append(r)
        info = r.get("info", {})
        total_eps += info.get("total_episodes", 0)
        total_frames += info.get("total_frames", 0)
        total_videos += info.get("mp4_count", 0) or 0

    lines = []
    lines.append("# VLA Pretrain Dataset - Conversion Status")
    lines.append("")
    lines.append(f"> Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("## 📊 Overall Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total Datasets | {len(results)} |")
    lines.append(f"| ✅ Passed | {len(ok)} |")
    lines.append(f"| ❌ Failed | {len(fail)} |")
    lines.append(f"| Total Episodes | {format_number(total_eps)} |")
    lines.append(f"| Total Frames | {format_number(total_frames)} |")
    lines.append(f"| Total Videos (.mp4) | {format_number(total_videos)} |")
    lines.append("")

    if fail:
        lines.append("## ⚠️ Failed Datasets")
        lines.append("")
        for r in fail:
            lines.append(f"### ❌ `{r['name']}`")
            for issue in r.get("issues", []):
                lines.append(f"- {issue}")
            lines.append("")

    cat_order = ["OpenX", "Agibot", "Galaxea", "NeuralTraj", "Humanoid", "ActionNet", "Other"]
    for c in cat_order:
        if c not in categories:
            continue
        ds_list = categories[c]
        lines.append(f"## {c} Datasets ({len(ds_list)} datasets)")
        lines.append("")
        lines.append("| Dataset | Status | Episodes | Frames | Parquets | Videos | FPS | Robot |")
        lines.append("|---------|--------|----------|--------|----------|--------|-----|-------|")
        for r in ds_list:
            info = r.get("info", {})
            status = "✅" if r.get("ok") else "❌"
            name = r["name"]
            ep = format_number(info.get("total_episodes"))
            fr = format_number(info.get("total_frames"))
            pq = format_number(info.get("parquet_count"))
            vid = format_number(info.get("mp4_count")) if info.get("mp4_count") else "n/a"
            fps = info.get("fps", "?")
            robot = info.get("robot_type", "?")
            lines.append(f"| `{name}` | {status} | {ep} | {fr} | {pq} | {vid} | {fps} | {robot} |")
        lines.append("")

    lines.append("## Data Format")
    lines.append("")
    lines.append("All datasets are in **LeRobot v2.1** format (`codebase_version: v2.1`):")
    lines.append("")
    lines.append("```")
    lines.append("dataset/")
    lines.append("├── data/")
    lines.append("│   └── chunk-000/")
    lines.append("│       ├── episode_000000.parquet")
    lines.append("│       └── ...")
    lines.append("├── meta/")
    lines.append("│   ├── info.json")
    lines.append("│   ├── stats.json")
    lines.append("│   ├── episodes.jsonl")
    lines.append("│   └── tasks.jsonl")
    lines.append("└── videos/")
    lines.append("    └── <camera_key>/")
    lines.append("        └── chunk-000/")
    lines.append("            └── episode_000000.mp4")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)

if __name__ == "__main__":
    results = collect_results()
    print(f"Found {len(results)} result files.")

    ok = sum(1 for r in results if r.get("ok"))
    fail = sum(1 for r in results if not r.get("ok"))
    print(f"OK: {ok} | FAIL: {fail}")

    if fail > 0:
        print("\nFailed datasets:")
        for r in results:
            if not r.get("ok"):
                print(f"  ❌ {r['name']}: {r.get('issues', [])}")

    readme_content = build_readme(results)
    with open(README_PATH, "w") as f:
        f.write(readme_content)
    print(f"\nREADME updated: {README_PATH}")
