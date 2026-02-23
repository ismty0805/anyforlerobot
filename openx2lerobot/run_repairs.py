import os
import subprocess
from pathlib import Path

# Mapping of dataset substrings to FPS
FPS_MAPPING = {
    "humanoid_everyday": 30,
    "action_net": 30,
    "agibot": 30,
    "galaxea": 15,
    "neural_robocurate": 16,
    "bc_z": 10,
    "fmb": 10,
    "fractal": 3,
    "furniture": 10,
    "iamlab": 20,
}

BASE_PATH = Path("/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset")

def run_repair(target_dir):
    target_path = Path(target_dir)
    if not (target_path / "meta/episodes.jsonl").exists():
        return
    
    # Check if already repaired (optional, but let's just do it)
    if (target_path / "meta/episodes_stats.jsonl").exists() and (target_path / "meta/info.json").exists():
        # Check if tasks is a list
        with open(target_path / "meta/episodes.jsonl", "r") as f:
            line = f.readline()
            if '"tasks": [' in line:
                print(f"Skipping {target_path.name} - already repaired.")
                return

    # Determine FPS
    fps = 10 # default
    name = target_path.name.lower()
    for key, val in FPS_MAPPING.items():
        if key in name:
            fps = val
            break
    
    print(f"Repairing {target_path} at {fps} FPS...")
    cmd = ["python", "repair_lerobot_dataset.py", "--dataset-dir", str(target_path), "--fps", str(fps)]
    subprocess.run(cmd)

def main():
    # 1. Top level datasets
    for d in BASE_PATH.iterdir():
        if d.is_dir():
            if d.name == "openx_lerobot":
                continue
            if (d / "meta/episodes.jsonl").exists():
                run_repair(d)
            else:
                # Check one level deeper
                for sub in d.iterdir():
                    if sub.is_dir() and (sub / "meta/episodes.jsonl").exists():
                        run_repair(sub)

if __name__ == "__main__":
    main()
