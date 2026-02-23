import sys
import os
from pathlib import Path
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import torch
    print("lerobot import success")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def test_load(dataset_dir):
    path = Path(dataset_dir)
    print(f"\nTesting: {path.name}")
    if not (path / "meta/info.json").exists():
        print("  - Skip: No info.json")
        return False
    
    try:
        # We use root=path.parent and repo_id=path.name if it's a standard structure
        # Or just path if it can handle absolute local paths
        dataset = LeRobotDataset(str(path))
        print(f"  - Load success! Episodes: {dataset.num_episodes}, Frames: {dataset.num_frames}")
        print(f"  - Features: {list(dataset.features.keys())}")
        
        # Try to get first item
        item = dataset[0]
        print(f"  - Data access success! Item keys: {list(item.keys())}")
        return True
    except Exception as e:
        print(f"  - Load failed: {e}")
        return False

# List of some critical ones to test
targets = [
    "/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/openx_lerobot/droid",
    "/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/openx_lerobot/bc_z",
    "/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/agibot/agibot_gripper0_part1",
    "/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/galaxea/galaxea_part1",
    "/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/humanoid_everyday_h1"
]

for t in targets:
    if os.path.isdir(t):
        test_load(t)
    else:
        print(f"\nPath not found: {t}")
