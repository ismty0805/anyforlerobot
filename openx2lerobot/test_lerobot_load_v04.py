import sys
import os
from pathlib import Path
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    print("lerobot import success (v0.4.x structure)")
except ImportError:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        print("lerobot import success (common structure)")
    except ImportError as e:
        print(f"ImportError: {e}")
        # List what is available in lerobot
        import lerobot
        print(f"lerobot members: {dir(lerobot)}")
        sys.exit(1)

def test_load(dataset_dir):
    path = Path(dataset_dir)
    print(f"\nTesting: {path.name}")
    if not (path / "meta/info.json").exists():
        print("  - Skip: No info.json")
        return False
    
    try:
        # v0.4.x LeRobotDataset often takes repo_id or local_path
        dataset = LeRobotDataset(dataset_id=str(path))
        print(f"  - Load success! Episodes: {dataset.num_episodes}, Frames: {dataset.num_frames}")
        # Try to get first item
        item = dataset[0]
        print(f"  - Data access success! Item keys: {list(item.keys())}")
        return True
    except Exception as e:
        print(f"  - Load failed: {e}")
        return False

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
