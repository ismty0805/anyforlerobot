#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""
Optimized for High-Core CPU Systems (64+ Cores).
For all datasets in the RLDS format (OpenX).
"""

import argparse
import re
import shutil
import json
import glob
from functools import partial
from pathlib import Path
import os

# TensorFlow GPU 메모리 점유 방지 (CPU 전용 처리 시 충돌 방지)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import pytorch3d.transforms as pt
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from oxe_utils.configs import OXE_DATASET_CONFIGS, ActionEncoding, StateEncoding
from oxe_utils.transforms import OXE_STANDARDIZATION_TRANSFORMS

# -----------------------------------------------------------------------------
# 1. Rotation Transform Helper (No changes, needed for geometry)
# -----------------------------------------------------------------------------
class RotationTransform:
    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(self, from_rep="axis_angle", to_rep="rotation_6d", from_convention=None, to_convention=None):
        if from_rep.startswith("euler_angles") and from_convention is None:
            try:
                from_convention = from_rep.split("_")[-1]
                from_rep = "euler_angles"
            except IndexError:
                pass
        
        if to_rep.startswith("euler_angles") and to_convention is None:
            try:
                to_convention = to_rep.split("_")[-1]
                to_rep = "euler_angles"
            except IndexError:
                pass

        if from_convention is not None:
             from_convention = from_convention.upper()
        if to_convention is not None:
             to_convention = to_convention.upper()

        self.from_rep = from_rep
        self.to_rep = to_rep
        self.from_convention = from_convention
        self.to_convention = to_convention

        forward_funcs = list()
        
        if from_rep != "matrix":
            if from_rep == "euler_angles":
                f1 = getattr(pt, f"{from_rep}_to_matrix")
                funcs = [partial(f1, convention=from_convention), getattr(pt, f"matrix_to_{from_rep}")]
                funcs[1] = partial(funcs[1], convention=from_convention)
            else:
                funcs = [getattr(pt, f"{from_rep}_to_matrix"), getattr(pt, f"matrix_to_{from_rep}")]
            forward_funcs.append(funcs[0])

        if to_rep != "matrix":
            if to_rep == "euler_angles":
                f1 = getattr(pt, f"matrix_to_{to_rep}")
                funcs = [partial(f1, convention=to_convention), getattr(pt, f"{to_rep}_to_matrix")]
                funcs[1] = partial(funcs[1], convention=to_convention)
            else:
                funcs = [getattr(pt, f"matrix_to_{to_rep}"), getattr(pt, f"{to_rep}_to_matrix")]
            forward_funcs.append(funcs[0])

        self.forward_funcs = forward_funcs

    @staticmethod
    def _apply_funcs(x: torch.Tensor, funcs: list) -> torch.Tensor:
        for func in funcs:
            x = func(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return self._apply_funcs(x, self.forward_funcs)

# -----------------------------------------------------------------------------
# 2. Logic for Action/State Handling
# -----------------------------------------------------------------------------
def validate_action_config(dataset_name):
    if dataset_name not in OXE_DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not found in OXE configs.")
    config = OXE_DATASET_CONFIGS[dataset_name]
    state_obs_keys = config["state_obs_keys"]
    action_encoding = config["action_encoding"]

    if action_encoding == ActionEncoding.JOINT_POS:
        for i in range(min(7, len(state_obs_keys))):
             if state_obs_keys[i] is None:
                 raise ValueError(f"Missing joint state key at index {i} for JOINT_POS action.")
    elif action_encoding == ActionEncoding.EEF_POS:
        for i in range(min(6, len(state_obs_keys))):
            if state_obs_keys[i] is None:
                raise ValueError(f"Missing EEF state key at index {i} for EEF_POS action.")
    return True

def compute_absolute_action(state, action, dataset_name):
    config = OXE_DATASET_CONFIGS[dataset_name]
    action_encoding = config["action_encoding"]
    abs_action = np.zeros_like(action)

    if action_encoding == ActionEncoding.JOINT_POS:
        n_dims = len(action)
        abs_action = state[:n_dims] + action
        
    elif action_encoding == ActionEncoding.EEF_POS:
        state_encoding = config["state_encoding"]
        abs_action[:3] = state[:3] + action[:3]

        if state_encoding == StateEncoding.POS_EULER:
            state_rot = torch.from_numpy(state[3:6]).float().unsqueeze(0)
            action_rot = torch.from_numpy(action[3:6]).float().unsqueeze(0)
            
            tf_to_mat = RotationTransform(from_rep="euler_angles_XYZ", to_rep="matrix")
            tf_to_euler = RotationTransform(from_rep="matrix", to_rep="euler_angles_XYZ")
            
            mat_state = tf_to_mat.forward(state_rot)
            mat_action = tf_to_mat.forward(action_rot)
            
            mat_abs = torch.bmm(mat_action, mat_state)
            abs_rot = tf_to_euler.forward(mat_abs).squeeze(0).numpy()
            
            abs_action[3:6] = abs_rot
            if len(state) == 8 and len(action) == 7:
                 abs_action[6] = state[7] + action[6]
            else:
                 abs_action[-1] = state[-1] + action[-1]

        elif state_encoding == StateEncoding.POS_QUAT:
            s_q = state[3:7] 
            a_q = action[3:7] 
            
            state_rot = torch.tensor([[s_q[3], s_q[0], s_q[1], s_q[2]]], dtype=torch.float32)
            action_rot = torch.tensor([[a_q[3], a_q[0], a_q[1], a_q[2]]], dtype=torch.float32)

            tf_to_mat = RotationTransform(from_rep="quaternion", to_rep="matrix")
            tf_to_quat = RotationTransform(from_rep="matrix", to_rep="quaternion")

            mat_state = tf_to_mat.forward(state_rot)
            mat_action = tf_to_mat.forward(action_rot)
            
            mat_abs = torch.bmm(mat_action, mat_state)
            abs_rot_wxyz = tf_to_quat.forward(mat_abs).squeeze(0).numpy()
            abs_action[3:7] = np.array([abs_rot_wxyz[1], abs_rot_wxyz[2], abs_rot_wxyz[3], abs_rot_wxyz[0]])
            
            if len(action) > 7:
                abs_action[7] = state[7] + action[7]

    return abs_action

# -----------------------------------------------------------------------------
# 3. TF Dataset Transformation (Executed in Parallel via AUTOTUNE)
# -----------------------------------------------------------------------------
def transform_raw_dataset(episode, dataset_name):
    # 1. 에피소드 전체를 하나의 텐서로 묶습니다 (Batching)
    traj = episode["steps"]
    traj = next(iter(traj.batch(50000))) 

    # --- [핵심 수정] Standardization Transform 적용 ---
    # 원본 데이터의 키(예: 'cartesian_position')를 
    # LeRobot이 기대하는 키(예: 'EEF_state')로 변환해주는 필수 과정입니다.
    if dataset_name in OXE_STANDARDIZATION_TRANSFORMS:
        traj = OXE_STANDARDIZATION_TRANSFORMS[dataset_name](traj)
    # ----------------------------------------------------

    if dataset_name in OXE_DATASET_CONFIGS:
        state_obs_keys = OXE_DATASET_CONFIGS[dataset_name]["state_obs_keys"]
    else:
        state_obs_keys = [None for _ in range(8)]

    # TF 로직: Proprioception 데이터 합치기
    proprio_list = []
    for key in state_obs_keys:
        if key is None:
            # Padding
            pad = tf.zeros((tf.shape(traj["action"])[0], 1), dtype=tf.float32)
            proprio_list.append(pad)
        else:
            # 여기서 이제 변환된 'EEF_state' 키를 찾을 수 있게 됩니다.
            proprio_list.append(tf.cast(traj["observation"][key], tf.float32))
            
    proprio = tf.concat(proprio_list, axis=1)

    return {
        "observation": traj["observation"],
        "action": tf.cast(traj["action"], tf.float32),
        "task": traj["language_instruction"],
        "proprio": proprio,
        "is_first": traj["is_first"],
        "is_last": traj["is_last"],
        "is_terminal": traj["is_terminal"],
    }

def generate_features_from_raw(builder: tfds.core.DatasetBuilder, use_videos: bool = True):
    dataset_name = Path(builder.data_dir).parent.name

    # --- [복구된 로직] 데이터셋 설정에 따라 State 차원 및 이름 결정 ---
    state_names = [f"motor_{i}" for i in range(8)]
    if dataset_name in OXE_DATASET_CONFIGS:
        state_encoding = OXE_DATASET_CONFIGS[dataset_name]["state_encoding"]
        if state_encoding == StateEncoding.POS_EULER:
            state_names = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
            if "libero" in dataset_name:
                state_names = [
                    "x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper", "gripper",
                ]
        elif state_encoding == StateEncoding.POS_QUAT:
            state_names = ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]
        elif state_encoding == StateEncoding.JOINT:
            state_names = [f"motor_{i}" for i in range(7)] + ["gripper"]
            state_obs_keys = OXE_DATASET_CONFIGS[dataset_name]["state_obs_keys"]
            pad_count = state_obs_keys[:-1].count(None)
            if pad_count > 0:
                state_names[-pad_count - 1 : -1] = ["pad"] * pad_count
            state_names[-1] = "pad" if state_obs_keys[-1] is None else state_names[-1]

    # --- [복구된 로직] 데이터셋 설정에 따라 Action 차원 및 이름 결정 ---
    action_names = [f"motor_{i}" for i in range(8)]
    if dataset_name in OXE_DATASET_CONFIGS:
        action_encoding = OXE_DATASET_CONFIGS[dataset_name]["action_encoding"]
        if action_encoding == ActionEncoding.EEF_POS:
            # DROID는 보통 여기 해당되어 7차원이 됩니다.
            action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            if "libero" in dataset_name:
                action_names = ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]
        elif action_encoding == ActionEncoding.JOINT_POS:
            action_names = [f"motor_{i}" for i in range(7)] + ["gripper"]

    DEFAULT_FEATURES = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": {"motors": state_names},
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": {"motors": action_names},
        },
    }

    try:
        validate_action_config(dataset_name)
        DEFAULT_FEATURES["absolute_action"] = {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": {"motors": action_names},
        }
    except Exception:
        pass

    obs = builder.info.features["steps"]["observation"]
    features = {
        f"observation.images.{key}": {
            "dtype": "video" if use_videos else "image",
            "shape": value.shape,
            "names": ["height", "width", "rgb"],
        }
        for key, value in obs.items()
        if "depth" not in key and any(x in key for x in ["image", "rgb"])
    }
    return {**features, **DEFAULT_FEATURES}

# -----------------------------------------------------------------------------
# 4. Main Conversion Loop (The Speed Engine)
# -----------------------------------------------------------------------------
def save_as_lerobot_dataset(
    lerobot_dataset: LeRobotDataset, 
    raw_dataset: tf.data.Dataset, 
    dataset_name: str, 
    total_episodes: int = None,
):
    compute_abs = "absolute_action" in lerobot_dataset.features

    # Convert TF Dataset to Numpy Iterator
    # Prefetching is handled in the pipeline construction in create_lerobot_dataset
    iterator = raw_dataset.as_numpy_iterator()

    for episode_idx, episode in tqdm(enumerate(iterator), total=total_episodes, desc=f"Converting {dataset_name}"):
        
        num_frames = episode["action"].shape[0]
        
        # Determine task description (usually same for whole episode)
        # Handle cases where language instruction might be bytes or string
        task_desc = episode["task"][0]
        if isinstance(task_desc, bytes):
            task_desc = task_desc.decode("utf-8")

        for i in range(num_frames):
            current_state = episode["proprio"][i]
            current_action = episode["action"][i]
            
            # Extract images
            frame_data = {
                f"observation.images.{key}": value[i]
                for key, value in episode["observation"].items()
                if "depth" not in key and any(x in key for x in ["image", "rgb"])
            }
            
            frame_data.update({
                "observation.state": current_state,
                "action": current_action,
                "task": task_desc,
            })

            # Compute absolute action if needed
            if compute_abs:
                # Note: This is fast enough in Python for most cases, 
                # but could be moved to TF if strictly necessary.
                # Given the bottleneck is usually Video I/O, this is fine.
                try:
                    abs_action = compute_absolute_action(current_state, current_action, dataset_name)
                    frame_data["absolute_action"] = abs_action
                except Exception:
                    pass

            lerobot_dataset.add_frame(frame_data)
        
        lerobot_dataset.save_episode()


def create_lerobot_dataset(
    raw_dir: Path,
    repo_id: str = None,
    local_dir: Path = None,
    push_to_hub: bool = False,
    fps: int = None,
    robot_type: str = None,
    use_videos: bool = True,
    image_writer_process: int = 40,
    image_writer_threads: int = 2,
):
    # Path resolution
    last_part = raw_dir.name
    if re.match(r"^\d+\.\d+\.\d+$", last_part):
        version = last_part
        dataset_name = raw_dir.parent.name
        data_dir = raw_dir.parent.parent
    else:
        version = ""
        dataset_name = last_part
        data_dir = raw_dir.parent

    if local_dir is None:
        local_dir = Path(HF_LEROBOT_HOME)
    local_dir /= f"{dataset_name}_{version}_lerobot"
    
    # Safe cleanup
    if local_dir.exists():
        print(f"Removing existing directory: {local_dir}")
        shutil.rmtree(local_dir)

    print(f"Loading TFDS Builder for {dataset_name}...")
    builder = tfds.builder(dataset_name, data_dir=data_dir, version=version)
    features = generate_features_from_raw(builder, use_videos)
    
    filter_fn = lambda e: e["success"] if dataset_name == "kuka" else True
    
    # --- OPTIMIZATION START ---
    print("Constructing Optimized TF Pipeline...")
    
    # 1. AUTOTUNE for parallel calls
    AUTOTUNE = tf.data.AUTOTUNE
    
    # 2. Map with parallelism (This uses your CPU cores to read/parse TFRecords)
    raw_dataset = (
        builder.as_dataset(split="train")
        .filter(filter_fn)
        .map(
            lambda e: transform_raw_dataset(e, dataset_name), 
            num_parallel_calls=AUTOTUNE,
            deterministic=False # Faster if order strictness isn't 100% critical (usually fine for training sets)
        )
        .prefetch(AUTOTUNE) # Keep the buffer full
    )
    # --- OPTIMIZATION END ---

    if fps is None:
        if dataset_name in OXE_DATASET_CONFIGS:
            fps = OXE_DATASET_CONFIGS[dataset_name]["control_frequency"]
        else:
            fps = 10

    if robot_type is None:
        if dataset_name in OXE_DATASET_CONFIGS:
            robot_type = OXE_DATASET_CONFIGS[dataset_name]["robot_type"]
            robot_type = robot_type.lower().replace(" ", "_").replace("-", "_")
        else:
            robot_type = "unknown"

    print(f"Initializing LeRobotDataset with {image_writer_process} processes...")
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        root=local_dir,
        fps=int(fps),
        use_videos=use_videos,
        features=features,
        # High parallelism for encoding
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_process, 
    )

    # Stats finding
    total_episodes = None
    try:
        json_files = glob.glob(str(raw_dir / "dataset_statistics*.json"))
        if not json_files:
             json_files = glob.glob(str(Path(builder.data_dir) / "dataset_statistics*.json"))
        if json_files:
            with open(json_files[0], 'r') as f:
                stats = json.load(f)
                if "num_trajectories" in stats:
                    total_episodes = stats["num_trajectories"]
    except Exception:
        pass

    save_as_lerobot_dataset(
        lerobot_dataset, 
        raw_dataset, 
        dataset_name=dataset_name, 
        total_episodes=total_episodes
    )

    if push_to_hub:
        assert repo_id is not None
        tags = ["LeRobot", dataset_name, "rlds", "openx"]
        if robot_type != "unknown":
            tags.append(robot_type)
        lerobot_dataset.push_to_hub(
            tags=tags,
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--robot-type", type=str, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--use-videos", action="store_true")
    
    # Tuned defaults for high-core machines
    parser.add_argument("--image-writer-process", type=int, default=40, help="Processes for video encoding")
    parser.add_argument("--image-writer-threads", type=int, default=1, help="Threads per encoding process")

    args = parser.parse_args()
    create_lerobot_dataset(**vars(args))

if __name__ == "__main__":
    main()