#!/usr/bin/env python

"""
Slurm Worker Script for OpenX to LeRobot Conversion.
Fixes KeyError: 'steps' by handling the flattened dictionary structure.
"""

import argparse
import shutil
import os
import re
import json
from pathlib import Path
from functools import partial

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

# Disable GPU for Slurm CPU nodes
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# -----------------------------------------------------------------------------
# Helper Classes (RotationTransform) & Functions
# -----------------------------------------------------------------------------
class RotationTransform:
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

        if from_convention is not None: from_convention = from_convention.upper()
        if to_convention is not None: to_convention = to_convention.upper()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray): x = torch.from_numpy(x)
        for func in self.forward_funcs:
            x = func(x)
        return x

def validate_action_config(dataset_name):
    if dataset_name not in OXE_DATASET_CONFIGS: return True
    config = OXE_DATASET_CONFIGS[dataset_name]
    if config["action_encoding"] == ActionEncoding.JOINT_POS:
        for i in range(min(7, len(config["state_obs_keys"]))):
             if config["state_obs_keys"][i] is None: raise ValueError(f"Missing joint state key")
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
            if len(state) == 8 and len(action) == 7: abs_action[6] = state[7] + action[6]
            else: abs_action[-1] = state[-1] + action[-1]

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
            if len(action) > 7: abs_action[7] = state[7] + action[7]

    return abs_action

def transform_raw_dataset(episode, dataset_name):
    # This function flattens the dataset structure
    traj = episode["steps"]
    traj = next(iter(traj.batch(50000))) 

    if dataset_name in OXE_STANDARDIZATION_TRANSFORMS:
        traj = OXE_STANDARDIZATION_TRANSFORMS[dataset_name](traj)

    if dataset_name in OXE_DATASET_CONFIGS:
        state_obs_keys = OXE_DATASET_CONFIGS[dataset_name]["state_obs_keys"]
    else:
        state_obs_keys = [None for _ in range(8)]

    proprio_list = []
    for key in state_obs_keys:
        if key is None:
            pad = tf.zeros((tf.shape(traj["action"])[0], 1), dtype=tf.float32)
            proprio_list.append(pad)
        else:
            proprio_list.append(tf.cast(traj["observation"][key], tf.float32))
            
    proprio = tf.concat(proprio_list, axis=1)

    # Returns a flat dictionary, NOT containing "steps"
    return {
        "observation": traj["observation"],
        "action": tf.cast(traj["action"], tf.float32),
        "task": traj["language_instruction"],
        "proprio": proprio,
    }

def generate_features_from_raw(builder: tfds.core.DatasetBuilder, use_videos: bool = True):
    dataset_name = Path(builder.data_dir).parent.name
    
    state_names = [f"motor_{i}" for i in range(8)]
    action_names = [f"motor_{i}" for i in range(8)]

    if dataset_name in OXE_DATASET_CONFIGS:
        state_encoding = OXE_DATASET_CONFIGS[dataset_name]["state_encoding"]
        if state_encoding == StateEncoding.POS_EULER:
            state_names = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
        elif state_encoding == StateEncoding.POS_QUAT:
            state_names = ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]
        elif state_encoding == StateEncoding.JOINT:
            state_names = [f"motor_{i}" for i in range(7)] + ["gripper"]
        
        action_encoding = OXE_DATASET_CONFIGS[dataset_name]["action_encoding"]
        if action_encoding == ActionEncoding.EEF_POS:
            action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        elif action_encoding == ActionEncoding.JOINT_POS:
            action_names = [f"motor_{i}" for i in range(7)] + ["gripper"]

    DEFAULT_FEATURES = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": {"motors": state_names}},
        "action": {"dtype": "float32", "shape": (len(action_names),), "names": {"motors": action_names}},
    }

    try:
        validate_action_config(dataset_name)
        DEFAULT_FEATURES["absolute_action"] = {"dtype": "float32", "shape": (len(action_names),), "names": {"motors": action_names}}
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
# Main Worker Function
# -----------------------------------------------------------------------------
def process_single_shard(args):
    raw_dir = args.raw_dir
    local_dir = args.local_dir
    shard_id = args.shard_id
    num_shards = args.num_shards
    
    # --- Auto-detect parameters ---
    raw_dir_clean = raw_dir.resolve()
    last_part = raw_dir_clean.name

    if re.match(r"^\d+\.\d+\.\d+$", last_part):
        version = last_part
        dataset_name = raw_dir_clean.parent.name
        data_dir = raw_dir_clean.parent.parent
    else:
        version = None
        dataset_name = last_part
        data_dir = raw_dir_clean.parent

    # Detect FPS & Robot Type from OXE Configs
    fps = 10
    robot_type = "unknown"
    
    if dataset_name in OXE_DATASET_CONFIGS:
        fps = OXE_DATASET_CONFIGS[dataset_name]["control_frequency"]
        robot_type = OXE_DATASET_CONFIGS[dataset_name]["robot_type"]
        robot_type = robot_type.lower().replace(" ", "_").replace("-", "_")
    
    # Unique output directory for this shard
    shard_output_dir = local_dir / "_temp_shards" / f"shard_{shard_id:04d}"
    
    if shard_output_dir.exists():
        try:
            shutil.rmtree(shard_output_dir)
        except Exception:
            pass

    print(f"[Shard {shard_id}/{num_shards}] Start processing {dataset_name} (v:{version})")

    builder = tfds.builder(dataset_name, data_dir=data_dir, version=version)
    features = generate_features_from_raw(builder, use_videos=args.use_videos)

    ds = builder.as_dataset(split="train")
    if dataset_name == "kuka":
        ds = ds.filter(lambda e: e["success"])
    
    # SHARDING
    ds = ds.shard(num_shards=num_shards, index=shard_id)
    ds = ds.map(lambda e: transform_raw_dataset(e, dataset_name)) 
    
    lerobot_dataset = LeRobotDataset.create(
        repo_id=None,
        robot_type=robot_type,
        root=shard_output_dir,
        fps=int(fps),
        use_videos=args.use_videos,
        features=features,
        image_writer_processes=2,
        image_writer_threads=2,
    )
    
    compute_abs = "absolute_action" in lerobot_dataset.features
    iterator = ds.as_numpy_iterator()

    count = 0
    # [수정됨] transform_raw_dataset은 이미 steps를 풀어서 반환하므로 episode 자체가 traj입니다.
    for episode in tqdm(iterator, desc=f"Shard {shard_id}", mininterval=10.0):
        # traj = episode["steps"]  <-- 이 부분이 에러 원인이었음. 삭제.
        traj = episode 
        
        task_desc = traj["task"][0].decode("utf-8")
        
        num_frames = traj["action"].shape[0]
        for i in range(num_frames):
            frame_data = {
                f"observation.images.{key}": value[i]
                for key, value in traj["observation"].items()
                if "depth" not in key and any(x in key for x in ["image", "rgb"])
            }
            frame_data.update({
                "observation.state": traj["proprio"][i],
                "action": traj["action"][i],
                "task": task_desc,
            })
            
            if compute_abs:
                try:
                    abs_act = compute_absolute_action(traj["proprio"][i], traj["action"][i], dataset_name)
                    frame_data["absolute_action"] = abs_act
                except:
                    pass

            lerobot_dataset.add_frame(frame_data)
        
        lerobot_dataset.save_episode()
        count += 1

    print(f"[Shard {shard_id}] Done. Saved {count} episodes.")

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--use-videos", action="store_true")
    
    # Slurm Array를 위한 내부 인자
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    
    args = parser.parse_args()
    process_single_shard(args)

if __name__ == "__main__":
    main()