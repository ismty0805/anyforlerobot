#!/usr/bin/env python

"""
Others Dataset Worker for Non-OpenX datasets (agibot, galaxea, humanoid_everyday, etc.)
Similar to openx_native_worker.py but uses OTHERS_STANDARDIZATION_TRANSFORMS and OTHERS_DATASET_CONFIGS
"""

import argparse
import shutil
import os
import re
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pathlib import Path
from functools import partial
import math

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import pytorch3d.transforms as pt
from tqdm import tqdm

from others_utils.configs import OTHERS_DATASET_CONFIGS, ActionEncoding, StateEncoding
from others_utils.transforms import OTHERS_STANDARDIZATION_TRANSFORMS

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import subprocess

# -----------------------------------------------------------------------------
# Helper Classes & Functions
# -----------------------------------------------------------------------------
class RotationTransform:
    def __init__(self, from_rep="axis_angle", to_rep="rotation_6d", from_convention=None, to_convention=None):
        if from_rep.startswith("euler_angles") and from_convention is None:
            try: from_convention = from_rep.split("_")[-1]; from_rep = "euler_angles"
            except IndexError: pass
        if to_rep.startswith("euler_angles") and to_convention is None:
            try: to_convention = to_rep.split("_")[-1]; to_rep = "euler_angles"
            except IndexError: pass
        if from_convention: from_convention = from_convention.upper()
        if to_convention: to_convention = to_convention.upper()
        self.forward_funcs = []
        if from_rep != "matrix":
            if from_rep == "euler_angles":
                f1 = getattr(pt, f"{from_rep}_to_matrix")
                funcs = [partial(f1, convention=from_convention), getattr(pt, f"matrix_to_{from_rep}")]
                funcs[1] = partial(funcs[1], convention=from_convention)
            else:
                funcs = [getattr(pt, f"{from_rep}_to_matrix"), getattr(pt, f"matrix_to_{from_rep}")]
            self.forward_funcs.append(funcs[0])
        if to_rep != "matrix":
            if to_rep == "euler_angles":
                f1 = getattr(pt, f"matrix_to_{to_rep}")
                funcs = [partial(f1, convention=to_convention), getattr(pt, f"{to_rep}_to_matrix")]
                funcs[1] = partial(funcs[1], convention=to_convention)
            else:
                funcs = [getattr(pt, f"matrix_to_{to_rep}"), getattr(pt, f"{to_rep}_to_matrix")]
            self.forward_funcs.append(funcs[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray): x = torch.from_numpy(x)
        for func in self.forward_funcs: x = func(x)
        return x

def compute_absolute_action(state, action, dataset_name):
    """
    Compute absolute action from state and relative action.
    For special encodings (AGIBOT, GALAXEA, etc.), actions are already absolute.
    """
    config = OTHERS_DATASET_CONFIGS[dataset_name]
    action_encoding = config["action_encoding"]
    abs_action = np.zeros_like(action)
    
    # For special robot types, actions are already absolute joint positions
    if action_encoding in [ActionEncoding.AGIBOT_DEXHAND, ActionEncoding.AGIBOT_GRIPPER, 
                           ActionEncoding.GALAXEA, ActionEncoding.HUMANOID_EVERYDAY_G1,
                           ActionEncoding.HUMANOID_EVERYDAY_H1, ActionEncoding.ACTION_NET,
                           ActionEncoding.NEURAL_GR1]:
        return action  # Already absolute
    
    # Standard EEF/Joint position logic (same as OpenX)
    if action_encoding == ActionEncoding.JOINT_POS:
        n_dims = len(action)
        abs_action = state[:n_dims] + action
    elif action_encoding == ActionEncoding.EEF_POS:
        state_encoding = config["state_encoding"]
        abs_action[:3] = state[:3] + action[:3]
        if state_encoding == StateEncoding.POS_EULER:
            state_rot = torch.from_numpy(state[3:6].copy()).float().unsqueeze(0)
            action_rot = torch.from_numpy(action[3:6].copy()).float().unsqueeze(0)
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

def transform_raw_dataset(episode, dataset_name):
    traj = episode["steps"]
    # Remove ragged features that cause batching to fail
    def filter_ragged(step):
        if "observation" in step:
            # Create a copy of observation without tactile keys
            obs = {k: v for k, v in step["observation"].items() if k not in ["tactile_sensor_id", "tactile_values"]}
            step["observation"] = obs
        return step
    
    traj = traj.map(filter_ragged)
    traj = next(iter(traj.batch(50000)))
    if dataset_name in OTHERS_STANDARDIZATION_TRANSFORMS:
        traj = OTHERS_STANDARDIZATION_TRANSFORMS[dataset_name](traj)
    state_obs_keys = OTHERS_DATASET_CONFIGS[dataset_name]["state_obs_keys"] if dataset_name in OTHERS_DATASET_CONFIGS else [None]*8
    proprio_list = []
    for key in state_obs_keys:
        if key is None:
            proprio_list.append(tf.zeros((tf.shape(traj["action"])[0], 1), dtype=tf.float32))
        else:
            proprio_list.append(tf.cast(traj["observation"][key], tf.float32))
    proprio = tf.concat(proprio_list, axis=1)
    # Handle action (cast if it's a tensor, otherwise keep as is if it's a structure)
    action = traj["action"]
    if isinstance(action, tf.Tensor):
        action = tf.cast(action, tf.float32)
    
    # Handle task description (try multiple possible keys)
    task = traj.get("language_instruction", None)
    if task is None:
        task = traj.get("task", None)
    if task is None:
        # Default to empty strings
        task = tf.fill([tf.shape(list(traj["observation"].values())[0])[0]], "")

    res = {
        "observation": traj["observation"],
        "action": action,
        "task": task,
        "proprio": proprio,
    }
    if "original_observation" in traj:
        res["original_observation"] = traj["original_observation"]
    if "original_action" in traj:
        res["original_action"] = traj["original_action"]
    return res

def generate_features_from_raw(builder: tfds.core.DatasetBuilder, dataset_name: str, use_videos: bool = True):
    # Get state and action dimensions from config
    config = OTHERS_DATASET_CONFIGS.get(dataset_name, {})
    state_encoding = config.get("state_encoding", StateEncoding.NONE)
    action_encoding = config.get("action_encoding", ActionEncoding.EEF_POS)
    
    # Determine state and action dimensions
    if state_encoding == StateEncoding.AGIBOT_DEXHAND:
        state_dim = 44  # 7+6+6+3+7+6+6+3
        action_dim = 44
        state_names = [f"motor_{i}" for i in range(44)]
        action_names = [f"motor_{i}" for i in range(44)]
    elif state_encoding == StateEncoding.AGIBOT_GRIPPER:
        state_dim = 34  # 7+1+6+3+7+1+6+3
        action_dim = 34
        state_names = [f"motor_{i}" for i in range(34)]
        action_names = [f"motor_{i}" for i in range(34)]
    elif state_encoding == StateEncoding.GALAXEA:
        state_dim = 18  # 6+1+6+1+4
        action_dim = 26  # From config
        state_names = [f"motor_{i}" for i in range(18)]
        action_names = [f"motor_{i}" for i in range(26)]
    elif state_encoding == StateEncoding.HUMANOID_EVERYDAY_G1:
        state_dim = 28  # 7+7+7+7
        action_dim = 28
        state_names = [f"motor_{i}" for i in range(28)]
        action_names = [f"motor_{i}" for i in range(28)]
    elif state_encoding == StateEncoding.HUMANOID_EVERYDAY_H1:
        state_dim = 26  # 7+6+7+6
        action_dim = 26
        state_names = [f"motor_{i}" for i in range(26)]
        action_names = [f"motor_{i}" for i in range(26)]
    elif state_encoding == StateEncoding.ACTION_NET:
        state_dim = 44
        action_dim = 44
        state_names = [f"motor_{i}" for i in range(44)]
        action_names = [f"motor_{i}" for i in range(44)]
    elif state_encoding == StateEncoding.NEURAL_GR1:
        state_dim = 8  # Default
        action_dim = 8
        state_names = [f"motor_{i}" for i in range(8)]
        action_names = [f"motor_{i}" for i in range(8)]
    else:
        # Default 8-dim state/action
        state_dim = 8
        action_dim = 7
        state_names = [f"motor_{i}" for i in range(8)]
        action_names = [f"motor_{i}" for i in range(7)]
    
    DEFAULT_FEATURES = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": {"motors": state_names}},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": {"motors": action_names}},
        
    }
    
    # Add absolute_action feature for datasets that support it
    try:
        if dataset_name in OTHERS_DATASET_CONFIGS:
            DEFAULT_FEATURES["absolute_action"] = DEFAULT_FEATURES["action"]
    except:
        pass

    obs = builder.info.features["steps"]["observation"]
    features = {
        f"observation.images.{key}": {"dtype": "video" if use_videos else "image", "shape": value.shape, "names": ["height", "width", "rgb"]}
        for key, value in obs.items() if "depth" not in key and any(x in key for x in ["image", "rgb"])
    }
    return {**features, **DEFAULT_FEATURES}

# -----------------------------------------------------------------------------
# Native Shard Processing Logic
# -----------------------------------------------------------------------------
def process_shards_native(args):
    # 1. Path and configuration detection
    raw_dir_clean = args.raw_dir.resolve()
    last_part = raw_dir_clean.name
    if re.match(r"^\d+\.\d+\.\d+$", last_part):
        version = last_part
        detected_dataset_name = raw_dir_clean.parent.name
        data_dir = raw_dir_clean.parent.parent
    else:
        version = None
        detected_dataset_name = last_part
        data_dir = raw_dir_clean.parent

    dataset_name = args.dataset_name if args.dataset_name else detected_dataset_name

    # 2. Native shard calculation
    total_physical_shards = args.total_physical_shards
    shards_per_job = total_physical_shards / args.num_slurm_jobs
    start_shard_idx = math.floor(args.job_id * shards_per_job)
    end_shard_idx = math.floor((args.job_id + 1) * shards_per_job)
    
    if args.job_id == args.num_slurm_jobs - 1:
        end_shard_idx = total_physical_shards

    if start_shard_idx >= end_shard_idx:
        print(f"Job {args.job_id}: No shards to process.")
        return

    print(f"Job {args.job_id}: Processing physical shards {start_shard_idx} to {end_shard_idx} (Total: {total_physical_shards})")
    
    # 3. TFDS Split String
    split_arg = f"train[{start_shard_idx}shard:{end_shard_idx}shard]"
    print(f"TFDS Split Argument: {split_arg}")

    shard_output_dir = args.local_dir / "_temp_shards" / f"job_{args.job_id:04d}"
    if shard_output_dir.exists():
        shutil.rmtree(shard_output_dir)

    # 4. Dataset loading
    builder = tfds.builder(dataset_name, data_dir=data_dir, version=version)
    features = generate_features_from_raw(builder, dataset_name, use_videos=args.use_videos)
    
    ds = builder.as_dataset(split=split_arg)
    ds = ds.map(lambda e: transform_raw_dataset(e, dataset_name)).prefetch(1)

    # FPS configuration
    fps = 10
    robot_type = "unknown"
    if dataset_name in OTHERS_DATASET_CONFIGS:
        fps = OTHERS_DATASET_CONFIGS[dataset_name].get("control_frequency", 10)
        robot_type = dataset_name  # Use dataset name as robot type

    if args.mode == "v3":
        # 5. LeRobot Dataset creation (Original v3 mode)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        lerobot_dataset = LeRobotDataset.create(
            repo_id=None,
            robot_type=robot_type,
            root=shard_output_dir,
            fps=int(fps),
            use_videos=args.use_videos,
            features=features,
            image_writer_processes=1,
            image_writer_threads=4,
            vcodec="h264"
        )
        
        # Conversion loop
        compute_abs = "absolute_action" in lerobot_dataset.features
        iterator = ds.as_numpy_iterator()
        count = 0

        for episode in tqdm(iterator, desc=f"Job {args.job_id}", mininterval=10.0):
            traj = episode
            task_desc = traj["task"][0].decode("utf-8") if isinstance(traj["task"][0], bytes) else str(traj["task"][0])
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
                if "original_observation" in traj:
                    frame_data["original_observation"] = traj["original_observation"][i]
                if "original_action" in traj:
                    frame_data["original_action"] = traj["original_action"][i]
                if compute_abs:
                    try:
                        frame_data["absolute_action"] = compute_absolute_action(traj["proprio"][i], traj["action"][i], dataset_name)
                    except:
                        pass
                
                lerobot_dataset.add_frame(frame_data)
            
            lerobot_dataset.save_episode()
            count += 1
            if count % 10 == 0:
                import gc
                gc.collect()

    else:
        # 6. Legacy (v2.1) mode: Save each episode individually
        print(f"Job {args.job_id}: Running in LEGACY (v2.1) mode.")
        shard_output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = shard_output_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        image_obs_keys = OTHERS_DATASET_CONFIGS[dataset_name]["image_obs_keys"] if dataset_name in OTHERS_DATASET_CONFIGS else {}
        
        iterator = ds.as_numpy_iterator()
        count = 0
        metadata_records = []

        for episode in tqdm(iterator, desc=f"Job {args.job_id}", mininterval=10.0):
            num_frames = episode["action"].shape[0]
            task_desc = episode["task"][0].decode("utf-8") if isinstance(episode["task"][0], bytes) else str(episode["task"][0])
            
            if count == 0:
                print(f"DEBUG: Observation keys: {list(episode['observation'].keys())}")
            
            # Save Parquet
            rows = []
            for i in range(num_frames):
                row = {
                    "observation.state": episode["proprio"][i].tolist(),
                    "action": episode["action"][i].tolist(),
                    "task": task_desc,
                }
                # Add original observation keys
                if "original_observation" in episode:
                    for k, v in episode["original_observation"].items():
                        row[f"observation.{k}"] = v[i].tolist()
                
                # Add original action keys
                if "original_action" in episode:
                    if isinstance(episode["original_action"], dict):
                        for k, v in episode["original_action"].items():
                            row[f"action.{k}"] = v[i].tolist()
                    else:
                        row["original_action"] = episode["original_action"][i].tolist()

                try:
                    abs_act = compute_absolute_action(episode["proprio"][i], episode["action"][i], dataset_name)
                    row["absolute_action"] = abs_act.tolist()
                    if count == 0 and i == 0:
                        print(f"DEBUG: Successfully computed absolute_action (dim={len(abs_act)})")
                except Exception as e:
                    if count == 0 and i == 0:
                        print(f"WARNING: Absolute action computation failed: {e}")
                    pass
                rows.append(row)
            
            df = pd.DataFrame(rows)
            pq_path = data_dir / f"episode_{count:06d}.parquet"
            pq.write_table(pa.Table.from_pandas(df), pq_path)
            
            # Save Videos
            if args.use_videos:
                for cam_name, config_key in image_obs_keys.items():
                    if config_key is None: continue
                    if config_key not in episode["observation"]:
                        if count == 0:
                            print(f"WARNING: Camera key {config_key} not found in observation!")
                        continue
                        
                    vid_dir = shard_output_dir / "videos" / f"observation.images.{cam_name}"
                    vid_dir.mkdir(parents=True, exist_ok=True)
                    vid_path = vid_dir / f"episode_{count:06d}.mp4"
                    
                    img_seq = episode["observation"][config_key]
                    h, w = img_seq.shape[1:3]
                    
                    proc = subprocess.Popen([
                        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps),
                        '-i', '-', '-vcodec', 'libx264', '-crf', '21', '-pix_fmt', 'yuv420p', str(vid_path)
                    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    for i in range(num_frames):
                        proc.stdin.write(img_seq[i].tobytes())
                    proc.stdin.close()
                    proc.wait()
            
            metadata_records.append({
                "local_id": count,
                "length": num_frames,
                "task": task_desc
            })
            count += 1
            if count % 10 == 0:
                import gc
                gc.collect()

        # Save local metadata list
        with open(shard_output_dir / "metadata.jsonl", "w") as f:
            for rec in metadata_records:
                f.write(json.dumps(rec) + "\n")

    print(f"Job {args.job_id} Finished. Processed {count} episodes.")

def main():
    print("Starting worker...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--use-videos", action="store_true")
    
    # Slurm arguments
    parser.add_argument("--job-id", type=int, required=True, help="Slurm Array Task ID (0 to N-1)")
    parser.add_argument("--num-slurm_jobs", type=int, required=True, help="Total number of Slurm jobs")
    
    # Dataset physical information
    parser.add_argument("--total-physical-shards", type=int, default=2048, help="Total number of .tfrecord files")
    parser.add_argument("--dataset-name", type=str, default=None, help="Manually specify dataset name for config/TFDS")
    
    # Mode
    parser.add_argument("--mode", type=str, default="v3", choices=["v3", "legacy"], help="Output format mode")

    args = parser.parse_args()
    process_shards_native(args)

if __name__ == "__main__":
    main()
