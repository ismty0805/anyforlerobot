#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
For all datasets in the RLDS format.
For https://github.com/google-deepmind/open_x_embodiment (OPENX) datasets.

NOTE: You need to install tensorflow, tensorflow_datsets, torch, and pytorch3d.

Example:
    python openx_rlds.py \
        --raw-dir /path/to/bridge_orig/1.0.0 \
        --local-dir /path/to/local_dir \
        --repo-id your_id \
        --use-videos \
        --push-to-hub
"""

import argparse
import re
import shutil
import functools
import json
import glob
from functools import partial
from pathlib import Path

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

np.set_printoptions(precision=2)


class RotationTransform:
    """Adapted from https://github.com/real-stanford/diffusion_policy/blob/548a52bbb105518058e27bf34dcf90bf6f73681a/diffusion_policy/model/common/rotation_transformer.py"""

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
        inverse_funcs = list()

        if from_rep != "matrix":
            if from_rep == "euler_angles":
                f1 = getattr(pt, f"{from_rep}_to_matrix")
                funcs = [partial(f1, convention=from_convention), getattr(pt, f"matrix_to_{from_rep}")]
                funcs[1] = partial(funcs[1], convention=from_convention)
            else:
                funcs = [getattr(pt, f"{from_rep}_to_matrix"), getattr(pt, f"matrix_to_{from_rep}")]
            
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            if to_rep == "euler_angles":
                f1 = getattr(pt, f"matrix_to_{to_rep}")
                funcs = [partial(f1, convention=to_convention), getattr(pt, f"{to_rep}_to_matrix")]
                funcs[1] = partial(funcs[1], convention=to_convention)
            else:
                funcs = [getattr(pt, f"matrix_to_{to_rep}"), getattr(pt, f"{to_rep}_to_matrix")]
            
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: torch.Tensor, funcs: list) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        for func in funcs:
            x = func(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return self._apply_funcs(x, self.forward_funcs)


def transform_raw_dataset(episode, dataset_name):
    traj = next(iter(episode["steps"].batch(episode["steps"].cardinality())))

    if dataset_name in OXE_STANDARDIZATION_TRANSFORMS:
        traj = OXE_STANDARDIZATION_TRANSFORMS[dataset_name](traj)

    if dataset_name in OXE_DATASET_CONFIGS:
        state_obs_keys = OXE_DATASET_CONFIGS[dataset_name]["state_obs_keys"]
    else:
        state_obs_keys = [None for _ in range(8)]

    proprio = tf.concat(
        [
            (
                tf.zeros((tf.shape(traj["action"])[0], 1), dtype=tf.float32)  # padding
                if key is None
                else tf.cast(traj["observation"][key], tf.float32)
            )
            for key in state_obs_keys
        ],
        axis=1,
    )

    traj.update(
        {
            "proprio": proprio,
            "task": traj.pop("language_instruction"),
            "action": tf.cast(traj["action"], tf.float32),
        }
    )

    episode["steps"] = traj
    return episode


def validate_action_config(dataset_name):
    """
    Validates if absolute action can be computed based on dataset config.
    Raises ValueError if configuration is missing or insufficient.
    """
    if dataset_name not in OXE_DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not found in OXE configs.")
    
    config = OXE_DATASET_CONFIGS[dataset_name]
    state_obs_keys = config["state_obs_keys"]
    action_encoding = config["action_encoding"]

    if action_encoding == ActionEncoding.JOINT_POS:
        # Check if joint states are present (usually first 7)
        for i in range(min(7, len(state_obs_keys))):
             if state_obs_keys[i] is None:
                 raise ValueError(f"Missing joint state key at index {i} for JOINT_POS action.")

    elif action_encoding == ActionEncoding.EEF_POS:
        # Check XYZ (0-2) and Rotation (3-5 or 3-6)
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


def generate_features_from_raw(builder: tfds.core.DatasetBuilder, use_videos: bool = True):
    dataset_name = Path(builder.data_dir).parent.name

    state_names = [f"motor_{i}" for i in range(8)]
    if dataset_name in OXE_DATASET_CONFIGS:
        state_encoding = OXE_DATASET_CONFIGS[dataset_name]["state_encoding"]
        if state_encoding == StateEncoding.POS_EULER:
            state_names = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
            if "libero" in dataset_name:
                state_names = [
                    "x",
                    "y",
                    "z",
                    "axis_angle1",
                    "axis_angle2",
                    "axis_angle3",
                    "gripper",
                    "gripper",
                ]  # 2D gripper state
        elif state_encoding == StateEncoding.POS_QUAT:
            state_names = ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]
        elif state_encoding == StateEncoding.JOINT:
            state_names = [f"motor_{i}" for i in range(7)] + ["gripper"]
            state_obs_keys = OXE_DATASET_CONFIGS[dataset_name]["state_obs_keys"]
            pad_count = state_obs_keys[:-1].count(None)
            state_names[-pad_count - 1 : -1] = ["pad"] * pad_count
            state_names[-1] = "pad" if state_obs_keys[-1] is None else state_names[-1]

    action_names = [f"motor_{i}" for i in range(8)]
    if dataset_name in OXE_DATASET_CONFIGS:
        action_encoding = OXE_DATASET_CONFIGS[dataset_name]["action_encoding"]
        if action_encoding == ActionEncoding.EEF_POS:
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

    # [수정됨] 절대 행동 계산 가능 여부를 미리 확인하고, 
    # 불가능하거나 에러가 발생하면 features 딕셔너리에 'absolute_action'을 추가하지 않음.
    try:
        validate_action_config(dataset_name)
        
        DEFAULT_FEATURES["absolute_action"] = {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": {"motors": action_names},
        }
        print(f"[{dataset_name}] Absolute action enabled.")
    except Exception as e:
        print(f"[{dataset_name}] Absolute action DISABLED. Reason: {e}")
        # features에 추가하지 않음 -> info에도 남지 않음

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


def save_as_lerobot_dataset(
    lerobot_dataset: LeRobotDataset, 
    raw_dataset: tf.data.Dataset, 
    dataset_name: str, 
    total_episodes: int = None,
    **kwargs
):
    # features에 absolute_action이 있는 경우에만 계산 시도
    compute_abs = "absolute_action" in lerobot_dataset.features

    for episode in tqdm(raw_dataset.as_numpy_iterator(), total=total_episodes, desc=f"Converting {dataset_name}"):
        traj = episode["steps"]
        for i in range(traj["action"].shape[0]):
            
            current_state = traj["proprio"][i]
            current_action = traj["action"][i]
            
            frame_data = {
                f"observation.images.{key}": value[i]
                for key, value in traj["observation"].items()
                if "depth" not in key and any(x in key for x in ["image", "rgb"])
            }
            
            frame_data.update({
                "observation.state": current_state,
                "action": current_action,
                "task": traj["task"][0].decode(),
            })

            # features에 정의되어 있을 때만 계산 및 추가
            if compute_abs:
                absolute_action = None
                try:
                    absolute_action = compute_absolute_action(current_state, current_action, dataset_name)
                except Exception:
                    pass
                
                if absolute_action is not None:
                    frame_data["absolute_action"] = absolute_action

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
    image_writer_process: int = 60,
    image_writer_threads: int = 1,
    keep_images: bool = True,
):
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
    if local_dir.exists():
        shutil.rmtree(local_dir)

    builder = tfds.builder(dataset_name, data_dir=data_dir, version=version)
    features = generate_features_from_raw(builder, use_videos)
    filter_fn = lambda e: e["success"] if dataset_name == "kuka" else True
    
    # [수정됨] partial 대신 lambda 사용 (AutoGraph 에러 방지)
    raw_dataset = (
        builder.as_dataset(split="train")
        .filter(filter_fn)
        .map(lambda e: transform_raw_dataset(e, dataset_name),
            num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

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

    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        root=local_dir,
        fps=int(fps),
        use_videos=use_videos,
        features=features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_process,
    )

    # --- Find Total Episodes from dataset_statistics.json if available ---
    total_episodes = None
    try:
        json_files = glob.glob(str(raw_dir / "dataset_statistics*.json"))
        if not json_files:
             json_files = glob.glob(str(Path(builder.data_dir) / "dataset_statistics*.json"))
        
        if json_files:
            print(f"Found statistics file: {json_files[0]}")
            with open(json_files[0], 'r') as f:
                stats = json.load(f)
                if "num_trajectories" in stats:
                    total_episodes = stats["num_trajectories"]
                    print(f"Loaded total episodes: {total_episodes}")
    except Exception as e:
        print(f"Could not load dataset statistics for tqdm: {e}")

    save_as_lerobot_dataset(
        lerobot_dataset, 
        raw_dataset, 
        dataset_name=dataset_name, 
        total_episodes=total_episodes, 
        keep_images=keep_images
    )

    if push_to_hub:
        assert repo_id is not None
        tags = ["LeRobot", dataset_name, "rlds"]
        if dataset_name in OXE_DATASET_CONFIGS:
            tags.append("openx")
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

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing input raw datasets (e.g. `path/to/dataset` or `path/to/dataset/version).",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="When provided, writes the dataset converted to LeRobotDataset format in this directory  (e.g. `data/lerobot/aloha_mobile_chair`).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="Repositery identifier on Hugging Face: a community or a user name `/` the name of the dataset, required when push-to-hub is True",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload to hub.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default=None,
        help="Robot type of this dataset.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frame rate used to collect videos. Default fps equals to the control frequency of the robot.",
    )
    parser.add_argument(
        "--use-videos",
        action="store_true",
        help="Convert each episode of the raw dataset to an mp4 video. This option allows 60 times lower disk space consumption and 25 faster loading time during training.",
    )
    parser.add_argument(
        "--image-writer-process",
        type=int,
        default=60,
        help="Number of processes of image writer for saving images.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=1,
        help="Number of threads per process of image writer for saving images.",
    )

    args = parser.parse_args()
    create_lerobot_dataset(**vars(args))


if __name__ == "__main__":
    main()