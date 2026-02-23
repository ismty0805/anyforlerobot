#!/usr/bin/env python

"""
Native Sharding Worker for OpenX to LeRobot.
Leverages existing TFRecord shards (e.g., of-02048) to minimize memory usage.
Enforces serial processing to survive SVT-AV1 encoding.
"""

import argparse
import os
import re
import json
from pathlib import Path
from functools import partial
import math
import shutil
import subprocess
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import pytorch3d.transforms as pt
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from oxe_utils.configs import OXE_DATASET_CONFIGS, ActionEncoding, StateEncoding
from oxe_utils.transforms import OXE_STANDARDIZATION_TRANSFORMS

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# -----------------------------------------------------------------------------
# [필수] Helper Classes & Functions (이전과 동일, 복사됨)
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
    config = OXE_DATASET_CONFIGS[dataset_name]
    action_encoding = config["action_encoding"]
    abs_action = np.zeros_like(action)
    if action_encoding == ActionEncoding.JOINT_POS:
        n_dims = len(action); abs_action = state[:n_dims] + action
    elif action_encoding == ActionEncoding.EEF_POS:
        state_encoding = config["state_encoding"]
        abs_action[:3] = state[:3] + action[:3]
        if state_encoding == StateEncoding.POS_EULER:
            # [Fix] Memory safety copy
            state_rot = torch.from_numpy(state[3:6].copy()).float().unsqueeze(0)
            action_rot = torch.from_numpy(action[3:6].copy()).float().unsqueeze(0)
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
            s_q = state[3:7]; a_q = action[3:7]
            state_rot = torch.tensor([[s_q[3], s_q[0], s_q[1], s_q[2]]], dtype=torch.float32)
            action_rot = torch.tensor([[a_q[3], a_q[0], a_q[1], a_q[2]]], dtype=torch.float32)
            tf_to_mat = RotationTransform(from_rep="quaternion", to_rep="matrix")
            tf_to_quat = RotationTransform(from_rep="matrix", to_rep="quaternion")
            mat_state = tf_to_mat.forward(state_rot); mat_action = tf_to_mat.forward(action_rot)
            mat_abs = torch.bmm(mat_action, mat_state)
            abs_rot_wxyz = tf_to_quat.forward(mat_abs).squeeze(0).numpy()
            abs_action[3:7] = np.array([abs_rot_wxyz[1], abs_rot_wxyz[2], abs_rot_wxyz[3], abs_rot_wxyz[0]])
            if len(action) > 7: abs_action[7] = state[7] + action[7]
    return abs_action

def transform_raw_dataset(episode, dataset_name):
    traj = episode["steps"]
    traj = next(iter(traj.batch(50000))) 
    if dataset_name in OXE_STANDARDIZATION_TRANSFORMS:
        traj = OXE_STANDARDIZATION_TRANSFORMS[dataset_name](traj)
    state_obs_keys = OXE_DATASET_CONFIGS[dataset_name]["state_obs_keys"] if dataset_name in OXE_DATASET_CONFIGS else [None]*8
    proprio_list = []
    for key in state_obs_keys:
        if key is None: proprio_list.append(tf.zeros((tf.shape(traj["action"])[0], 1), dtype=tf.float32))
        else: proprio_list.append(tf.cast(traj["observation"][key], tf.float32))
    proprio = tf.concat(proprio_list, axis=1)
    return {
        "observation": traj["observation"], "action": tf.cast(traj["action"], tf.float32),
        "task": traj["language_instruction"], "proprio": proprio,
    }

def generate_features_from_raw(builder: tfds.core.DatasetBuilder, use_videos: bool = True):
    dataset_name = Path(builder.data_dir).parent.name
    state_names = [f"motor_{i}" for i in range(8)]; action_names = [f"motor_{i}" for i in range(8)]
    if dataset_name in OXE_DATASET_CONFIGS:
        if OXE_DATASET_CONFIGS[dataset_name]["state_encoding"] == StateEncoding.POS_EULER:
            state_names = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
        if OXE_DATASET_CONFIGS[dataset_name]["action_encoding"] == ActionEncoding.EEF_POS:
            action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    
    DEFAULT_FEATURES = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": {"motors": state_names}},
        "action": {"dtype": "float32", "shape": (len(action_names),), "names": {"motors": action_names}},
    }
    try:
        if dataset_name in OXE_DATASET_CONFIGS: DEFAULT_FEATURES["absolute_action"] = DEFAULT_FEATURES["action"]
    except: pass

    obs = builder.info.features["steps"]["observation"]
    features = {
        f"observation.images.{key}": {"dtype": "video" if use_videos else "image", "shape": value.shape, "names": ["height", "width", "rgb"]}
        for key, value in obs.items() if "depth" not in key and any(x in key for x in ["image", "rgb"])
    }
    return {**features, **DEFAULT_FEATURES}

# -----------------------------------------------------------------------------
# [핵심] Native Shard Processing Logic
# -----------------------------------------------------------------------------
def process_shards_native(args):
    # 1. 경로 및 설정 감지
    raw_dir_clean = args.raw_dir.resolve()
    last_part = raw_dir_clean.name
    if re.match(r"^\d+\.\d+\.\d+$", last_part):
        version = last_part; dataset_name = raw_dir_clean.parent.name; data_dir = raw_dir_clean.parent.parent
    else:
        version = None; dataset_name = last_part; data_dir = raw_dir_clean.parent

    # 2. Native Shard 계산
    # 총 TFRecord 파일 수 (DROID의 경우 2048개)
    total_physical_shards = args.total_physical_shards 
    
    # 현재 Slurm Job이 처리할 파일 범위 계산
    # 예: 2048개 파일 / 200개 잡 = 잡당 약 10.24개 -> 10개 또는 11개
    shards_per_job = total_physical_shards / args.num_slurm_jobs
    
    start_shard_idx = math.floor(args.job_id * shards_per_job)
    end_shard_idx = math.floor((args.job_id + 1) * shards_per_job)
    
    # 마지막 Job 보정
    if args.job_id == args.num_slurm_jobs - 1:
        end_shard_idx = total_physical_shards

    # 처리할 파일이 없으면 종료
    if start_shard_idx >= end_shard_idx:
        print(f"Job {args.job_id}: No shards to process.")
        return

    print(f"Job {args.job_id}: Processing physical shards {start_shard_idx} to {end_shard_idx} (Total: {total_physical_shards})")
    
    # 3. TFDS Split String 생성 (물리적 파일 지정)
    # 문법: train[StartShard shard : EndShard shard]
    split_arg = f"train[{start_shard_idx}shard:{end_shard_idx}shard]"
    print(f"TFDS Split Argument: {split_arg}")

    shard_output_dir = args.local_dir / "_temp_shards" / f"job_{args.job_id:04d}"
    if shard_output_dir.exists(): shutil.rmtree(shard_output_dir)

    # 4. Dataset 로딩
    builder = tfds.builder(dataset_name, data_dir=data_dir, version=version)
    features = generate_features_from_raw(builder, use_videos=args.use_videos)
    
    # 여기서 shard() 함수를 쓰지 않고, split 인자 자체로 범위를 제한합니다.
    ds = builder.as_dataset(split=split_arg)
    
    # [최적화] Prefetch를 작게 잡아서 메모리 절약
    ds = ds.map(lambda e: transform_raw_dataset(e, dataset_name)).prefetch(1) 

    # FPS 설정
    fps = 10
    robot_type = "unknown"
    if dataset_name in OXE_DATASET_CONFIGS:
        fps = OXE_DATASET_CONFIGS[dataset_name]["control_frequency"]
        robot_type = OXE_DATASET_CONFIGS[dataset_name]["robot_type"]

    if args.mode == "v3":
        # 5. LeRobot Dataset creation
        # [중요] SVT-AV1 OOM 방지를 위한 단일 프로세스 모드
        lerobot_dataset = LeRobotDataset.create(
            repo_id=None,
            robot_type=robot_type,
            root=shard_output_dir,
            fps=int(fps),
            use_videos=args.use_videos,
            features=features,
            image_writer_processes=1, # [핵심] 병렬 인코딩 금지
            image_writer_threads=1,   # 코어 수만큼 스레드 할당
            vcodec="h264"
        )
        
        # 변환 루프
        compute_abs = "absolute_action" in lerobot_dataset.features
        iterator = ds.as_numpy_iterator()
        count = 0

        # tqdm ETA를 위한 총 에피소드 수 계산
        total_episodes = None
        try:
            shard_lengths = builder.info.splits['train'].shard_lengths
            if shard_lengths:
                total_episodes = sum(shard_lengths[start_shard_idx:end_shard_idx])
        except:
            pass

        for episode in tqdm(iterator, total=total_episodes, desc=f"Job {args.job_id}", mininterval=10.0):
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
                if compute_abs:
                    try: frame_data["absolute_action"] = compute_absolute_action(traj["proprio"][i], traj["action"][i], dataset_name)
                    except: pass
                
                lerobot_dataset.add_frame(frame_data)
            
            lerobot_dataset.save_episode()
            count += 1
            
            if count % 10 == 0:
                import gc; gc.collect()

        print(f"Job {args.job_id} Finished. Processed {count} episodes.")
        del lerobot_dataset
    else:
        # 6. Legacy (v2.1) mode: Save each episode individually
        print(f"Job {args.job_id}: Running in LEGACY (v2.1) mode.")
        shard_output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = shard_output_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        image_obs_keys = OXE_DATASET_CONFIGS[dataset_name]["image_obs_keys"] if dataset_name in OXE_DATASET_CONFIGS else {}
        
        iterator = ds.as_numpy_iterator()
        count = 0
        metadata_records = []

        for episode in tqdm(iterator, desc=f"Job {args.job_id}", mininterval=10.0):
            num_frames = episode["action"].shape[0]
            task_desc = episode["task"][0].decode("utf-8") if isinstance(episode["task"][0], bytes) else str(episode["task"][0])
            
            # Save Parquet
            rows = []
            for i in range(num_frames):
                row = {
                    "observation.state": episode["proprio"][i].tolist(),
                    "action": episode["action"][i].tolist(),
                    "task": task_desc,
                }
                try:
                    abs_act = compute_absolute_action(episode["proprio"][i], episode["action"][i], dataset_name)
                    row["absolute_action"] = abs_act.tolist()
                except:
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
                import gc; gc.collect()

        # Save local metadata list
        with open(shard_output_dir / "metadata.jsonl", "w") as f:
            for rec in metadata_records:
                f.write(json.dumps(rec) + "\n")

    import gc; gc.collect()
    print(f"Job {args.job_id} cleaned up.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--use-videos", action="store_true")
    
    # Slurm 관련 인자
    parser.add_argument("--job-id", type=int, required=True, help="Slurm Array Task ID (0 to N-1)")
    parser.add_argument("--num-slurm_jobs", type=int, required=True, help="Total number of Slurm jobs")
    
    # 데이터셋 물리 정보
    parser.add_argument("--total-physical-shards", type=int, default=2048, help="Total number of .tfrecord files (e.g. 2048 for DROID)")

    # Mode
    parser.add_argument("--mode", type=str, default="v3", choices=["v3", "legacy"], help="Output format mode")

    args = parser.parse_args()
    process_shards_native(args)

if __name__ == "__main__":
    main()