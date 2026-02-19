#!/usr/bin/env python
import argparse
import shutil
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

def merge_shards(local_dir):
    base_dir = Path(local_dir)
    shards_root = base_dir / "_temp_shards"
    final_dir = base_dir # 혹은 원하는 최종 경로

    if not shards_root.exists():
        print(f"No temporary shards found at {shards_root}")
        return

    print(f"Merging shards from {shards_root} into {final_dir}...")
    
    (final_dir / "videos").mkdir(parents=True, exist_ok=True)
    (final_dir / "meta").mkdir(parents=True, exist_ok=True)

    # 샤드 폴더 찾기
    shard_dirs = sorted(list(shards_root.glob("job_*")))
    print(f"Found {len(shard_dirs)} shards.")

    episodes_dfs = []
    video_counter = 0
    total_episodes = 0

    for shard_dir in tqdm(shard_dirs, desc="Merging"):
        meta_path = shard_dir / "meta/episodes.csv"
        if not meta_path.exists():
            print(f"[Warning] Skipping empty/failed shard: {shard_dir.name}")
            continue

        ep_df = pd.read_csv(meta_path)
        num_eps = len(ep_df)
        
        # 인덱스 리매핑
        # 기존 0, 1, 2... 를 현재 video_counter + 0, video_counter + 1... 로 변경
        old_indices = ep_df["index"].values
        new_indices = range(video_counter, video_counter + num_eps)
        
        # 비디오 파일 이동 (이름 충돌 방지)
        for old_idx, new_idx in zip(old_indices, new_indices):
            src_vid = shard_dir / f"videos/video_{old_idx}.mp4"
            dst_vid = final_dir / f"videos/video_{new_idx}.mp4"
            if src_vid.exists():
                # Move가 Copy보다 훨씬 빠름 (같은 디스크 내라면)
                shutil.move(str(src_vid), str(dst_vid))
        
        ep_df["index"] = list(new_indices)
        episodes_dfs.append(ep_df)
        
        video_counter += num_eps
        total_episodes += num_eps

    # 메타데이터 병합
    if episodes_dfs:
        full_df = pd.concat(episodes_dfs, ignore_index=True)
        full_df.to_csv(final_dir / "meta/episodes.csv", index=False)
        
        # info.json 복사 (첫 번째 정상 샤드에서)
        # 단, total_episodes 수는 수정해야 할 수 있음
        for shard_dir in shard_dirs:
            info_src = shard_dir / "meta/info.json"
            if info_src.exists():
                with open(info_src, 'r') as f:
                    info_data = json.load(f)
                
                # LeRobot info 업데이트 (선택 사항)
                # info_data['total_episodes'] = total_episodes 
                
                with open(final_dir / "meta/info.json", 'w') as f:
                    json.dump(info_data, f, indent=4)
                
                # stats.json 도 있으면 복사
                stats_src = shard_dir / "meta/stats.json"
                if stats_src.exists():
                    shutil.copy(stats_src, final_dir / "meta/stats.json")
                break
    
    print(f"Merge Complete! Total Episodes: {total_episodes}")
    print(f"You can remove {shards_root} manually if everything looks good.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=str, required=True, help="Parent directory containing _temp_shards")
    args = parser.parse_args()
    
    merge_shards(args.local_dir)