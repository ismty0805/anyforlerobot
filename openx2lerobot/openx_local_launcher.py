#!/usr/bin/env python

"""
Local Launcher for OpenX to LeRobot Conversion (Fix for Daemon Error).
Manages processes manually to allow child processes (ffmpeg writers).
"""

import argparse
import time
import sys
import multiprocessing
from pathlib import Path
from tqdm import tqdm

# openx_slurm_worker.py에서 함수 import
try:
    from openx_slurm_worker import process_single_shard
except ImportError:
    print("Error: 'openx_slurm_worker.py' not found. Make sure it is in the same directory.")
    sys.exit(1)

# Argument 값을 전달하기 위한 간단한 클래스 (worker 함수가 args 객체를 기대하므로)
class WorkerArgs:
    def __init__(self, raw_dir, local_dir, use_videos, shard_id, num_shards):
        self.raw_dir = Path(raw_dir)
        self.local_dir = Path(local_dir)
        self.use_videos = use_videos
        self.shard_id = shard_id
        self.num_shards = num_shards

def worker_func(args_dict):
    """
    개별 프로세스에서 실행될 함수
    """
    try:
        # 딕셔너리를 객체로 변환
        args = WorkerArgs(
            raw_dir=args_dict['raw_dir'],
            local_dir=args_dict['local_dir'],
            use_videos=args_dict['use_videos'],
            shard_id=args_dict['shard_id'],
            num_shards=args_dict['num_shards']
        )
        process_single_shard(args)
    except Exception as e:
        # 에러 발생 시 로그 출력 후 종료 (메인 프로세스에 영향 없음)
        print(f"\n[Error] Shard {args.shard_id} failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Local Parallel Launcher (Non-Daemon)")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--use-videos", action="store_true")
    
    # 병렬 설정
    parser.add_argument("--num-shards", type=int, default=200, help="Total number of shards")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of parallel processes")
    
    args = parser.parse_args()

    print(f"Starting Local Conversion (Manual Process Management)")
    print(f"Workers: {args.num_workers}")
    print(f"Total Shards: {args.num_shards}")

    # 작업 리스트 생성
    pending_shards = list(range(args.num_shards))
    active_processes = []
    
    # 진행 상황바
    pbar = tqdm(total=args.num_shards, desc="Overall Progress")
    
    try:
        while pending_shards or active_processes:
            # 1. 완료된 프로세스 정리 (Clean up zombies)
            # is_alive()가 False인 프로세스를 리스트에서 제거
            finished_procs = [p for p in active_processes if not p.is_alive()]
            for p in finished_procs:
                p.join() # 자원 회수
                pbar.update(1)
                active_processes.remove(p)
            
            # 2. 새 프로세스 시작 (슬롯이 비어있고 남은 작업이 있다면)
            while len(active_processes) < args.num_workers and pending_shards:
                shard_id = pending_shards.pop(0)
                
                task_args = {
                    'raw_dir': args.raw_dir,
                    'local_dir': args.local_dir,
                    'use_videos': args.use_videos,
                    'shard_id': shard_id,
                    'num_shards': args.num_shards
                }
                
                # multiprocessing.Process 사용 (Pool 아님 -> Daemon 아님 -> 자식 생성 가능)
                p = multiprocessing.Process(target=worker_func, args=(task_args,))
                p.start()
                active_processes.append(p)
                
                # 너무 빨리 시작하면 리소스 스파이크가 튈 수 있으므로 약간의 딜레이
                time.sleep(0.5) 

            # 3. CPU 과부하 방지 대기
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping all processes...")
        for p in active_processes:
            p.terminate()
            p.join()
            
    pbar.close()
    print("\nAll jobs finished. Run merge_results.py to combine.")

if __name__ == "__main__":
    main()