#!/usr/bin/env python
"""
Reformat merged intermediate data to LeRobot v3 format.
This script takes the merged data from intermediate directory and reorganizes it
into the proper LeRobot v3 directory structure.
"""
import argparse
import shutil
import json
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

def reformat_to_lerobot_v3(source_dir, output_dir, dataset_name):
    """
    Reformat data from intermediate format to LeRobot v3 format.
    
    Expected source structure (after merge):
        source_dir/
            _temp_shards/
                job_0000/
                    data/chunk-000/file-000.parquet
                    meta/episodes/chunk-000/file-000.parquet
                    meta/info.json
                    meta/stats.json
                    meta/tasks.parquet
                    videos/...
                    images/...
                job_0001/
                ...
    
    Target LeRobot v3 structure:
        output_dir/
            {dataset_name}/
                data/
                    chunk-000/
                        *.parquet
                meta/
                    episodes/
                        chunk-000/
                            *.parquet
                    info.json
                    stats.json
                    tasks.parquet
                videos/
                    chunk-000/
                        *.mp4
                images/
                    chunk-000/
                        *.png
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir) / dataset_name
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "data").mkdir(exist_ok=True)
    (output_path / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    (output_path / "videos").mkdir(exist_ok=True)
    (output_path / "images").mkdir(exist_ok=True)
    
    print(f"Reformatting {dataset_name} from {source_path} to {output_path}")
    
    # Find all job directories
    temp_shards = source_path / "_temp_shards"
    if not temp_shards.exists():
        print(f"Error: {temp_shards} does not exist!")
        return
    
    job_dirs = sorted(list(temp_shards.glob("job_*")))
    print(f"Found {len(job_dirs)} job directories")
    
    if len(job_dirs) == 0:
        print("No job directories found! Exiting.")
        return
    
    # Copy metadata from first job (they should all be the same structure)
    first_job = job_dirs[0]
    
    # Copy info.json and stats.json
    for meta_file in ["info.json", "stats.json"]:
        src = first_job / "meta" / meta_file
        if src.exists():
            shutil.copy2(src, output_path / "meta" / meta_file)
            print(f"Copied {meta_file}")
    
    # Copy tasks.parquet if exists
    tasks_src = first_job / "meta" / "tasks.parquet"
    if tasks_src.exists():
        shutil.copy2(tasks_src, output_path / "meta" / "tasks.parquet")
        print("Copied tasks.parquet")
    
    # Process each job directory with sequential file numbering
    data_file_counter = 0
    episode_file_counter = 0
    episode_total_counter = 0
    video_file_counter = 0
    image_file_counter = 0
    
    for job_dir in tqdm(job_dirs, desc="Processing jobs"):
        job_name = job_dir.name
        
        # Copy data parquet files
        data_src = job_dir / "data"
        if data_src.exists():
            for chunk_dir in data_src.glob("chunk-*"):
                chunk_name = chunk_dir.name
                dest_chunk = output_path / "data" / chunk_name
                dest_chunk.mkdir(exist_ok=True)
                
                for parquet_file in sorted(chunk_dir.glob("*.parquet")):
                    # Use sequential numbering: file-000.parquet, file-001.parquet, etc.
                    new_name = f"file-{data_file_counter:03d}.parquet"
                    shutil.copy2(parquet_file, dest_chunk / new_name)
                    data_file_counter += 1
        
        # Process episodes metadata - READ, UPDATE INDEX, AND WRITE
        episodes_src = job_dir / "meta" / "episodes"
        if episodes_src.exists():
            for chunk_dir in sorted(episodes_src.glob("chunk-*")):
                chunk_name = chunk_dir.name
                dest_chunk = output_path / "meta" / "episodes" / chunk_name
                dest_chunk.mkdir(exist_ok=True, parents=True)
                
                for parquet_file in sorted(chunk_dir.glob("*.parquet")):
                    table = pq.read_table(parquet_file)
                    df = table.to_pandas()
                    
                    # Update episode_index and index to be globally unique
                    # Number of episodes in this file
                    num_episodes = len(df)
                    
                    # Offset the index based on how many episodes we have seen so far
                    df["episode_index"] = np.arange(episode_total_counter, episode_total_counter + num_episodes)
                    # Often 'index' reflects the same as episode_index in v3 meta
                    if "index" in df.columns:
                        df["index"] = df["episode_index"]
                    
                    # Write back updated table
                    new_table = pa.Table.from_pandas(df)
                    new_name = f"file-{episode_file_counter:03d}.parquet"
                    pq.write_table(new_table, dest_chunk / new_name)
                    
                    episode_file_counter += 1
                    episode_total_counter += num_episodes
        
        # Copy videos - handle nested video_key structure
        videos_src = job_dir / "videos"
        if videos_src.exists():\
            # Videos are organized as: videos/{video_key}/chunk-000/file-000.mp4
            for video_key_dir in videos_src.iterdir():
                if not video_key_dir.is_dir():
                    continue
                video_key = video_key_dir.name
                
                # Use per-video-key counter (each video_key starts from 0)
                video_key_counter = 0
                
                for chunk_dir in video_key_dir.glob("chunk-*"):
                    chunk_name = chunk_dir.name
                    # Create destination: videos/{video_key}/chunk-000/
                    dest_chunk = output_path / "videos" / video_key / chunk_name
                    dest_chunk.mkdir(parents=True, exist_ok=True)
                    
                    for video_file in sorted(chunk_dir.glob("*.mp4")):
                        new_name = f"file-{video_key_counter:03d}.mp4"
                        shutil.copy2(video_file, dest_chunk / new_name)
                        video_key_counter += 1
        
        # Copy images if they exist - handle nested image_key structure
        images_src = job_dir / "images"
        if images_src.exists():
            # Images might be organized similarly: images/{image_key}/chunk-000/
            for image_key_dir in images_src.iterdir():
                if not image_key_dir.is_dir():
                    continue
                image_key = image_key_dir.name
                
                # Use per-image-key counter (each image_key starts from 0)
                image_key_counter = 0
                
                for chunk_dir in image_key_dir.glob("chunk-*"):
                    chunk_name = chunk_dir.name
                    dest_chunk = output_path / "images" / image_key / chunk_name
                    dest_chunk.mkdir(parents=True, exist_ok=True)
                    
                    for image_file in sorted(chunk_dir.glob("*.png")):
                        new_name = f"file-{image_key_counter:03d}.png"
                        shutil.copy2(image_file, dest_chunk / new_name)
                        image_key_counter += 1
    
    print(f"\n✓ Reformatting complete for {dataset_name}!")
    print(f"Output directory: {output_path}")
    
    # Print summary
    data_files = list((output_path / "data").rglob("*.parquet"))
    episode_files = list((output_path / "meta" / "episodes").rglob("*.parquet"))
    video_files = list((output_path / "videos").rglob("*.mp4"))
    image_files = list((output_path / "images").rglob("*.png"))
    
    print(f"\nSummary:")
    print(f"  Data parquet files: {len(data_files)}")
    print(f"  Episode parquet files: {len(episode_files)}")
    print(f"  Video files: {len(video_files)}")
    print(f"  Image files: {len(image_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reformat data to LeRobot v3 format")
    parser.add_argument("--source-dir", type=str, required=True, 
                        help="Source directory containing _temp_shards")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for LeRobot v3 formatted data")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Name of the dataset (e.g., 'bridge_openx', 'droid_100')")
    
    args = parser.parse_args()
    
    reformat_to_lerobot_v3(args.source_dir, args.output_dir, args.dataset_name)
