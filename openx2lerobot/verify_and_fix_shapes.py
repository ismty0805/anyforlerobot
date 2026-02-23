import json
import subprocess
from pathlib import Path

def collect_datasets(base):
    results = []
    for top in base.iterdir():
        if not top.is_dir():
            continue
        if (top / "meta/info.json").exists():
            results.append(top)
        else:
            # Check subdirectories (like in agibot, galaxea, openx_lerobot)
            for sub in top.iterdir():
                if sub.is_dir() and (sub / "meta/info.json").exists():
                    results.append(sub)
    return results

all_bases = [
    Path("/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset"),
    Path("/fsx/ubuntu/taeyoung/data/processing/vla_pretrain_dataset/openx_lerobot")
]

datasets_paths = []
for b in all_bases:
    datasets_paths.extend(collect_datasets(b))

# Remove duplicates if any (though unlikely with current structure)
datasets_paths = sorted(list(set(datasets_paths)))

def get_video_info(video_path):
    ffprobe_path = "/fsx/ubuntu/miniconda3/envs/convert/bin/ffprobe"
    cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name",
        "-of", "json",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        data = json.loads(result.stdout)
        if "streams" in data and len(data["streams"]) > 0:
            return data["streams"][0]
    return None

def process_dataset(ds_path):
    info_json_path = ds_path / "meta" / "info.json"
    
    if not info_json_path.exists():
        return

    with open(info_json_path, 'r') as f:
        info = json.load(f)

    print(f"\nProcessing {ds_path.name}...")
    
    video_keys = [k for k, v in info['features'].items() if v.get('dtype') == 'video']
    
    for vk in video_keys:
        feature = info['features'][vk]
        current_shape = feature.get('shape')
        current_codec = feature.get('info', {}).get('video.codec')
        
        # Try to find a representative video file.
        # Structure can be: videos/<camera>/chunk-xxx/episode_xxx.mp4
        # OR: videos/chunk-xxx/<camera>/episode_xxx.mp4
        # OR: videos/episode_xxx.mp4
        
        video_files = []
        # Pattern 1: videos/<vk>/**/*.mp4
        video_files.extend(list(ds_path.glob(f"videos/{vk}/**/*.mp4")))
        # Pattern 2: videos/**/<vk>/*.mp4
        if not video_files:
            video_files.extend(list(ds_path.glob(f"videos/**/{vk}/*.mp4")))
        # Pattern 3: any mp4 if singular
        if not video_files:
            video_files.extend(list(ds_path.glob("videos/**/*.mp4")))
        
        if not video_files:
            print(f"  Warning: No video files found for {vk}")
            continue
        
        sample_video = video_files[0]
        vinfo = get_video_info(sample_video)
        
        if vinfo:
            actual_w = vinfo['width']
            actual_h = vinfo['height']
            actual_codec = vinfo['codec_name']
            
            print(f"  Feature: {vk}")
            print(f"    Current Shape: {current_shape}, Codec: {current_codec}")
            print(f"    Actual Shape: [{actual_h}, {actual_w}], Codec: {actual_codec}")
            
            # Update info.json logic (in memory for now, will apply if valid)
            updated = False
            
            missing_shape = False
            if current_shape is None or not isinstance(current_shape, list) or len(current_shape) < 2:
                missing_shape = True
            elif current_shape[0] in (None, 0) or current_shape[1] in (None, 0):
                missing_shape = True
                
            if missing_shape:
                if isinstance(current_shape, list) and len(current_shape) == 3:
                    feature['shape'] = [actual_h, actual_w, current_shape[2]]
                else:
                    feature['shape'] = [actual_h, actual_w]
                    
                if 'info' not in feature:
                    feature['info'] = {}
                feature['info']['video.height'] = actual_h
                feature['info']['video.width'] = actual_w
                updated = True
                
            if actual_codec != current_codec:
                if 'info' in feature:
                    feature['info']['video.codec'] = actual_codec
                updated = True
            
            if updated:
                print(f"    -> Updated in memory")
        else:
            print(f"  Error: Could not probe video {sample_video}")

    # Write back
    with open(info_json_path, 'w') as f:
        json.dump(info, f, indent=4)
    print(f"  Done updating {info_json_path}")

for ds_p in datasets_paths:
    process_dataset(ds_p)
