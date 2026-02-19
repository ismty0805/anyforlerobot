import tensorflow as tf
import tensorflow_datasets as tfds
import os

dataset_name = "humanoid_everyday_28action"
data_dir = "/fsx/ubuntu/taeyoung/data/rlds/"
version = "1.0.0"

def test_shard(shard_idx):
    print(f"Testing shard {shard_idx}...")
    split_arg = f"train[{shard_idx}shard:{shard_idx+1}shard]"
    builder = tfds.builder(dataset_name, data_dir=data_dir, version=version)
    ds = builder.as_dataset(split=split_arg)
    
    count = 0
    for episode in ds:
        print(f"  Episode {count}")
        steps = episode["steps"]
        try:
            # Try normal batch
            _ = next(iter(steps.batch(50000)))
            print(f"    Normal batch success")
        except Exception as e:
            print(f"    Normal batch failed: {e}")
            try:
                # Try padded batch
                # To use padded_batch, we need to know the shapes or use default (None)
                # But padded_batch(50000) on a dataset of structures can be tricky
                # We can use tf.data.experimental.get_structure(steps) but that's complex
                # Simply try to iterate and see where it fails
                step_count = 0
                for step in steps:
                    step_count += 1
                print(f"    Manual iteration success: {step_count} steps")
            except Exception as e2:
                print(f"    Manual iteration failed: {e2}")
        count += 1

test_shard(131)
test_shard(142)
