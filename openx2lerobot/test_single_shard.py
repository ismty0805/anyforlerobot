import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from others_native_worker import transform_raw_dataset, OTHERS_DATASET_CONFIGS
from others_utils.transforms import OTHERS_STANDARDIZATION_TRANSFORMS
import argparse

def test(dataset_name, data_dir):
    print(f"Testing {dataset_name}...")
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    ds = builder.as_dataset(split="train[:1shard]")
    
    # Try mapping
    ds = ds.map(lambda e: transform_raw_dataset(e, dataset_name))
    
    try:
        for i, episode in enumerate(ds.take(1)):
            print(f"Successfully processed episode {i}")
            print(f"Observation keys: {episode['observation'].keys()}")
            print(f"Action shape: {episode['action'].shape}")
            if 'original_observation' in episode:
                print(f"Original Observation keys: {episode['original_observation'].keys()}")
    except Exception as e:
        print(f"FAILED {dataset_name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    test(args.dataset, args.path)
