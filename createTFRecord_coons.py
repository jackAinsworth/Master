"""

nohup python createTFRecord_coons.py \
  --csv_file=dataset_coons/batch_filepaths.csv \
  --out_dir=tfrecords_coons \
  --num_shards=69 \
  --start_index=167 \
  > logs/tfrecords_coons.log 2>&1 &
"""


import os
import sys
import math
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

def load_and_process_surface_data(data_file,
                                  max_ctrlpts_u=12,
                                  max_ctrlpts_v=12,
                                  num_samples=50,
                                  pad_value=-10.0):
    """
    Loads a single .pkl or .npz file and returns two NumPy arrays:
      - X: shape [N_i, num_samples, num_samples, 3]
      - Y: shape [N_i, max_ctrlpts_u, max_ctrlpts_v, 3]
    where N_i = number of surfaces in that file.
    """
    if data_file.endswith('.pkl'):
        with open(data_file, "rb") as f:
            loaded = pickle.load(f)
            training_data = loaded['data']
    else:
        loaded = np.load(data_file, allow_pickle=True)
        if isinstance(loaded, dict) and "training_data" in loaded:
            training_data = loaded["data"]["training_data"]
        else:
            training_data = loaded

    X_list, Y_list = [], []
    for item in training_data:
        noisy = np.array(item['points'], dtype=np.float32)
        ctrl_net = np.array(item['control_net'], dtype=np.float32)
        current_u, current_v, _ = ctrl_net.shape

        # pad/truncate in the u-dimension
        if current_u < max_ctrlpts_u:
            pad_u = np.full((max_ctrlpts_u - current_u, current_v, 3),
                            pad_value, dtype=np.float32)
            ctrl_net = np.concatenate([ctrl_net, pad_u], axis=0)
        else:
            ctrl_net = ctrl_net[:max_ctrlpts_u, :, :]

        # pad/truncate in the v-dimension
        if current_v < max_ctrlpts_v:
            pad_v = np.full((max_ctrlpts_u, max_ctrlpts_v - current_v, 3),
                            pad_value, dtype=np.float32)
            ctrl_net = np.concatenate([ctrl_net, pad_v], axis=1)
        else:
            ctrl_net = ctrl_net[:, :max_ctrlpts_v, :]

        X_list.append(noisy)
        Y_list.append(ctrl_net)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y


def _bytes_feature(value):
    """Returns a bytes_list from a single string / byte array."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(noisy_array: np.ndarray, ctrl_array: np.ndarray) -> bytes:
    """
    Creates a tf.train.Example message ready to be written to a file.

    We serialize each NumPy array using tf.io.serialize_tensor(...) so we can reconstruct
    its shape automatically. During parsing, we'll call tf.io.parse_tensor(...).

    Args:
      noisy_array:  numpy array of shape (50,50,3), dtype float32
      ctrl_array:   numpy array of shape (12,12,3), dtype float32

    Returns:
      A byte string: the serialized Example.
    """
    noisy_tensor = tf.io.serialize_tensor(noisy_array)
    ctrl_tensor  = tf.io.serialize_tensor(ctrl_array)

    feature = {
        "noisy_raw": _bytes_feature(noisy_tensor),
        "ctrl_raw":  _bytes_feature(ctrl_tensor),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecord_shards(csv_file: str,
                          output_dir: str,
                          num_shards: int = 100,
                          max_ctrlpts_u: int = 12,
                          max_ctrlpts_v: int = 12,
                          num_samples: int = 50,
                          pad_value: float = -0.2,
                          start_index: int = 0,
                          start_shard: int = 0):
    """
    Reads 'csv_file' (with a column 'filepath' listing .pkl/.npz paths),
    skips the first 'start_index' examples, splits the remaining examples into
    'num_shards' roughly equal shards, and writes TFRecord files named:
       output_dir/data_{start_shard:04d}-of-{start_shard+num_shards:04d}.tfrecord
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    if "filepath" not in df.columns:
        raise ValueError("CSV must have a 'filepath' column")

    all_paths = df["filepath"].tolist()

    print("Counting total number of examples...")
    total_examples = 0
    per_file_counts = []
    for fp in all_paths:
        X_i, _ = load_and_process_surface_data(fp,
                                               max_ctrlpts_u=max_ctrlpts_u,
                                               max_ctrlpts_v=max_ctrlpts_v,
                                               num_samples=num_samples,
                                               pad_value=pad_value)
        n_i = X_i.shape[0]
        per_file_counts.append(n_i)
        total_examples += n_i

    if total_examples == 0:
        raise RuntimeError("No examples found in any file listed in the CSV.")

    # Skip already-processed examples
    if start_index > 0:
        if start_index >= total_examples:
            raise ValueError(f"--start_index ({start_index}) >= total examples ({total_examples})")
        print(f"→ Skipping first {start_index} examples (already processed)")
        remaining = total_examples - start_index
    else:
        remaining = total_examples

    # Compute shard sizes for the remaining examples
    base  = remaining // num_shards
    extra = remaining % num_shards

    shard_boundaries = []
    idx = start_index
    for shard_id in range(num_shards):
        count_here = base + (1 if shard_id < extra else 0)
        shard_boundaries.append((idx, idx + count_here))
        idx += count_here

    print("Writing TFRecord shards ... this may take a while.")
    writer_list = []
    total_shards = start_shard + num_shards
    for shard_id, (start, end) in enumerate(shard_boundaries):
        global_shard = start_shard + shard_id
        shard_name   = f"data_{global_shard:04d}-of-{total_shards:04d}.tfrecord"
        shard_path   = os.path.join(output_dir, shard_name)
        writer_list.append(tf.io.TFRecordWriter(shard_path))

    example_index = 0
    file_idx = 0

    for fp, n_i in zip(all_paths, per_file_counts):
        X_i, Y_i = load_and_process_surface_data(fp,
                                                 max_ctrlpts_u=max_ctrlpts_u,
                                                 max_ctrlpts_v=max_ctrlpts_v,
                                                 num_samples=num_samples,
                                                 pad_value=pad_value)
        for j in range(n_i):
            # Skip until we reach start_index
            if example_index < start_index:
                example_index += 1
                continue

            # Move to correct shard
            while not (shard_boundaries[file_idx][0] <= example_index < shard_boundaries[file_idx][1]):
                file_idx += 1

            writer = writer_list[file_idx]
            writer.write(serialize_example(X_i[j], Y_i[j]))
            example_index += 1

        sys.stdout.write(f"  → Processed file '{fp}', total so far: {example_index}/{total_examples}\r")
        sys.stdout.flush()

    for w in writer_list:
        w.close()

    print("\nAll shards written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert your surfaces into TFRecord shards")
    parser.add_argument("--csv_file",   type=str, required=True,
                        help="CSV file listing .pkl/.npz paths (column: 'filepath')")
    parser.add_argument("--out_dir",    type=str, required=True,
                        help="Directory where TFRecord shards will be saved")
    parser.add_argument("--num_shards", type=int, default=100,
                        help="How many TFRecord shards to produce")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Skip the first N examples (so you can re-shard the rest)")
    parser.add_argument("--start_shard", type=int, default=0,
                        help="Offset the shard numbering by this amount")
    args = parser.parse_args()

    write_tfrecord_shards(csv_file=args.csv_file,
                          output_dir=args.out_dir,
                          num_shards=args.num_shards,
                          start_index=args.start_index,
                          start_shard=args.start_shard)
