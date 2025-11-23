import os
import sys
import math
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# ── Revised load_and_process_surface_data for the new data structure ──
def load_and_process_surface_data(data_file,
                                  max_ctrlpts_u=10,
                                  max_ctrlpts_v=10,
                                  num_samples=35,
                                  pad_value=-10.0):
    """
    Loads a single .pkl or .npz file and returns two NumPy arrays:
      - X: shape [N_i, num_samples, num_samples, 3]
      - Y: shape [N_i, max_ctrlpts_u, max_ctrlpts_v, 3]
    where N_i = number of surfaces in that file.

    The new dataset entries use keys:
      - 'points'      : (num_samples, num_samples, 3)
      - 'control_net' : (current_u, current_v, 3)
    """
    if data_file.endswith('.pkl'):
        with open(data_file, "rb") as f:
            loaded = pickle.load(f)
            # The top‐level dict has 'data' and 'configuration_options'
            training_data = loaded['data']
    else:
        # If it’s .npz, assume a similar structure or a direct array‐of‐dicts
        loaded = np.load(data_file, allow_pickle=True)
        if isinstance(loaded, dict) and "data" in loaded:
            training_data = loaded["data"]
        else:
            training_data = loaded  # fall back if it's directly stored

    X_list, Y_list = [], []
    for item in training_data:
        # Fetch new keys 'points' and 'control_net'
        noisy = np.array(item['points'], dtype=np.float32)        # shape (num_samples, num_samples, 3)
        ctrl_net = np.array(item['control_net'], dtype=np.float32)  # shape (current_u, current_v, 3)
        current_u, current_v, _ = ctrl_net.shape

        # Pad or truncate in the u‐dimension
        if current_u < max_ctrlpts_u:
            pad_u = np.full((max_ctrlpts_u - current_u, current_v, 3),
                            pad_value, dtype=np.float32)
            ctrl_net = np.concatenate([ctrl_net, pad_u], axis=0)
        else:
            ctrl_net = ctrl_net[:max_ctrlpts_u, :, :]

        # Pad or truncate in the v‐dimension
        if current_v < max_ctrlpts_v:
            pad_v = np.full((max_ctrlpts_u, max_ctrlpts_v - current_v, 3),
                            pad_value, dtype=np.float32)
            ctrl_net = np.concatenate([ctrl_net, pad_v], axis=1)
        else:
            ctrl_net = ctrl_net[:, :max_ctrlpts_v, :]

        X_list.append(noisy)
        Y_list.append(ctrl_net)

    X = np.array(X_list, dtype=np.float32)  # shape = [N_i, num_samples, num_samples, 3]
    Y = np.array(Y_list, dtype=np.float32)  # shape = [N_i, max_ctrlpts_u, max_ctrlpts_v, 3]
    return X, Y


def _bytes_feature(value):
    """Returns a bytes_list from a single string / byte array."""
    if isinstance(value, type(tf.constant(0))):  # if value is a Tensor
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(noisy_array: np.ndarray, ctrl_array: np.ndarray) -> bytes:
    """
    Creates a tf.train.Example message ready to be written to a file.

    We serialize each NumPy array using tf.io.serialize_tensor(...) so we can reconstruct
    its shape automatically. During parsing, we'll call tf.io.parse_tensor(...).

    Args:
      noisy_array:  numpy array of shape (num_samples, num_samples, 3), dtype float32
      ctrl_array:   numpy array of shape (max_ctrlpts_u, max_ctrlpts_v, 3), dtype float32

    Returns:
      A byte string: the serialized Example.
    """
    noisy_tensor = tf.io.serialize_tensor(noisy_array)  # dtype=float32
    ctrl_tensor = tf.io.serialize_tensor(ctrl_array)    # dtype=float32

    feature = {
        "noisy_raw": _bytes_feature(noisy_tensor),
        "ctrl_raw": _bytes_feature(ctrl_tensor),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecord_shards(csv_file: str,
                          output_dir: str,
                          num_shards: int = 100,
                          max_ctrlpts_u: int = 10,
                          max_ctrlpts_v: int = 10,
                          num_samples: int = 35,
                          pad_value: float = -10.0):
    """
    Reads 'csv_file' (with a column 'filepath' listing .pkl/.npz paths),
    splits the total_examples into 'num_shards' roughly equal shards,
    and writes TFRecord files named
       output_dir/data_0000-of-0100.tfrecord
       output_dir/data_0001-of-0100.tfrecord
       ...
       output_dir/data_0099-of-0100.tfrecord

    Each Example has two bytes features ("noisy_raw" and "ctrl_raw").
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    if "filepath" not in df.columns:
        raise ValueError("CSV must have a 'filepath' column")

    all_paths = df["filepath"].tolist()

    # First pass: count total examples
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

    print(f"Total examples across all files: {total_examples}")

    # Compute how many examples go into each shard (floor division),
    # and compute remainders so final shards are balanced.
    base = total_examples // num_shards
    extra = total_examples % num_shards  # first 'extra' shards will have (base+1) examples

    shard_boundaries = []
    start_idx = 0
    for shard_id in range(num_shards):
        count_here = base + (1 if shard_id < extra else 0)
        end_idx = start_idx + count_here
        shard_boundaries.append((start_idx, end_idx))
        start_idx = end_idx

    # Now do a second pass: iterate over all examples in order
    print("Writing TFRecord shards ... this may take a while.")
    writer_list = []
    for shard_id, (start, end) in enumerate(shard_boundaries):
        shard_name = f"data_{shard_id:04d}-of-{num_shards:04d}.tfrecord"
        shard_path = os.path.join(output_dir, shard_name)
        writer = tf.io.TFRecordWriter(shard_path)
        writer_list.append(writer)

    example_index = 0  # global index from 0..total_examples-1
    shard_id_ptr = 0

    for fp, n_i in zip(all_paths, per_file_counts):
        X_i, Y_i = load_and_process_surface_data(fp,
                                                 max_ctrlpts_u=max_ctrlpts_u,
                                                 max_ctrlpts_v=max_ctrlpts_v,
                                                 num_samples=num_samples,
                                                 pad_value=pad_value)
        # For each example in this file:
        for j in range(n_i):
            # Advance shard_id_ptr until example_index falls in its boundary
            while not (shard_boundaries[shard_id_ptr][0] <= example_index < shard_boundaries[shard_id_ptr][1]):
                shard_id_ptr += 1

            shard_writer = writer_list[shard_id_ptr]
            serialized = serialize_example(X_i[j], Y_i[j])
            shard_writer.write(serialized)
            example_index += 1

        # small progress print
        sys.stdout.write(
            f"  → Processed file {os.path.basename(fp)!r}, "
            f"wrote {n_i} examples, total so far: {example_index}/{total_examples}\r"
        )
        sys.stdout.flush()

    # Close all writers
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
                        help="How many TFRecord shards to produce (e.g. 100)")
    args = parser.parse_args()

    write_tfrecord_shards(csv_file=args.csv_file,
                          output_dir=args.out_dir,
                          num_shards=args.num_shards,
                          max_ctrlpts_u=12,
                          max_ctrlpts_v=12,
                          num_samples=50,
                          pad_value=-10.0)
