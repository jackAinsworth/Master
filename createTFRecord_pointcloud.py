#!/usr/bin/env python3
"""
write_tfrecords_from_csv.py

Reads a CSV with column “filepath” pointing to your dataset .pkl files
(each containing {'data': [entries], …} where each entry has
  "rotated_noisy_points" and "rotated_control_net"),
and writes N balanced TFRecord shards with features "xyz_raw" & "ctrl_raw".

Usage:
  python createTFRecord_pointcloud.py \
      --csv_file ./pointcloud/data.csv \
      --out_dir ./tfrecords_pointcloud \
      --num_shards 100 \
      --start_index 0
"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf


def load_dataset_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _bytes_feature(b: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))


def serialize_example(points: np.ndarray, ctrl_net: np.ndarray) -> bytes:
    xyz_b  = tf.io.serialize_tensor(points).numpy()
    ctrl_b = tf.io.serialize_tensor(ctrl_net).numpy()
    feat   = {
        "xyz_raw":  _bytes_feature(xyz_b),
        "ctrl_raw": _bytes_feature(ctrl_b),
    }
    return tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString()


def pad_truncate_ctrl(ctrl: np.ndarray, max_u: int, max_v: int, pad_value: float) -> np.ndarray:
    curr_u, curr_v, _ = ctrl.shape
    # pad or truncate U dimension
    if curr_u < max_u:
        pad_u = np.full((max_u - curr_u, curr_v, 3), pad_value, dtype=np.float32)
        ctrl = np.concatenate([ctrl, pad_u], axis=0)
    else:
        ctrl = ctrl[:max_u, :, :]
    # pad or truncate V dimension
    if curr_v < max_v:
        pad_v = np.full((max_u, max_v - curr_v, 3), pad_value, dtype=np.float32)
        ctrl = np.concatenate([ctrl, pad_v], axis=1)
    else:
        ctrl = ctrl[:, :max_v, :]
    return ctrl


def main(csv_file: str, out_dir: str, num_shards: int, start_index: int,
         max_ctrl_u: int, max_ctrl_v: int, pad_value: float):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Read manifest
    df = pd.read_csv(csv_file)
    if 'batch_filepath' not in df.columns:
        raise ValueError("CSV must have a 'batch_filepath' column")
    paths = df['batch_filepath'].tolist()

    # 2) Count total examples
    counts = [len(load_dataset_pkl(p)) for p in paths]
    total = sum(counts)
    if start_index >= total:
        raise ValueError(f"--start_index {start_index} >= total examples {total}")
    remaining = total - start_index
    print(f"Found {len(paths)} files, {total} examples total, skipping first {start_index}, writing {remaining}")

    # 3) Compute balanced shard ranges
    base, extra = divmod(remaining, num_shards)
    bounds = []
    lo = start_index
    for i in range(num_shards):
        cnt = base + (1 if i < extra else 0)
        bounds.append((lo, lo + cnt))
        lo += cnt

    # 4) Open TFRecord writers
    writers = [
        tf.io.TFRecordWriter(os.path.join(
            out_dir, f"data_{i:04d}-of-{num_shards:04d}.tfrecord"))
        for i in range(num_shards)
    ]

    # 5) Stream, pad, and shard
    idx = 0      # global example index
    shard = 0
    written = 0
    for p, cnt in zip(paths, counts):
        entries = load_dataset_pkl(p)
        for entry in entries:
            if idx < start_index:
                idx += 1
                continue
            # advance to correct shard
            while not (bounds[shard][0] <= idx < bounds[shard][1]):
                shard += 1
            pts  = np.array(entry["rotated_noisy_points"], dtype=np.float32)
            ctrl = np.array(entry["rotated_control_net"], dtype=np.float32)
            # pad/truncate control net
            ctrl = pad_truncate_ctrl(ctrl, max_ctrl_u, max_ctrl_v, pad_value)

            writers[shard].write(serialize_example(pts, ctrl))
            written += 1
            if written % 1000 == 0 or written == remaining:
                print(f"\rWritten {written}/{remaining} examples", end="")
            idx += 1
    print()

    for w in writers:
        w.close()
    print(f"Done – wrote {written} examples into {num_shards} shards in '{out_dir}'")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Shard .pkl dataset into padded TFRecord shards")
    p.add_argument("--csv_file", "-i", required=True,
                   help="CSV manifest with column 'batch_filepath'")
    p.add_argument("--out_dir", "-o", required=True,
                   help="Directory to write TFRecord shards into")
    p.add_argument("--num_shards", "-n", type=int, default=100,
                   help="Number of TFRecord shards to produce")
    p.add_argument("--start_index", "-s", type=int, default=0,
                   help="Skip the first N examples")
    p.add_argument("--max_ctrl_u", type=int, default=10,
                   help="Max control points in U dimension")
    p.add_argument("--max_ctrl_v", type=int, default=10,
                   help="Max control points in V dimension")
    p.add_argument("--pad_value", type=float, default=-10.0,
                   help="Padding value for missing control points")
    args = p.parse_args()
    main(args.csv_file, args.out_dir, args.num_shards, args.start_index,
         args.max_ctrl_u, args.max_ctrl_v, args.pad_value)
