#!/usr/bin/env python3
# create_tfrecords_per_batch_static.py
import os, sys, pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"
import numpy as np
import pandas as pd
import tensorflow as tf

# ── Fixed settings taken from our chat ─────────────────────────────────────────
CSV_FILE         = "dataset/filepaths.csv"
OUT_DIR          = "dataset/tfrecords_ruled"
SHARDS_PER_BATCH = 25          # e.g., 500000 examples/file -> ~10,000 per shard
NUM_SAMPLES      = 50         # your generator commonly uses 161×161
MAX_CTRLPTS_U    = 12
MAX_CTRLPTS_V    = 12
PAD_VALUE        = -0.1 # -0.2

os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(noisy_array: np.ndarray, ctrl_array: np.ndarray) -> bytes:
    noisy_tensor = tf.io.serialize_tensor(noisy_array.astype(np.float32))
    ctrl_tensor  = tf.io.serialize_tensor(ctrl_array.astype(np.float32))
    feat = {"noisy_raw": _bytes_feature(noisy_tensor),
            "ctrl_raw":  _bytes_feature(ctrl_tensor)}
    return tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString()

def _pad_or_truncate_ctrl(ctrl, max_u, max_v, pad_value=-0.2):
    u, v, c = ctrl.shape
    assert c == 3, f"control-net channels must be 3, got {c}"
    # U
    if u < max_u:
        ctrl = np.concatenate([ctrl, np.full((max_u - u, v, 3), pad_value, np.float32)], axis=0)
        u = max_u
    else:
        ctrl = ctrl[:max_u]; u = max_u
    # V
    if v < max_v:
        ctrl = np.concatenate([ctrl, np.full((u, max_v - v, 3), pad_value, np.float32)], axis=1)
    else:
        ctrl = ctrl[:, :max_v]
    return ctrl

def _load_data_list_from_pkl(path):
    # NOTE: This loads the entire pickle list into RAM because of how the data was saved.
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        if "data" not in obj:
            raise ValueError(f"{path}: pickle dict missing 'data' (keys={list(obj.keys())})")
        data = obj["data"]
    elif isinstance(obj, (list, tuple)):
        data = obj
    else:
        raise ValueError(f"{path}: unsupported pickle root {type(obj)}")
    if not isinstance(data, (list, tuple)):
        raise ValueError(f"{path}: 'data' must be list/tuple, got {type(data)}")
    return data

def _compute_shard_plan(total, shards):
    base = total // shards
    extra = total % shards
    return [base + (1 if i < extra else 0) for i in range(shards)]

def write_shards_for_batch_file(pkl_path: str, out_dir: str, batch_index: int, shards_per_batch: int):
    print(f"[Batch {batch_index}] Loading {pkl_path} ...")
    data = _load_data_list_from_pkl(pkl_path)  # may be large
    total = len(data)
    if total == 0:
        print(f"[Batch {batch_index}] WARNING: empty file; skipping.")
        return 0

    counts = _compute_shard_plan(total, shards_per_batch)
    print(f"[Batch {batch_index}] total={total}, shards={shards_per_batch}, "
          f"avg~{total//shards_per_batch} (+1 for first {total%shards_per_batch})")

    example_cursor = 0
    written_total = 0

    # Keep only one writer open at a time to keep FD usage low
    for shard_id, n_this in enumerate(counts):
        shard_name = f"data_batch_new{batch_index:03d}_{shard_id:04d}-of-{shards_per_batch:04d}.tfrecord"
        shard_path = os.path.join(out_dir, shard_name)
        writer = tf.io.TFRecordWriter(shard_path)

        end_cursor = example_cursor + n_this
        while example_cursor < end_cursor:
            item = data[example_cursor]
            try:
                noisy = np.asarray(item["rotated_noisy_points"], np.float32)
                ctrl  = np.asarray(item["rotated_control_net"],  np.float32)
            except KeyError as e:
                writer.close()
                raise ValueError(f"{pkl_path}: item {example_cursor} missing {e!s}")

            # sanity checks
            if noisy.ndim != 3 or noisy.shape[2] != 3:
                writer.close()
                raise ValueError(f"{pkl_path}: item {example_cursor} noisy shape {noisy.shape} invalid")
            if noisy.shape[0] != NUM_SAMPLES or noisy.shape[1] != NUM_SAMPLES:
                writer.close()
                raise ValueError(f"{pkl_path}: item {example_cursor} noisy grid {noisy.shape[:2]} "
                                 f"!= expected ({
                                 NUM_SAMPLES},{NUM_SAMPLES}). "
                                 f"Adjust NUM_SAMPLES at top of this file.")

            ctrl = _pad_or_truncate_ctrl(ctrl, MAX_CTRLPTS_U, MAX_CTRLPTS_V, PAD_VALUE)
            writer.write(serialize_example(noisy, ctrl))

            example_cursor += 1
            written_total += 1

        writer.close()
        print(f"[Batch {batch_index}] wrote shard {shard_id+1}/{shards_per_batch} "
              f"({n_this} ex) -> {shard_name}")

    print(f"[Batch {batch_index}] DONE: {written_total} examples")
    return written_total

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"CONFIG -> CSV_FILE={CSV_FILE} | OUT_DIR={OUT_DIR} | SHARDS_PER_BATCH={SHARDS_PER_BATCH} | "
          f"NUM_SAMPLES={NUM_SAMPLES} | MAX_CTRLPTS_U={MAX_CTRLPTS_U} | MAX_CTRLPTS_V={MAX_CTRLPTS_V} | PAD_VALUE={PAD_VALUE}")

    df = pd.read_csv(CSV_FILE)
    if "filepath" not in df.columns:
        raise SystemExit("CSV must contain a 'filepath' column")

    total_written = 0
    for batch_idx, pkl_path in enumerate(df["filepath"].tolist()):
        total_written += write_shards_for_batch_file(
            pkl_path=pkl_path,
            out_dir=OUT_DIR,
            batch_index=batch_idx,
            shards_per_batch=SHARDS_PER_BATCH,
        )

    print(f"ALL DONE — total examples written: {total_written}")

if __name__ == "__main__":
    # unbuffered print so logs flush in nohup/tee
    os.environ["PYTHONUNBUFFERED"] = "1"
    main()
