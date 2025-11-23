#!/usr/bin/env python3
"""
PointNet++‑style Surface Reconstruction
======================================

This script trains a regression model that maps an **un‑ordered point‑cloud**
(`NUM_POINTS × 3`) to a grid of B‑spline control‑points
(`MAX_CTRLS_U × MAX_CTRLS_V × 3`).

The architecture follows the spirit of *PointNet++* (hierarchical point‑set
encoding) but is implemented entirely with standard TensorFlow / Keras layers
so it runs out‑of‑the‑box—no custom CUDA/TF ops required.  The trade‑off is
that we approximate set‑abstraction with shared `Conv1D` MLPs + global pooling
instead of explicit **sampling & grouping** kernels; you can later swap our
`PointNetBackbone` for a full PointNet++ implementation if you have the
compiled ops available.

The training loop mirrors the workflow you used for the Inception‑based
surface model so you can run both scripts side‑by‑side and compare.


python pointnet2_recon_model.py \
  --tfrecord_glob "./tfrecords_pointcloud/data_*.tfrecord" \
  --epochs 300



"""

import os
import datetime
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, CSVLogger)

# ──────────────────────────────────────────────────────────────────────────────
# Constants (tweak to match your dataset)
# ──────────────────────────────────────────────────────────────────────────────
NUM_POINTS   = 35*35     # points per cloud (must match TFRecord)
MAX_CTRLS_U  = 10       # B‑spline control net resolution (U)
MAX_CTRLS_V  = 10       # B‑spline control net resolution (V)
BATCH_SIZE   = 500
SHUFFLE_BUF  = 50_000
AUTOTUNE     = tf.data.AUTOTUNE
NUM_EXAMPLES = 1_660_000

# Features stored in TFRecord (adjust if you changed names)
FEATURE_DESC = {
    "xyz_raw" : tf.io.FixedLenFeature([], tf.string),
    "ctrl_raw": tf.io.FixedLenFeature([], tf.string),
}

# ──────────────────────────────────────────────────────────────────────────────
# Data Pipeline helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_function(example_proto):
    """Parse one TFExample → (point_cloud, control_net)."""
    parsed = tf.io.parse_single_example(example_proto, FEATURE_DESC)

    pts  = tf.io.parse_tensor(parsed["xyz_raw"], out_type=tf.float32)
    pts  = tf.reshape(pts, (NUM_POINTS, 3))            # (N,3)

    ctrl = tf.io.parse_tensor(parsed["ctrl_raw"], out_type=tf.float32)
    ctrl = tf.reshape(ctrl, (MAX_CTRLS_U, MAX_CTRLS_V, 3))  # (U,V,3)
    return pts, ctrl


def make_dataset_from_tfrecords(tfrecord_glob_pattern: str,
                                shuffle: bool = True) -> tf.data.Dataset:
    files = tf.io.gfile.glob(tfrecord_glob_pattern)
    if not files:
        raise ValueError(f"No TFRecord files match pattern {tfrecord_glob_pattern}")

    files = tf.random.shuffle(files)
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.interleave(lambda fn: tf.data.TFRecordDataset(fn),
                       cycle_length=AUTOTUNE,
                       num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=SHUFFLE_BUF,
                        reshuffle_each_iteration=True)

    ds = ds.map(_parse_function, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# ──────────────────────────────────────────────────────────────────────────────
# Loss & regularisers (same as your Inception model)
# ──────────────────────────────────────────────────────────────────────────────

def laplacian(grid):
    """4‑neighbour Laplacian per channel (B,H,W,C) → same shape."""
    lap_k = tf.constant([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]], tf.float32)
    lap_k = lap_k[..., tf.newaxis, tf.newaxis]  # (3,3,1,1)
    C = tf.shape(grid)[-1]
    lap_k = tf.tile(lap_k, [1, 1, C, 1])        # (3,3,C,1)
    return tf.nn.depthwise_conv2d(grid, lap_k,
                                  strides=[1,1,1,1], padding='SAME')


def total_loss(y_true, y_pred, w_lap: float = 0.10):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    lap = tf.reduce_mean(tf.square(laplacian(y_true) - laplacian(y_pred)))
    return mse + w_lap * lap

# ──────────────────────────────────────────────────────────────────────────────
# PointNet++‑style backbone implemented with shared MLPs
# ──────────────────────────────────────────────────────────────────────────────

def build_pointnet2_surface_model(num_points: int = NUM_POINTS,
                                  output_ctrlpts_u: int = MAX_CTRLS_U,
                                  output_ctrlpts_v: int = MAX_CTRLS_V):
    """Minimal PointNet/PointNet++ backbone → control‑point regression head."""

    inputs = layers.Input(shape=(num_points, 3))  # un‑ordered point set
    print(inputs.shape)
    # --- Shared MLP over points ------------------------------------------------
    x = layers.Conv1D(64, 1, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # --- Global feature --------------------------------------------------------
    x = layers.GlobalMaxPooling1D()(x)           # (B, 256)

    # --- Fully‑connected regression head --------------------------------------
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    dense_units = output_ctrlpts_u * output_ctrlpts_v * 3
    ctrl_pred   = layers.Dense(dense_units, activation='linear', name='ctrl')(x)
    ctrl_pred   = layers.Reshape((output_ctrlpts_u, output_ctrlpts_v, 3),
                                 name='ctrl_net')(ctrl_pred)

    return models.Model(inputs=inputs, outputs=ctrl_pred,
                        name="pointnet2_surface_ctrl")

# ──────────────────────────────────────────────────────────────────────────────
# Training script (mirrors your earlier main())
# ──────────────────────────────────────────────────────────────────────────────

def main(tfrecord_path_pattern: str, epochs: int):
    all_files = tf.io.gfile.glob(tfrecord_path_pattern)
    if not all_files:
        raise RuntimeError(f"No files match {tfrecord_path_pattern}")

    all_files = tf.random.shuffle(all_files)
    num_train = int(len(all_files) * 0.80)
    train_files = all_files[:num_train]
    val_files   = all_files[num_train:]

    def make_ds(files, shuffle):
        ds = (tf.data.Dataset.from_tensor_slices(files)
              .interleave(lambda fn: tf.data.TFRecordDataset(fn),
                          cycle_length=AUTOTUNE,
                          num_parallel_calls=AUTOTUNE))
        if shuffle:
            ds = ds.shuffle(buffer_size=SHUFFLE_BUF,
                            reshuffle_each_iteration=True)
        ds = (ds.map(_parse_function, num_parallel_calls=AUTOTUNE)
                .batch(BATCH_SIZE, drop_remainder=True)
                .prefetch(AUTOTUNE))
        return ds

    train_ds = make_ds(train_files, shuffle=True)
    val_ds   = make_ds(val_files,   shuffle=False)

    # ---------- Multi‑GPU if available ----------------------------------------
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_pointnet2_surface_model()
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=2e-4,
                                                          weight_decay=1e-4),
                      loss=total_loss,
                      metrics=['mse'])

    model.summary()

    ckpt_path = f"models/best_pointnet_surface_{epochs}.weights.h5"
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=True),
        CSVLogger('logs/training_log_pointcloud _simple.csv')
    ]

    steps = NUM_EXAMPLES // BATCH_SIZE

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=epochs,
              steps_per_epoch=steps,
              callbacks=callbacks,
              verbose=2)

    # ---------- Save / export --------------------------------------------------
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_name = f"models/pointnet_surface_{stamp}_{epochs}ep.keras"
    Path('models').mkdir(exist_ok=True)
    model.save(out_name)
    print(f"Training complete — model saved to {out_name}")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1") #dd 0,

    parser = argparse.ArgumentParser(description="Train PointNet++ surface recon")
    parser.add_argument("--tfrecord_glob", type=str, required=True,
                        help="Glob pattern for TFRecord shards, e.g. '/data/points_*.tfrecord'")
    parser.add_argument("--epochs", type=int, default=250,
                        help="Number of training epochs")
    args = parser.parse_args()

    main(tfrecord_path_pattern=args.tfrecord_glob, epochs=args.epochs)
