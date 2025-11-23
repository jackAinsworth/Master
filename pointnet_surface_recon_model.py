#!/usr/bin/env python3
"""
train_pointnet_surface.py

All‑in‑one script that:
  • Builds a PointNet++ surface regressor (control‑net output).
  • Consumes TFRecord shards produced from raw point‑cloud data.
  • Trains on 1 or more GPUs with tf.distribute.MirroredStrategy.

Example:
  python pointnet_surface_recon_model.py \
      --tfrecord_glob "./tfrecords_pointcloud/data_*.tfrecord" \
      --epochs 100
"""
import os, argparse, datetime, glob, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import register_keras_serializable

# ──────────────────────────────────────────────────────────────────────────
# Configuration — adapt to your dataset
# ──────────────────────────────────────────────────────────────────────────
NUM_POINTS   = 35*35        # points per cloud
MAX_CTRLS_U  = 10
MAX_CTRLS_V  = 10
BATCH_SIZE   = 4           # tune to your GPU memory
SHUFFLE_BUF  =   1_000 # 50_000
AUTOTUNE     = tf.data.AUTOTUNE
NUM_EXAMPLES = 1_660_000


# ── helper: channel-wise Laplacian ────────────────────────────────────
def laplacian(grid):
    """
    grid : (B, H, W, C) tensor – C = 3 for (x,y,z)
    returns channel-wise 4-neighbour Laplacian with SAME shape.
    """
    lap_k = tf.constant([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]], tf.float32)  # (3,3)
    lap_k = lap_k[..., tf.newaxis, tf.newaxis]  # (3,3,1,1)

    C = tf.shape(grid)[-1]  # channel count at run-time
    lap_k = tf.tile(lap_k, [1, 1, C, 1])  # (3,3,C,1)

    return tf.nn.depthwise_conv2d(grid, lap_k,
                                  strides=[1, 1, 1, 1],
                                  padding='SAME')  # (B,H,W,C)


# ─────────────────────────────────────────────────────────────────────

@register_keras_serializable(package='CustomLosses', name='total_loss')
def total_loss(y_true, y_pred, w_lap=0.10):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    lap = tf.reduce_mean(tf.square(laplacian(y_true) - laplacian(y_pred)))
    return mse + w_lap * lap


# ── metrics that Keras will compute on every batch ────────────
@register_keras_serializable(package='CustomMetrics', name='pred_mean')
def pred_mean(y_true, y_pred):
    return tf.reduce_mean(y_pred)  # ⟵ prediction mean


pred_mean.__name__ = "pred_mean"  # give it a stable name


@register_keras_serializable(package='CustomMetrics', name='target_mean')
def target_mean(y_true, y_pred):
    return tf.reduce_mean(y_true)  # ⟵ target mean


target_mean.__name__ = "target_mean"






# ──────────────────────────────────────────────────────────────────────────
# Import Layer-based PointNet++ modules
# ──────────────────────────────────────────────────────────────────────────
from pointnet2.pnet2_layers.layers import Pointnet_SA, Pointnet_SA_MSG, Pointnet_FP
# ──────────────────────────────────────────────────────────────────────────
# Build PointNet++ encoder + dense head → control net regressor
# ──────────────────────────────────────────────────────────────────────────






def build_pointnet2_surface_regressor(
    num_points=NUM_POINTS,
    output_ctrlpts_u=MAX_CTRLS_U,
    output_ctrlpts_v=MAX_CTRLS_V
):
    xyz_in = layers.Input(shape=(num_points, 3), name="xyz_input")

    # Multi-scale Set Abstraction layers
    l1_xyz, l1_feats = Pointnet_SA_MSG(
        npoint= num_points//2,
        radius_list=[0.05, 0.1, 0.2],
        nsample_list=[34, 34, 34],
        mlp=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
        use_xyz=False, activation=tf.nn.relu, bn=False
    )(xyz_in, None)


    l2_xyz, l2_feats = Pointnet_SA_MSG(
        npoint=num_points//4,
        radius_list=[0.1, 0.2, 0.4],
        nsample_list=[32, 32, 32],
        mlp=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
        use_xyz=False, activation=tf.nn.relu, bn=False
    )(l1_xyz, l1_feats)

    #l2_feats = pad_to_multiple(l2_feats, 8, "l1_align")

    l3_xyz, l3_feats = Pointnet_SA_MSG(
        npoint=num_points//16,
        radius_list=[0.2, 0.4, 0.8],
        nsample_list=[32, 32, 32],
        mlp=[[128, 128, 256], [256, 256, 512], [256, 256, 512]],
        use_xyz=False, activation=tf.nn.relu, bn=False
    )(l2_xyz, l2_feats)


    # Global feature aggregation
    global_feat = layers.GlobalMaxPooling1D()(l3_feats)
    x = layers.Dense(512, activation="relu")(global_feat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)


    # Output control points grid
    total_ctrl = output_ctrlpts_u * output_ctrlpts_v * 3
    x = layers.Dense(total_ctrl, activation="linear")(x)
    ctrl_out = layers.Reshape((output_ctrlpts_u, output_ctrlpts_v, 3),
                              name="ctrlpts_output")(x)

    return models.Model(inputs=xyz_in, outputs=ctrl_out,
                        name="PointNet2_SurfaceRegressor")












# ──────────────────────────────────────────────────────────────────────────
# Dataset – TFRecord → (xyz, ctrl) pairs
# ──────────────────────────────────────────────────────────────────────────
# ─── TFRecord feature spec ────────────────────────────────────────────────
# ─── TFRecord feature spec ─────────────────────────────────────────────────
FEATURE_DESC = {
    "xyz_raw" : tf.io.FixedLenFeature([], tf.string),
    "ctrl_raw": tf.io.FixedLenFeature([], tf.string),
}

# ─── Parse one Example into (xyz, ctrl) ─────────────────────────────────────
def _parse_function(example_proto):
    """
    Parses a single tf.train.Example into:
      • xyz:  (NUM_POINTS, 3) float32 tensor
      • ctrl: (MAX_CTRLS_U, MAX_CTRLS_V, 3) float32 tensor
    """
    # 1) Parse the raw proto
    parsed = tf.io.parse_single_example(example_proto, FEATURE_DESC)

    # 2) Decode and reshape the pointcloud
    xyz = tf.io.parse_tensor(parsed["xyz_raw"], out_type=tf.float32)
    xyz = tf.reshape(xyz, (NUM_POINTS, 3))

    # 3) Decode and reshape the control net
    ctrl = tf.io.parse_tensor(parsed["ctrl_raw"], out_type=tf.float32)
    ctrl = tf.reshape(ctrl, (MAX_CTRLS_U, MAX_CTRLS_V, 3))


    return xyz, ctrl

# ─── Build a tf.data.Dataset from TFRecord shards ──────────────────────────
def make_dataset_from_tfrecords(tfrecord_glob_pattern, shuffle: bool = True):
    """
    1) Glob the pattern into a Python list of filenames.
    2) (Optionally) shuffle the filename list.
    3) Interleave all shards, parse with _parse_function.
    4) (Optionally) shuffle at example level, then batch & prefetch.
    """
    # 1) find all shard files
    files = tf.io.gfile.glob(tfrecord_glob_pattern)
    if not files:
        raise ValueError(f"No TFRecord files match pattern {tfrecord_glob_pattern}")

    # 2) shuffle the Python list once
    if shuffle:
        files = list(files)
        np.random.shuffle(files)

    # 3) interleave & parse
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.interleave(
        lambda fn: tf.data.TFRecordDataset(fn),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.map(_parse_function, num_parallel_calls=AUTOTUNE)

    # 4) example‐level shuffle, batch, prefetch
    if shuffle:
        ds = ds.shuffle(buffer_size=SHUFFLE_BUF, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(AUTOTUNE)




# ──────────────────────────────────────────────────────────────────────────
# Training entry‑point
# ──────────────────────────────────────────────────────────────────────────

def main(tfrecord_glob, epochs):
    # 1) Discover & split
    all_files = tf.io.gfile.glob(tfrecord_glob)
    if not all_files:
        raise RuntimeError(f"No TFRecord files match {tfrecord_glob}")
    tf.random.set_seed(42)
    all_files = tf.random.shuffle(all_files)
    num_train = int(0.8 * len(all_files))
    print("Train shards:", num_train, "Val shards:", len(all_files) - num_train)

    train_files = all_files[:num_train].numpy().tolist()
    val_files = all_files[num_train:].numpy().tolist()

    # 2) Build tf.data pipelines exactly as you had them
    train_ds = (tf.data.Dataset.from_tensor_slices(train_files).repeat()
                .interleave(lambda fn: tf.data.TFRecordDataset(fn),
                            cycle_length=AUTOTUNE,
                            num_parallel_calls=AUTOTUNE)
                .shuffle(buffer_size=SHUFFLE_BUF, reshuffle_each_iteration=True)
                .map(_parse_function, num_parallel_calls=AUTOTUNE)
                .batch(BATCH_SIZE, drop_remainder=True)
                .prefetch(AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices(val_files)
              .interleave(lambda fn: tf.data.TFRecordDataset(fn),
                          cycle_length=AUTOTUNE,
                          num_parallel_calls=AUTOTUNE)
              .map(_parse_function, num_parallel_calls=AUTOTUNE)
              .batch(BATCH_SIZE, drop_remainder=True)
              .prefetch(AUTOTUNE))

    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:1"]) #dd "/gpu:0",
    with strategy.scope():
        model = build_pointnet2_surface_regressor()



        model.compile(optimizer=AdamW(2e-4, weight_decay=1e-4),
                      loss=total_loss,
                      metrics=["mse"])

    model.summary()

    ckpt_path = f"models/best_pointnet_surface_{epochs}.weights.h5"
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=True),
        CSVLogger('logs/training_log_pointcloud.csv')
    ]

    num_train = NUM_EXAMPLES * 0.8
    num_val = NUM_EXAMPLES * 0.2
    train_steps = num_train // BATCH_SIZE
    val_steps = num_val // BATCH_SIZE


    model.fit(train_ds,
              validation_data=val_ds,
              #steps_per_epoch=train_steps,
              #validation_steps=val_steps,
              epochs=epochs,
              callbacks=callbacks,
              verbose=2)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    final_name = f"models/pointnet_surface_{stamp}_{epochs}ep.keras"
    model.save(final_name)
    print(f"Finished training → saved {final_name}")

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1") #dd 0,
    parser = argparse.ArgumentParser(description="Train PointNet++ surface regressor")
    parser.add_argument("--tfrecord_glob", required=True,
                        help="Glob pattern for TFRecord shards, e.g. '/data/pc_*.tfrecord'")
    parser.add_argument("--epochs", type=int, default=250)
    args = parser.parse_args()

    info = tf.sysconfig.get_build_info()

    print(tf.__version__)
    # CPU / GPU build?
    print(tf.config.list_physical_devices('GPU'))

    # Full build configuration (CUDA, cuDNN versions, compile flags)
    print(info )
    #tf.config.experimental.enable_op_determinism()  # TF ≥2.9
    tf.config.experimental.enable_op_determinism()

    main(args.tfrecord_glob, args.epochs)
