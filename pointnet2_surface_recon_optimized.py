import os
import datetime
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, CSVLogger)


from tensorflow.keras.utils import register_keras_serializable

# ──────────────────────────────────────────────────────────────────────────────
# Constants (adapt to your dataset)
# ──────────────────────────────────────────────────────────────────────────────
NUM_POINTS   = 35*35   # points per cloud
MAX_CTRLS_U  = 10      # B-spline control net resolution (U)
MAX_CTRLS_V  = 10      # B-spline control net resolution (V)
BATCH_SIZE   = 4 #250
SHUFFLE_BUF  = 50_000
AUTOTUNE     = tf.data.AUTOTUNE
NUM_EXAMPLES = 1_660_000

FEATURE_DESC = {
    "xyz_raw":  tf.io.FixedLenFeature([], tf.string),
    "ctrl_raw": tf.io.FixedLenFeature([], tf.string),
}

# ──────────────────────────────────────────────────────────────────────────────
# Utility: Farthest Point Sampling & Ball Query
# ──────────────────────────────────────────────────────────────────────────────

def farthest_point_sample_old(npoint, xyz):
    """
    B: batch, N: points, C: coords
    Return (B, npoint) indices
    """
    B = tf.shape(xyz)[0]
    N = tf.shape(xyz)[1]
    centroids = tf.zeros((B, npoint), dtype=tf.int32)
    distances = tf.ones((B, N), dtype=tf.float32) * 1e10
    farthest = tf.zeros((B,), dtype=tf.int32)

    def body(i, centroids, distances, farthest):
        # Prepare scatter indices: for each batch b, update position i
        batch_idx = tf.range(B, dtype=tf.int32)                    # (B,)
        batch_idx = tf.expand_dims(batch_idx, 1)                   # (B,1)
        idx_i = tf.fill([B, 1], i)                                 # (B,1)
        scatter_idx = tf.concat([batch_idx, idx_i], axis=1)        # (B,2)
        centroids = tf.tensor_scatter_nd_update(
            centroids,
            indices=scatter_idx,
            updates=farthest
        )
        centroid_xyz = tf.gather(xyz, farthest, batch_dims=1)
        diff = xyz - tf.expand_dims(centroid_xyz, 1)
        dist = tf.reduce_sum(diff * diff, axis=-1)
        distances = tf.minimum(distances, dist)
        farthest = tf.argmax(distances, axis=-1, output_type=tf.int32)
        return i+1, centroids, distances, farthest

    # Loop to sample farthest points
    i = tf.constant(0, dtype=tf.int32)
    _, centroids, _, _ = tf.while_loop(
        lambda i, *_: tf.less(i, npoint),
        body,
        loop_vars=(i, centroids, distances, farthest),
        shape_invariants=(i.get_shape(),
                          tf.TensorShape([None, None]),
                          tf.TensorShape([None, None]),
                          tf.TensorShape([None]))
    )
    return centroids



def query_ball_point_old(radius, nsample, xyz, new_xyz):
    """
    Find nsample neighbors in xyz for each point in new_xyz within radius.
    xyz: (B,N,3), new_xyz: (B,npoint,3)
    returns idx: (B,npoint,nsample)
    """
    B = tf.shape(xyz)[0]
    N = tf.shape(xyz)[1]
    npoint = tf.shape(new_xyz)[1]

    # pairwise distance
    sqrdists = tf.reduce_sum((
        tf.expand_dims(new_xyz, 2) - tf.expand_dims(xyz, 1)
    ) ** 2, axis=-1)

    # mask out-of-radius
    mask = sqrdists > (radius ** 2)
    # assign large dist to masked
    inf = tf.constant(1e10, sqrdists.dtype)
    sqrdists = tf.where(mask, inf, sqrdists)

    # find top-nsample smallest
    _, idx = tf.math.top_k(-sqrdists, k=nsample)
    return idx


from pointnet2.pnet2_layers.cpp_modules  import farthest_point_sample, query_ball_point, gather_point, group_point


def sample_and_group(npoint, nsample, xyz, points=None, radius=0.1):
    """
    Implements set abstraction: FPS + ball query + grouping
    xyz: (B,N,3), points: (B,N,C) or None
    returns:
      new_xyz: (B,npoint,3)
      new_points: (B,npoint,nsample,C+3)
    """
    B = tf.shape(xyz)[0]

    # 1) FPS
    print("xyz", xyz.shape, "npoint", npoint)
    idx = farthest_point_sample(npoint, xyz)
    print("idx", idx)
    #assert np.all((idx >= 0) & (idx < N)), "Index out of range!"

    new_xyz = gather_point(xyz, idx)



    # 2) Ball query
    print("ball query")
    #idx_group = query_ball_point(radius, nsample, xyz, new_xyz)  # (B,npoint,nsample)
    idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)

    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization

    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample, channel)
        new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, nample, 3+channel)

        '''        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points'''
    else:
        new_points = grouped_xyz

    return new_xyz, new_points


class PointNetSetAbstraction(layers.Layer):
    def __init__(self, npoint, nsample, mlp_channels, radius=0.1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.mlp_convs = [layers.Conv2D(c, 1, activation='relu') for c in mlp_channels]
        self.bn_layers = [layers.BatchNormalization() for _ in mlp_channels]
        self.mlp_channels = mlp_channels

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, xyz, points):
        print("xyz shape ", xyz.shape)
        new_xyz, new_points = sample_and_group(
            self.npoint, self.nsample, xyz, points, radius=self.radius
        )  # new_points: (B,npoint,nsample,C+3)

        # MLP on grouped points
        x = new_points
        for conv, bn in zip(self.mlp_convs, self.bn_layers):
            x = conv(x)
            x = bn(x)
        # pool
        new_features = tf.reduce_max(x, axis=2)  # (B,npoint,mlp[-1])
        return new_xyz, new_features


def build_pointnet2_surface_model(
    num_points=NUM_POINTS,
    output_ctrlpts_u=MAX_CTRLS_U,
    output_ctrlpts_v=MAX_CTRLS_V
):
    # Input
    inputs = layers.Input(shape=(num_points, 3))
    l0_xyz = inputs
    l0_points = None

    # SA layers
    l1_xyz, l1_points = PointNetSetAbstraction(512, 32, [64,64,128], name='sa1')(l0_xyz, l0_points)
    l2_xyz, l2_points = PointNetSetAbstraction(128, 64, [128,128,256], name='sa2')(l1_xyz, l1_points)

    # Global
    x = layers.GlobalMaxPooling1D()(l2_points)

    # FC head
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    dense_units = output_ctrlpts_u * output_ctrlpts_v * 3
    ctrl_pred = layers.Dense(dense_units, activation='linear')(x)
    ctrl_pred = layers.Reshape((output_ctrlpts_u, output_ctrlpts_v, 3))(ctrl_pred)

    return models.Model(inputs=inputs, outputs=ctrl_pred, name='pointnet2_surface_ctrl')

# (rest of data pipeline, loss, training loop unchanged)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop (mostly unchanged)
# ──────────────────────────────────────────────────────────────────────────────


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


def main(tfrecord_path_pattern: str, epochs: int):
    all_files = tf.io.gfile.glob(tfrecord_path_pattern)
    if not all_files:
        raise RuntimeError(f"No files match {tfrecord_path_pattern}")

    tf.random.shuffle(all_files)
    num_train = int(len(all_files) * 0.80)
    train_files = all_files[:num_train]
    val_files   = all_files[num_train:]

    def make_ds(files, shuffle):
        ds = (tf.data.Dataset.from_tensor_slices(files)
              .interleave(lambda fn: tf.data.TFRecordDataset(fn),
                          cycle_length=AUTOTUNE,
                          num_parallel_calls=AUTOTUNE))
        if shuffle:
            ds = ds.shuffle(buffer_size=SHUFFLE_BUF, reshuffle_each_iteration=True)
        ds = (ds.map(_parse_function, num_parallel_calls=AUTOTUNE)
                .batch(BATCH_SIZE, drop_remainder=True)
                .prefetch(AUTOTUNE))
        return ds

    train_ds = make_ds(train_files, shuffle=True)
    val_ds   = make_ds(val_files,   shuffle=False)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_pointnet2_surface_model()
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=2e-4,
                                                          weight_decay=1e-4),
                      loss=total_loss,
                      metrics=['mse'])

    model.summary()

    ckpt_path = f"models/best_pointnet2_surface_{epochs}.weights.h5"
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=True),
        CSVLogger('logs/training_log_pointcloud.csv')
    ]

    steps = NUM_EXAMPLES // BATCH_SIZE
    steps = 500
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=epochs,
              steps_per_epoch=steps,
              validation_steps=500,
              callbacks=callbacks,
              verbose=2)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_name = f"models/pointnet2_surface_{stamp}_{epochs}ep.keras"
    Path('models').mkdir(exist_ok=True)
    model.save(out_name)
    print(f"Training complete — model saved to {out_name}")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    parser = argparse.ArgumentParser(description="Train PointNet++ Surface Recon (TF‑only)")
    parser.add_argument("--tfrecord_glob", type=str, required=True,
                        help="Glob pattern for TFRecord shards, e.g. '/data/points_*.tfrecord'")
    parser.add_argument("--epochs", type=int, default=250,
                        help="Number of training epochs")
    args = parser.parse_args()

    main(tfrecord_path_pattern=args.tfrecord_glob, epochs=args.epochs)
