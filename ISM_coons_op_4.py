"""
Example:
  python surface_model_train_coons.py \
      --tfrecord_glob "./tfrecords_coons/data_*.tfrecord" \
      --epochs 300


      LOG=logs/coons_surface_$(date +%Y%m%d_%H%M).log   # e.g. pointnet_surface_20250618_1423.log
CUDA_VISIBLE_DEVICES=1 \

nohup python -u ISM_coons_op_3.py \
      --tfrecord_glob "./tfrecords_coons/data_*.tfrecord" \
      --epochs 150 \
      > "$LOG" 2>&1 &


LOG=logs/ruled_surface_$(date +%Y%m%d_%H%M).log   # e.g. pointnet_surface_20250618_1423.log

nohup python -u ISM_ruled_op_4.py   --tfrecord_glob "/home/ainsworth/master/dataset/tfrecords_ruled/*.tfrecord"   --epochs 150   > "$LOG" 2>&1 &

echo "Started PID $! – tail -f $LOG to follow progress"
"""

'''
changes
1 back to a more stable pattern in op 1 
2 changes to the reading of the data with a hash map or something needs to be double checked
'''


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"  # bb

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import register_keras_serializable

import numpy as np
import pandas as pd
import pickle
import datetime
import argparse
from geomdl import utilities, helpers, BSpline

try:
    from tensorflow.keras.layers import StochasticDepth  # TF ≥ 2.12

    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False  # fallback to plain Dropout
# Define a 2D inception module for surfaces.

NUM_SAMPLES = 50
MAX_CTRLS_U = 12
MAX_CTRLS_V = 12
BATCH_SIZE = 512 #512 #64
SHUFFLE_BUFFER = 50_000
AUTOTUNE = tf.data.AUTOTUNE

feature_description = {
    "noisy_raw": tf.io.FixedLenFeature([], tf.string),
    "ctrl_raw": tf.io.FixedLenFeature([], tf.string),
}



def inception_module_2d(x, filters):
    path1 = layers.Conv2D(filters, kernel_size=(1, 1), padding='same', activation='relu')(x)

    path2 = layers.Conv2D(filters, kernel_size=(1, 1), padding='same', activation='relu')(x)
    path2 = layers.Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(path2)

    path3 = layers.Conv2D(filters, kernel_size=(1, 1), padding='same', activation='relu')(x)
    path3 = layers.Conv2D(filters, kernel_size=(5, 5), padding='same', activation='relu')(path3)

    path4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    path4 = layers.Conv2D(filters, kernel_size=(1, 1), padding='same', activation='relu')(path4)


    return layers.Concatenate(axis=-1)([path1, path2, path3, path4])


# Build the inception-based surface model.
def build_incept_surface_model(input_shape=(100, 100, 3),
                               filters=256,
                               output_ctrlpts_u=8,
                               output_ctrlpts_v=8):
    inputs = layers.Input(shape=input_shape)

    # ─── First Conv + Inception ──────────────────────────────────────────────
    x = layers.Conv2D(64, (5, 5), padding='same', activation=None)(inputs)
    x = layers.ReLU()(x)
    # If you want, you could insert dropout here (rate=0.1–0.2), but it's optional:
    #x = layers.Dropout(0.5)(x)

    x = inception_module_2d(x, 64)
    # Encouraging diverse filters in Inception: small dropout (e.g. 0.1)
    # x = layers.Dropout(0.1)(x)

    # ─── Second Conv Block ───────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # After BatchNorm+ReLU, drop 10–20% of activations
    # x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling2D((3, 3), padding='same')(x)

    # ─── Third Inception + Conv Block ────────────────────────────────────────
    x = inception_module_2d(x, filters)
    # x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(512, (3, 3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling2D((3, 3), padding='same')(x)

    # ─── Fourth Inception + Conv Block ───────────────────────────────────────
    x = inception_module_2d(x, filters)
    # x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(256, (3, 3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling2D((3, 3), padding='same')(x)

    # ─── Fifth Inception + Conv Block ────────────────────────────────────────
    x = inception_module_2d(x, filters)
    # x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(512, (3, 3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling2D((3, 3), padding='same')(x)

    # ─── Flatten → Dense Head with Strong Dropout ────────────────────────────
    x = layers.Flatten()(x)
    # A heavier dropout (e.g. 0.5) right before the dense regression output:
    #x = layers.Dropout(0.1)(x)

    # 1) control-net prediction
    dense_units = output_ctrlpts_u * output_ctrlpts_v * 3
    ctrl_pred = layers.Dense(dense_units, activation='linear', name='ctrl')(x)
    ctrl_pred = layers.Reshape((output_ctrlpts_u, output_ctrlpts_v, 3),
                               name='ctrl_net')(ctrl_pred)

    # 2) reconstructed surface


    '''    surf_pred = layers.Lambda(reconstruct_surface, name='recon_surf')(ctrl_pred)
    model = models.Model(inputs=inputs,
                 outputs=[ctrl_pred, surf_pred],
                 name="surface_ctrl_and_recon")'''
    model = models.Model(inputs=inputs, outputs=ctrl_pred)

    return model


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
def total_loss(y_true, y_pred, w_lap=0.05):
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


@register_keras_serializable()
def surface_loss(y_true, y_pred):
    # average squared error over BATCH × H × W × C → a single scalar
    return tf.reduce_mean(tf.square(y_true - y_pred))




SPLIT_SEED = tf.constant(1337, tf.int64)
VAL_BUCKETS = 2          # 20% val if modulo=10 and <2
MODULO = 10

def in_val(serialized):
    # stable 64-bit-ish hash, cheap
    h = tf.strings.to_hash_bucket_fast(serialized, MODULO * 1000)
    # fold in a seed for reproducibility
    h = (h + tf.cast(SPLIT_SEED, tf.int64)) % MODULO
    return h < VAL_BUCKETS

def make_ds(pattern, batch_size=BATCH_SIZE, shuffle=True):
    files = tf.data.Dataset.list_files(pattern, shuffle=True, seed=123)
    ds = files.interleave(tf.data.TFRecordDataset,
                          cycle_length=AUTOTUNE,
                          num_parallel_calls=AUTOTUNE)

    # → split BEFORE parsing to keep it cheap
    train_raw = ds.filter(lambda s: tf.logical_not(in_val(s)))
    val_raw   = ds.filter(in_val)

    def parse_map(s):
        parsed = tf.io.parse_single_example(s, feature_description)
        noisy = tf.io.parse_tensor(parsed["noisy_raw"], out_type=tf.float32)
        noisy = tf.reshape(noisy, (NUM_SAMPLES, NUM_SAMPLES, 3))
        ctrl  = tf.io.parse_tensor(parsed["ctrl_raw"], out_type=tf.float32)
        ctrl  = tf.reshape(ctrl, (MAX_CTRLS_U, MAX_CTRLS_V, 3))
        return noisy, ctrl

    train = train_raw
    if shuffle:
        train = train.shuffle(SHUFFLE_BUFFER, reshuffle_each_iteration=True)
    train = (train.map(parse_map, num_parallel_calls=AUTOTUNE)
                  .batch(batch_size)
                  .prefetch(AUTOTUNE))

    val = (val_raw
           .map(parse_map, num_parallel_calls=AUTOTUNE)
           .batch(batch_size, drop_remainder=True)
           .prefetch(AUTOTUNE))
    return train, val










def _parse_function(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    noisy = tf.io.parse_tensor(parsed["noisy_raw"], out_type=tf.float32)
    noisy = tf.reshape(noisy, (NUM_SAMPLES, NUM_SAMPLES, 3))
    ctrl = tf.io.parse_tensor(parsed["ctrl_raw"], out_type=tf.float32)
    ctrl = tf.reshape(ctrl, (MAX_CTRLS_U, MAX_CTRLS_V, 3))
    return noisy, ctrl


def make_dataset_from_tfrecords(tfrecord_glob_pattern, shuffle: bool = True):
    # 1) List all matching files
    files = tf.io.gfile.glob(tfrecord_glob_pattern)
    if not files:
        raise ValueError(f"No TFRecord files match pattern {tfrecord_glob_pattern}")
    files = tf.random.shuffle(files)  # shuffle file order once

    # 2) Create one interleaved Dataset across all shards
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.interleave(
        lambda fn: tf.data.TFRecordDataset(fn),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE
    )

    # 3) Optionally shuffle at example level
    if shuffle:
        ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER, reshuffle_each_iteration=True)

    # 4) Parse, batch, prefetch
    ds = ds.map(_parse_function, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(AUTOTUNE)


def main(tfrecord_path_pattern: str, epochs: int):
    tf.keras.mixed_precision.set_global_policy('float32')

    # 2) Build tf.data pipelines
    train_ds, val_ds = make_ds(tfrecord_path_pattern)

    # 3) Build model under MirroredStrategy (if you have >1 GPU; otherwise omit):
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"]) #,"/gpu:2", "/gpu:3","/gpu:4", "/gpu:5","/gpu:6"
    with strategy.scope():


        model = build_incept_surface_model(
            input_shape=(NUM_SAMPLES, NUM_SAMPLES, 3),
            filters=256,
            output_ctrlpts_u=MAX_CTRLS_U,
            output_ctrlpts_v=MAX_CTRLS_V
        )

        model.compile(
           optimizer=tf.keras.optimizers.AdamW(learning_rate=2e-4, weight_decay=1e-4, clipnorm=1.0 ), #2e-4
            loss=total_loss,
            #loss="mse",
            metrics=['mse']
        )



    model.summary()






    # 4) Callbacks
    ckpt_path = f"models/best_coons_surface_{epochs}.weights.h5"
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=True),
        CSVLogger('logs/training_log_coons.csv')
    ]


    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )


    # 6) Save final model
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_name = f"models/coons_surface_incept_{stamp}_{epochs}ep.keras"
    model.save(out_name)
    print(f"Finished training → final model saved to {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using TFRecords + tf.data")
    parser.add_argument("--tfrecord_glob", type=str, required=True,
                        help="Glob pattern matching your TFRecord shards, e.g. '/path/to/data_*.tfrecord'")
    parser.add_argument("--epochs", type=int, default=250,
                        help="Number of epochs to train")
    args = parser.parse_args()

    main(tfrecord_path_pattern=args.tfrecord_glob, epochs=args.epochs)
