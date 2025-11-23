"""
Example:
  python surface_model_train_coons.py \
      --tfrecord_glob "./tfrecords_coons/data_*.tfrecord" \
      --epochs 300


      LOG=logs/coons_surface_$(date +%Y%m%d_%H%M).log   # e.g. pointnet_surface_20250618_1423.log
CUDA_VISIBLE_DEVICES=1 \

nohup python -u ISM_coons_op_2.py \
      --tfrecord_glob "./tfrecords_coons/data_*.tfrecord" \
      --epochs 30 \
      > "$LOG" 2>&1 &
echo "Started PID $! – tail -f $LOG to follow progress"
"""

'''
CHANGES 

 Conv layers follow the Conv → BatchNorm → ReLU pattern
 epsilon bn norm padding 
 

'''




import os

#os.environ["CUDA_VISIBLE_DEVICES"] = ""  # bb 1, 0
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



def conv_bn_relu(x, filters, kernel_size, strides=1, dilation_rate=1, eps=1e-3):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        dilation_rate=dilation_rate,
        use_bias=False,  # BN handles bias
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization(epsilon=eps)(x)
    x = layers.PReLU()(x)
    return x

def inception_module_2d(x, filters, eps=1e-3):
    path1 = conv_bn_relu(x, filters, (1, 1), eps=eps)

    path2 = conv_bn_relu(x, filters, (1, 1), eps=eps)
    path2 = conv_bn_relu(path2, filters, (3, 3), eps=eps)

    path3 = conv_bn_relu(x, filters, (1, 1), eps=eps)
    path3 = conv_bn_relu(path3, filters, (5, 5), eps=eps)

    path4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    path4 = conv_bn_relu(path4, filters, (1, 1), eps=eps)

    return layers.Concatenate(axis=-1)([path1, path2, path3, path4,])

def build_incept_surface_model(input_shape=(100, 100, 3),
                               filters=256,
                               output_ctrlpts_u=8,
                               output_ctrlpts_v=8):
    inputs = layers.Input(shape=input_shape)

    # First Conv + Inception
    x = conv_bn_relu(inputs, 64, (5, 5))
    x = inception_module_2d(x, 64)

    # Second Conv Block
    x = conv_bn_relu(x, 128, (3, 3))
    x = layers.MaxPooling2D((3, 3), padding='same')(x)

    # Third Inception + Conv Block
    x = inception_module_2d(x, filters)
    x = conv_bn_relu(x, 512, (3, 3))
    x = layers.MaxPooling2D((3, 3), padding='same')(x)

    # Fourth Inception + Conv Block
    x = inception_module_2d(x, filters)
    x = conv_bn_relu(x, 256, (3, 3))
    x = layers.MaxPooling2D((3, 3), padding='same')(x)

    # Fifth Inception + Conv Block
    x = inception_module_2d(x, filters)
    x = conv_bn_relu(x, 512, (3, 3))
    x = layers.MaxPooling2D((3, 3), padding='same')(x)

    # Flatten → Dense Head
    x = layers.Flatten()(x)
    # Optional: Dropout here for regularization
    # x = layers.Dropout(0.5)(x)

    # Control net output
    dense_units = output_ctrlpts_u * output_ctrlpts_v * 3
    ctrl_pred = layers.Dense(dense_units, activation='linear', name='ctrl')(x)
    ctrl_pred = layers.Reshape((output_ctrlpts_u, output_ctrlpts_v, 3), name='ctrl_net')(ctrl_pred)

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


@register_keras_serializable()
def surface_loss(y_true, y_pred):
    # average squared error over BATCH × H × W × C → a single scalar
    return tf.reduce_mean(tf.square(y_true - y_pred))


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
    # 1) Create train/val splits at file‐level (80% train, 20% val):
    all_files = tf.io.gfile.glob(tfrecord_path_pattern)
    print("All shards found:", all_files)

    if not all_files:
        raise RuntimeError(f"No files match {tfrecord_path_pattern}")
    all_files = tf.random.shuffle(all_files)
    num_train = int(len(all_files) * 0.80)
    print("Train shards:", num_train, "Val shards:", len(all_files) - num_train)

    train_files = all_files[:num_train]
    val_files = all_files[num_train:]

    train_pattern = None  # we’ll pass a list directly
    val_pattern = None

    # 2) Build tf.data pipelines
    train_ds = (tf.data.Dataset.from_tensor_slices(train_files)
                .interleave(lambda fn: tf.data.TFRecordDataset(fn),
                            cycle_length=AUTOTUNE,
                            num_parallel_calls=AUTOTUNE)
                .shuffle(buffer_size=SHUFFLE_BUFFER, reshuffle_each_iteration=True)
                .map(_parse_function, num_parallel_calls=AUTOTUNE)
                .batch(BATCH_SIZE)
                .prefetch(AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices(val_files)
              .interleave(lambda fn: tf.data.TFRecordDataset(fn),
                          cycle_length=AUTOTUNE,
                          num_parallel_calls=AUTOTUNE)
              .map(_parse_function, num_parallel_calls=AUTOTUNE)
              .batch(BATCH_SIZE, drop_remainder=True)
              .prefetch(AUTOTUNE))

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
           optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4), # learning_rate=2e-4
            #loss=total_loss,
            loss=total_loss,
            metrics=['mse']
        )



    model.summary()






    # 4) Callbacks
    ckpt_path = f"models/op_1_best_coons_surface_{epochs}.weights.h5"
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
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
    out_name = f"models/op_1_coons_surface_incept_{stamp}_{epochs}ep.keras"
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
