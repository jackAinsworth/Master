import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0" #"0,1,2,3,4"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np
import json
import argparse
from sklearn.model_selection import train_test_split
import pickle
import datetime
import pandas as pd
from tensorflow.keras.utils import register_keras_serializable

import DataLoader


from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    ModelCheckpoint, CSVLogger
)

from tensorflow.keras.optimizers import AdamW

from tensorflow.keras import layers
try:
    from tensorflow.keras.layers import StochasticDepth   # TF ≥ 2.12
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False      # fallback to plain Dropout
# Define a 2D inception module for surfaces.




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
    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(inputs)
    # If you want, you could insert dropout here (rate=0.1–0.2), but it's optional:
    # x = layers.Dropout(0.1)(x)

    x = inception_module_2d(x, 64)
    # Encouraging diverse filters in Inception: small dropout (e.g. 0.1)
    #x = layers.Dropout(0.1)(x)

    # ─── Second Conv Block ───────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # After BatchNorm+ReLU, drop 10–20% of activations
    #x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling2D((3, 3), padding='same')(x)

    # ─── Third Inception + Conv Block ────────────────────────────────────────
    x = inception_module_2d(x, filters)
    #x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling2D((3, 3), padding='same')(x)

    # ─── Fourth Inception + Conv Block ───────────────────────────────────────
    x = inception_module_2d(x, filters)
    #x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling2D((3, 3), padding='same')(x)

    # ─── Fifth Inception + Conv Block ────────────────────────────────────────
    x = inception_module_2d(x, filters)
    #x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling2D((3, 3), padding='same')(x)

    # ─── Flatten → Dense Head with Strong Dropout ────────────────────────────
    x = layers.Flatten()(x)
    # A heavier dropout (e.g. 0.5) right before the dense regression output:
    #x = layers.Dropout(0.5)(x)

    dense_units = output_ctrlpts_u * output_ctrlpts_v * 3  # 3 coords (x, y, z)
    x = layers.Dense(dense_units, activation='linear')(x)
    outputs = layers.Reshape((output_ctrlpts_u, output_ctrlpts_v, 3))(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def load_and_process_surface_data(data_file, max_ctrlpts_u=8, max_ctrlpts_v=8, num_samples=100, pad_value=-10.0):
    """
    Loads the dataset from a pickle (.pkl) or npz file.
    For a pickle file, the data is assumed to be a list of entries.
    Each entry is expected to have:
       - 'rotated_noisy_points': shape [num_samples, num_samples, 3]
       - 'rotated_control_net': shape [u, v, 3]
    Pads or truncates the control net to fixed shape (max_ctrlpts_u, max_ctrlpts_v, 3) using the specified pad_value.
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
        noisy = np.array(item['rotated_noisy_points'], dtype=np.float32)
        ctrl_net = np.array(item['rotated_control_net'], dtype=np.float32)
        current_u, current_v, _ = ctrl_net.shape
        # Adjust the u dimension.
        if current_u < max_ctrlpts_u:
            pad_u = np.full((max_ctrlpts_u - current_u, current_v, 3), pad_value, dtype=np.float32)
            ctrl_net = np.concatenate([ctrl_net, pad_u], axis=0)
        elif current_u > max_ctrlpts_u:
            ctrl_net = ctrl_net[:max_ctrlpts_u, :, :]
        # Adjust the v dimension.
        if current_v < max_ctrlpts_v:
            pad_v = np.full((max_ctrlpts_u, max_ctrlpts_v - current_v, 3), pad_value, dtype=np.float32)
            ctrl_net = np.concatenate([ctrl_net, pad_v], axis=1)
        elif current_v > max_ctrlpts_v:
            ctrl_net = ctrl_net[:, :max_ctrlpts_v, :]
        X_list.append(noisy)
        Y_list.append(ctrl_net)
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y



def load_multiple_surface_data_from_csv(
    csv_file,
    max_ctrlpts_u=8,
    max_ctrlpts_v=8,
    num_samples=100,
    pad_value=-10.0
):
    """
    Given a CSV whose rows list one `filepath` per row, load each .pkl/.npz in turn
    and concatenate all the (X, Y) pairs together.

    Expects the CSV to have a column named 'filepath' (adjust if yours is different).
    Returns:
        X_all: np.ndarray of shape (total_examples, H, W, 3)
        Y_all: np.ndarray of shape (total_examples, max_ctrlpts_u, max_ctrlpts_v, 3)
    """
    df = pd.read_csv(csv_file)
    if 'filepath' not in df.columns:
        raise ValueError(f"CSV must have a column named 'filepath' (found: {df.columns.tolist()})")

    X_list = []
    Y_list = []
    for fp in df['filepath'].tolist():
        # fp might be a .pkl or .npz (anything your original loader supports)
        X_i, Y_i = load_and_process_surface_data(
            data_file=fp,
            max_ctrlpts_u=max_ctrlpts_u,
            max_ctrlpts_v=max_ctrlpts_v,
            num_samples=num_samples,
            pad_value=pad_value
        )
        X_list.append(X_i)
        Y_list.append(Y_i)

    # vert‐stack along the first dimension
    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)
    return X_all, Y_all




# ── helper: channel-wise Laplacian ────────────────────────────────────
def laplacian(grid):
    """
    grid : (B, H, W, C) tensor – C = 3 for (x,y,z)
    returns channel-wise 4-neighbour Laplacian with SAME shape.
    """
    lap_k = tf.constant([[0, 1, 0],
                         [1,-4, 1],
                         [0, 1, 0]], tf.float32)          # (3,3)
    lap_k = lap_k[..., tf.newaxis, tf.newaxis]            # (3,3,1,1)

    C = tf.shape(grid)[-1]                                # channel count at run-time
    lap_k = tf.tile(lap_k, [1, 1, C, 1])                  # (3,3,C,1)

    return tf.nn.depthwise_conv2d(grid, lap_k,
                                  strides=[1,1,1,1],
                                  padding='SAME')         # (B,H,W,C)
# ─────────────────────────────────────────────────────────────────────

@register_keras_serializable(package='CustomLosses', name='total_loss')
def total_loss(y_true, y_pred, w_lap=0.10):
    mse  = tf.reduce_mean(tf.square(y_true - y_pred))
    lap  = tf.reduce_mean(tf.square(laplacian(y_true) - laplacian(y_pred)))
    return mse + w_lap * lap





# ── metrics that Keras will compute on every batch ────────────
@register_keras_serializable(package='CustomMetrics', name='pred_mean')
def pred_mean(y_true, y_pred):
    return tf.reduce_mean(y_pred)          # ⟵ prediction mean
pred_mean.__name__ = "pred_mean"           # give it a stable name

@register_keras_serializable(package='CustomMetrics', name='target_mean')
def target_mean(y_true, y_pred):
    return tf.reduce_mean(y_true)          # ⟵ target mean
target_mean.__name__ = "target_mean"














def main(epochs, data_path):
    # strategy
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1" ]) #,"/gpu:1"
    print(f"Number of devices: {strategy.num_replicas_in_sync}")



    # build & compile
    with strategy.scope():
        batch_size = 512 #512 64 128 256


        # original loader for noisy point‐clouds → control nets
        if data_path.lower().endswith('.csv'):
            # If it's a CSV, load each .pkl/.npz path from the 'filepath' column:
            X, Y = load_multiple_surface_data_from_csv(
                csv_file=data_path,
                max_ctrlpts_u=10,
                max_ctrlpts_v=10,
                num_samples=35,
                pad_value=-10.0
            )
        else:
            # Single‐file mode (exactly as before):
            X, Y = load_and_process_surface_data(
                data_file=data_path,
                max_ctrlpts_u=10,
                max_ctrlpts_v=10,
                num_samples=35,
                pad_value=-10.0
            )
        
        # train/val/test split
        X_train, X_tmp, Y_train, Y_tmp = train_test_split(X, Y, test_size=0.2, random_state=42) #0.3
        X_val, X_test, Y_val, Y_test = train_test_split(X_tmp, Y_tmp, test_size=0.5, random_state=42)

        print("Training set:", X_train.shape, Y_train.shape)
        print("Validation set:", X_val.shape, Y_val.shape)
        print("Test set:", X_test.shape, Y_test.shape)


        early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ckpt_best = ModelCheckpoint('best.h5', monitor='val_loss',
                                    save_best_only=True, save_weights_only=True)
        csv_logger = CSVLogger('log.csv')

        # ── work out how many steps in one “cycle” ─────────────────────────
        num_train_samples = X_train.shape[0]  # rows in your array

        steps_per_epoch = (num_train_samples + batch_size - 1) // batch_size
        epochs_per_cycle = 20  # 20-epoch cosine wave
        first_decay_steps = steps_per_epoch * epochs_per_cycle

        from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
        lr_schedule = CosineDecayRestarts(
            initial_learning_rate=2e-4,
            first_decay_steps=first_decay_steps,  # step count, *not* epochs
            t_mul=2.0,  # cycle length doubles each restart  (optional)
            m_mul=1.0,  # peak LR stays the same            (optional)
            alpha=1e-5  # final LR floor inside each cycle
        )

        model = build_incept_surface_model(
            input_shape=(35,35,3),
            filters=256,
            output_ctrlpts_u=10,
            output_ctrlpts_v=10
        )

        opt = tf.keras.optimizers.AdamW(learning_rate = lr_schedule, weight_decay=1e-4, clipnorm=1.0)
        model.compile(opt, loss=total_loss,  metrics   = ['mse'] )

        model.summary()





    print("fit")
    # fit
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size, #156 #365
        verbose=2
    )


    # evaluate
    loss = model.evaluate(X_test, Y_test, verbose=0)




    # save
    stamp = datetime.datetime.now().strftime("%d%m_%H%M")
    fname = f"models/incept_surface_{stamp}_{epochs}ep.keras"
    model.save(fname)
    print(f"Model saved to {fname}")

    print(f"Test Loss (MSE): {loss:.6f}")


    #hist_fname = f"models/history_{stamp}_{epochs}ep_{loss:.4f}.json"
    #with open(hist_fname, "w") as f:
        #json.dump(history.history, f)
    #print(f"History saved to {hist_fname}")


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,    default=250)
    p.add_argument("--data_path", type=str,    required=True,
                   help="Either .npz for old loader or pickles dir for subdiv")
    args = p.parse_args()
    main(args.epochs, args.data_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Inception Surface Model")
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs to train the model")
    parser.add_argument("--data_path", type=str, default="surface_data.npz", help="Path to the dataset npz file")
    args = parser.parse_args()
    main(args.epochs, args.data_path)
