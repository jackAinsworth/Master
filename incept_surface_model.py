import os
# Set CUDA devices to 0 and 1.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" #"0,1,2,3,4"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np
import json
import argparse
from sklearn.model_selection import train_test_split
import pickle
import datetime

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
def build_incept_surface_model(input_shape=(100, 100, 3), filters=256,
                               output_ctrlpts_u=8, output_ctrlpts_v=8):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu')(inputs)
    x = inception_module_2d(x, 64)

    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), padding='same')(x)

    x = inception_module_2d(x, filters)
    x = layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), padding='same')(x)

    x = inception_module_2d(x, filters)
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), padding='same')(x)

    x = inception_module_2d(x, filters)
    x = layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), padding='same')(x)

    x = layers.Flatten()(x)
    dense_units = output_ctrlpts_u * output_ctrlpts_v * 3  # 3 for (x, y, z)
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
















def load_subdivided_dataset(
    pickle_dir,
    shape_ids=None,
    max_ctrlpts_u=12,
    max_ctrlpts_v=12,
    nu=100,
    nv=100,
    pad_value=-10.0
):
    """
    Load a directory of “dataset_subdivision” .pkl files and return two arrays:

      X: shape (N, nu, nv, 3) — the sampled NURBS patch grids
      Y: shape (N, max_ctrlpts_u, max_ctrlpts_v, 3) — the padded control nets

    The .pkl may use either the old keys (“fitted_surfaces”) or the new
    key (“fitted_grids”), plus “control_nets”. If a `.pkl` nests its data
    under data['data'], those will be picked up too.

    Parameters
    ----------
    pickle_dir : str
        Directory containing `shape_XXXXXX.pkl` files.
    shape_ids : list[int], optional
        Which shape indices to load; by default, scans all files in pickle_dir.
    max_ctrlpts_u : int
        Maximum U dimension of the control‐net (will crop or pad up to this).
    max_ctrlpts_v : int
        Maximum V dimension of the control‐net.
    nu, nv : int
        Resolution of each sampled patch grid (e.g. 100×100).
    pad_value : float
        Value to fill any unused entries in the control‐net.

    Returns
    -------
    X : np.ndarray, shape (N, nu, nv, 3)
        Stacked patch grids.
    Y : np.ndarray, shape (N, max_ctrlpts_u, max_ctrlpts_v, 3)
        Stacked, padded control nets.
    """
    X_list, Y_list = [], []

    # discover all .pkl files if no specific shape_ids given
    if shape_ids is None:
        files = sorted(
            f for f in os.listdir(pickle_dir)
            if f.startswith("shape_") and f.endswith(".pkl")
        )
        shape_ids = [int(f.split("_")[1].split(".")[0]) for f in files]

    for idx in shape_ids:
        path = os.path.join(pickle_dir, f"shape_{idx:06d}.pkl")
        with open(path, "rb") as fp:
            data = pickle.load(fp)

        # try both old/new top-level keys
        fits = data.get("fitted_surfaces") or data.get("fitted_grids")
        nets = data.get("control_nets")

        # fallback to nested under data['data']
        if fits is None or nets is None:
            nested = data.get("data", {})
            fits = fits or nested.get("fitted_surfaces") or nested.get("fitted_grids")
            nets = nets or nested.get("control_nets")

        if fits is None or nets is None:
            raise KeyError(f"{path} missing 'fitted_surfaces'/'fitted_grids' or 'control_nets'")

        # unpack per-patch
        for fit_pts, ctrl_net in zip(fits, nets):
            if fit_pts.shape != (nu, nv, 3):
                raise ValueError(f"{path}: expected fit {(nu, nv, 3)}, got {fit_pts.shape}")

            # crop & pad control net to (max_ctrlpts_u, max_ctrlpts_v, 3)
            net = ctrl_net[:max_ctrlpts_u, :max_ctrlpts_v, :]
            cu_act, cv_act, _ = net.shape

            if cu_act < max_ctrlpts_u:
                pu = np.full((max_ctrlpts_u - cu_act, cv_act, 3), pad_value, dtype=np.float32)
                net = np.concatenate([net, pu], axis=0)
            if cv_act < max_ctrlpts_v:
                pv = np.full((max_ctrlpts_u, max_ctrlpts_v - cv_act, 3), pad_value, dtype=np.float32)
                net = np.concatenate([net, pv], axis=1)

            assert net.shape == (max_ctrlpts_u, max_ctrlpts_v, 3)
            X_list.append(fit_pts.astype(np.float32))
            Y_list.append(net.astype(np.float32))

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y









def main(epochs, data_path, use_subdiv):
    # strategy
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1"])
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # build & compile
    with strategy.scope():
        model = build_incept_surface_model(
            input_shape=(35,35,3),
            filters=256,
            output_ctrlpts_u=10,
            output_ctrlpts_v=10
        )
        model.compile(optimizer=Adam(1e-3), loss="mse")
        model.summary()

    # load dataset
    if use_subdiv:
        # subdivided‐patch dataset uses 100×100 grids and up to 12×12 nets
        X, Y = load_subdivided_dataset(
            pickle_dir=data_path,
            max_ctrlpts_u=12,
            max_ctrlpts_v=12,
            nu=100,
            nv=100,
            pad_value=-10.0
        )
    else:
        # original loader for noisy point‐clouds → control nets
        X, Y = load_and_process_surface_data(
            data_file=data_path,
            max_ctrlpts_u=10,
            max_ctrlpts_v=10,
            num_samples=35
        )

    # train/val/test split
    X_train, X_tmp, Y_train, Y_tmp = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_tmp, Y_tmp, test_size=0.5, random_state=42)

    print("Training set:", X_train.shape, Y_train.shape)
    print("Validation set:", X_val.shape, Y_val.shape)
    print("Test set:", X_test.shape, Y_test.shape)

    # fit
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=512, #156 #365
        verbose=2
    )

    # evaluate
    loss = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Loss (MSE): {loss:.6f}")

    # save
    stamp = datetime.datetime.now().strftime("%d%m_%H%M")
    fname = f"models/incept_surface_{stamp}_{epochs}ep_{loss:.4f}.keras"
    model.save(fname)
    print(f"Model saved to {fname}")
    hist_fname = f"models/history_{stamp}_{epochs}ep_{loss:.4f}.json"
    with open(hist_fname, "w") as f:
        json.dump(history.history, f)
    print(f"History saved to {hist_fname}")


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,    default=250)
    p.add_argument("--data_path", type=str,    required=True,
                   help="Either .npz for old loader or pickles dir for subdiv")
    p.add_argument("--use_subdiv", action="store_true",
                   help="If set, load the subdivided‐patch pickles instead of the original .npz")
    args = p.parse_args()
    main(args.epochs, args.data_path, args.use_subdiv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Inception Surface Model")
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs to train the model")
    parser.add_argument("--data_path", type=str, default="surface_data.npz", help="Path to the dataset npz file")
    args = parser.parse_args()
    main(args.epochs, args.data_path)
