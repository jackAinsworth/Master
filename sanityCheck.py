import numpy as np
import tensorflow as tf

feature_description = {
    "noisy_raw": tf.io.FixedLenFeature([], tf.string),
    "ctrl_raw":  tf.io.FixedLenFeature([], tf.string),
}
NUM_SAMPLES    = 35
MAX_CTRLS_U    = 10
MAX_CTRLS_V    = 10
BATCH_SIZE     = 512
SHUFFLE_BUFFER = 50_000
AUTOTUNE       = tf.data.AUTOTUNE


def _parse_function(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    noisy = tf.io.parse_tensor(parsed["noisy_raw"], out_type=tf.float32)
    noisy = tf.reshape(noisy, (NUM_SAMPLES, NUM_SAMPLES, 3))
    ctrl = tf.io.parse_tensor(parsed["ctrl_raw"], out_type=tf.float32)
    ctrl = tf.reshape(ctrl, (MAX_CTRLS_U, MAX_CTRLS_V, 3))
    return noisy, ctrl

def sanity_check_tfrecord(shard_path):
    raw_ds = tf.data.TFRecordDataset(shard_path)
    parsed_ds = raw_ds.map(_parse_function)  # your parse_function
    for i, (noisy, ctrl) in enumerate(parsed_ds.take(5000)):
        # Convert to NumPy to inspect quickly (this pulls a single batch off-device)
        nm = noisy.numpy()
        cm = ctrl.numpy()
        if np.isnan(nm).any() or np.isinf(nm).any():
            print(f"  NaN/Inf found in `noisy` at record {i}")
            break
        if np.isnan(cm).any() or np.isinf(cm).any():
            print(f"  NaN/Inf found in `ctrl` at record {i}")
            break
    else:
        print(f"{shard_path}: First 5000 records looked OK.")

# Run for each shard or a few random shards:
for s in tf.io.gfile.glob("tfrecords/data_*.tfrecord")[:100]:
    sanity_check_tfrecord(s)
