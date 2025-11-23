# tf_ops/ops.py

import os
import threading
import tensorflow as tf

from pointnet2.pnet2_layers.cpp_modules  import farthest_point_sample


# Make up a real batch of 4 point‐clouds, each with 1225 points in 3D:
real_xyz = tf.random.uniform((4, 1225, 3), dtype=tf.float32)

# Sample 512 farthest points:
idx = farthest_point_sample(512, real_xyz)
print(idx.shape)   # → (4, 512)

# Now do a real ball‐query around those:
#nb = query_ball_point(0.1, 32, real_xyz, tf.gather(real_xyz, idx, batch_dims=1))
#print(nb.shape)    # → (4, 512, 32)

