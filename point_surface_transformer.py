#!/usr/bin/env python3
"""
train_pointnet_surface_transformer.py

PyTorch re‑implementation of train_pointnet_surface using
Transformer‐Style Self‐Attention (Point Transformer / PCT) instead of
PointNet++ SA layers.

• Builds a PointTransformer surface regressor (control‑net output).
• Consumes TFRecord shards produced from raw point‑cloud data (unchanged pipeline).
• Trains on 1ormore GPUs via torch.nn.DataParallel.

Example:
  python point_surface_transformer.py \
      --tfrecord_glob "./tfrecords_pointcloud/data_*.tfrecord" \
      --epochs 100

LOG=logs/point_surface_transformer_$(date +%Y%m%d_%H%M).log   # e.g. pointnet_surface_20250618_1423.log
CUDA_VISIBLE_DEVICES=1 \

nohup python -u point_surface_transformer.py \
    --tfrecord_glob "./tfrecords_pointcloud/data_*.tfrecord" \
    --epochs 100 \
    --train_steps 1000 \
    --val_steps 250 \
    > "$LOG" 2>&1 &

echo "Started PID $! – tail -f $LOG to follow progress"
"""

import os, argparse, datetime, time
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import tensorflow as tf  # 🍭 only used for TFRecord decoding (kept unchanged)
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')


from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from point_transformer_pytorch import PointTransformerLayer
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────
# Main entry‑point – behaviour mirrors the original script
# ──────────────────────────────────────────────────────────────────────────
try:
    from torchinfo import summary as model_summary
except ImportError:
    model_summary = None

# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────
NUM_POINTS   = 35 * 35
MAX_CTRLS_U  = 10
MAX_CTRLS_V  = 10
BATCH_SIZE   = 4
AUTOTUNE     = tf.data.AUTOTUNE
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ──────────────────────────────────────────────────────────────────────────
# Laplacian loss definition
# ──────────────────────────────────────────────────────────────────────────
_LAP_K = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)

def laplacian(grid: torch.Tensor) -> torch.Tensor:
    g = grid.permute(0,3,1,2)
    kernel = _LAP_K.to(g.device).view(1,1,3,3).repeat(g.shape[1],1,1,1)
    lap = F.conv2d(g, kernel, padding=1, groups=g.shape[1])
    return lap.permute(0,2,3,1)

class TotalLoss(nn.Module):
    def __init__(self, w_lap=0.1):
        super().__init__()
        self.w_lap = w_lap
    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        lap = F.mse_loss(laplacian(pred), laplacian(target))
        return mse + self.w_lap * lap

# ──────────────────────────────────────────────────────────────────────────
# Model: SurfaceTransformer
# ──────────────────────────────────────────────────────────────────────────
class SurfaceTransformer(nn.Module):
    def __init__(self,
                 num_points=NUM_POINTS,
                 dim=256,
                 depth=4,
                 pos_mlp_hidden_dim=64,
                 attn_mlp_hidden_mult=4,
                 num_neighbors=32,
                 mlp_dim=512,
                 output_u=MAX_CTRLS_U,
                 output_v=MAX_CTRLS_V):
        super().__init__()
        self.out_u, self.out_v = output_u, output_v
        self.to_embedding = nn.Sequential(
            nn.Linear(3, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.transformer_layers = nn.ModuleList([
            PointTransformerLayer(
                dim=dim,
                pos_mlp_hidden_dim=pos_mlp_hidden_dim,
                attn_mlp_hidden_mult=attn_mlp_hidden_mult,
                num_neighbors=num_neighbors
            ) for _ in range(depth)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim), nn.GELU(),
            nn.Linear(mlp_dim, output_u * output_v * 3)
        )

    def forward(self, xyz: torch.Tensor):
        x = self.to_embedding(xyz)  # (B, N, dim)
        for layer in self.transformer_layers:
            x = layer(x, xyz)
        x = x.mean(dim=1)            # global average pool
        out = self.head(x)
        return out.view(-1, self.out_u, self.out_v, 3)

# ──────────────────────────────────────────────────────────────────────────
# TFRecord I/O
# ──────────────────────────────────────────────────────────────────────────
FEATURE_DESC = {
    'xyz_raw':  tf.io.FixedLenFeature([], tf.string),
    'ctrl_raw': tf.io.FixedLenFeature([], tf.string)
}

def _parse_fn(ex):
    p = tf.io.parse_single_example(ex, FEATURE_DESC)
    xyz = tf.io.parse_tensor(p['xyz_raw'], tf.float32)
    xyz = tf.reshape(xyz, (NUM_POINTS, 3))
    ctrl = tf.io.parse_tensor(p['ctrl_raw'], tf.float32)
    ctrl = tf.reshape(ctrl, (MAX_CTRLS_U, MAX_CTRLS_V, 3))
    return xyz, ctrl

def make_tf_dataset(files, shuffle=True, repeat=True):
    ds = tf.data.TFRecordDataset(files)
    ds = ds.map(_parse_fn, AUTOTUNE)
    if shuffle: ds = ds.shuffle(100_000)
    if repeat: ds = ds.repeat()
    return ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

# ──────────────────────────────────────────────────────────────────────────
# Training & evaluation routines
# ──────────────────────────────────────────────────────────────────────────
def numpy_to_torch(batch):
    xyz_np, ctrl_np = batch
    return torch.from_numpy(xyz_np).to(DEVICE).float(), torch.from_numpy(ctrl_np).to(DEVICE).float()

@torch.no_grad()
def evaluate(model, it, steps):
    model.eval()
    loss_fn, mse_fn = TotalLoss(), nn.MSELoss()
    running_loss, running_mse = 0.0, 0.0
    for step in range(1, steps+1):
        xyz, ctrl = numpy_to_torch(next(it))
        pred = model(xyz)
        running_loss += loss_fn(pred, ctrl).item()
        running_mse  += mse_fn(pred, ctrl).item()
        if step % 500 == 0 or step == steps:
            print(f"  [val] step {step}/{steps}  "
                  f"loss={running_loss/step:.4f}  "
                  f"mse={running_mse/step:.4f}")
    return running_loss/steps, running_mse/steps

def train_epoch(model, it, opt, steps):
    model.train()
    loss_fn, mse_fn = TotalLoss(), nn.MSELoss()
    running_loss, running_mse = 0.0, 0.0
    for step in range(1, steps+1):
        xyz, ctrl = numpy_to_torch(next(it))
        pred = model(xyz)
        loss = loss_fn(pred, ctrl)
        mse  = mse_fn(pred, ctrl)
        opt.zero_grad(); loss.backward(); opt.step()
        running_loss += loss.item(); running_mse += mse.item()
        if step % 500 == 0 or step == steps:
            print(f"  [train] step {step}/{steps}  "
                  f"loss={running_loss/step:.4f}  "
                  f"mse={running_mse/step:.4f}")
    return running_loss/steps, running_mse/steps

# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main(glob_pattern, epochs, train_steps_arg, val_steps_arg):
    files = tf.io.gfile.glob(glob_pattern)
    np.random.seed(42); np.random.shuffle(files)
    split = int(0.8 * len(files))
    train_files, val_files = files[:split], files[split:]
    train_ds = make_tf_dataset(train_files, True, True)
    val_ds   = make_tf_dataset(val_files, False, True)
    train_it, val_it = iter(train_ds.as_numpy_iterator()), iter(val_ds.as_numpy_iterator())

    model = SurfaceTransformer().to(DEVICE)
    #print("Model architecture:"); print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    if model_summary:
        try:
            dummy = torch.zeros(1, NUM_POINTS, 3).to(DEVICE)
            model_summary(model, input_data=(dummy,))
        except:
            print("Skipping detailed torchinfo summary.")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    opt = AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
    warmup = LinearLR(opt, start_factor=1e-2, total_iters=10)
    cosine = CosineAnnealingLR(opt, T_max=epochs-10, eta_min=1e-6)
    sched = SequentialLR(opt, [warmup, cosine], [10])

    # set steps per epoch based on args or defaults
    train_steps = train_steps_arg if train_steps_arg is not None else (600_000 // BATCH_SIZE)
    val_steps   = val_steps_arg   if val_steps_arg   is not None else (100_000 // BATCH_SIZE)

    best_val = float('inf')
    for epoch in range(1, epochs+1):
        t0 = time.time()
        print(f"Epoch {epoch}/{epochs} — train_steps={train_steps}, val_steps={val_steps}")
        tl, tm = train_epoch(model, train_it, opt, train_steps)
        vl, vm = evaluate(model, val_it, val_steps)
        print(f"Epoch {epoch} summary: train_loss={tl:.4f} mse={tm:.4f} "
              f"val_loss={vl:.4f} mse={vm:.4f} time={time.time()-t0:.1f}s")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), "models/best_transformer.pt")
        sched.step()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    torch.save(model.state_dict(), f"models/final_transformer_{stamp}_{epochs}ep.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_glob', required=True)
    parser.add_argument('--epochs',       type=int, default=300)
    parser.add_argument('--train_steps',  type=int, default=None,
                        help='Max training iterations per epoch')
    parser.add_argument('--val_steps',    type=int, default=None,
                        help='Max validation iterations per epoch')
    args = parser.parse_args()
    main(args.tfrecord_glob, args.epochs, args.train_steps, args.val_steps)
