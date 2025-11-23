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
  python surface_transformer.py \
      --tfrecord_glob "./tfrecords_pointcloud/data_*.tfrecord" \
      --epochs 100

LOG=logs/surface_transformer_$(date +%Y%m%d_%H%M).log   # e.g. pointnet_surface_20250618_1423.log
CUDA_VISIBLE_DEVICES=1 \

nohup python -u surface_transformer.py \
      --tfrecord_glob "./tfrecords_pointcloud/data_*.tfrecord" \
      --epochs 100 \
      > "$LOG" 2>&1 &
echo "Started PID $! – tail -f $LOG to follow progress"
"""
import os, argparse, datetime, time
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np
import tensorflow as tf    # for TFRecord I/O
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────
NUM_POINTS   = 35 * 35
MAX_CTRLS_U  = 10
MAX_CTRLS_V  = 10
BATCH_SIZE   = 4 #64
AUTOTUNE     = tf.data.AUTOTUNE
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ──────────────────────────────────────────────────────────────────────────
# Laplacian loss (unchanged)
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
# Transformer surface regressor
# ──────────────────────────────────────────────────────────────────────────
class SurfaceTransformer(nn.Module):
    def __init__(self,
                 num_points=NUM_POINTS,
                 d_model=256,
                 nhead=8,
                 num_layers=6,
                 dim_ff=512,
                 output_u=MAX_CTRLS_U,
                 output_v=MAX_CTRLS_V):
        super().__init__()
        self.num_points = num_points
        self.out_u, self.out_v = output_u, output_v
        # embed xyz into d_model dims
        self.xyz_embed = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model)
        )
        # positional encoding from coordinates (optional)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model)
        )
        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        # head to control points
        self.head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, output_u*output_v*3)
        )

    def forward(self, xyz: torch.Tensor):
        # xyz: (B, N, 3)
        # embed
        token = self.xyz_embed(xyz)                    # (B,N,d)
        pos   = self.pos_embed(xyz)                    # (B,N,d)
        x = token + pos                                # (B,N,d)
        # transformer
        x = self.transformer(x)                        # (B,N,d)
        # global pool
        x = x.mean(dim=1)                              # (B,d)
        # head
        out = self.head(x)                             # (B, U*V*3)
        return out.view(-1, self.out_u, self.out_v, 3)

# ──────────────────────────────────────────────────────────────────────────
# TFRecord I/O (unchanged)
# ──────────────────────────────────────────────────────────────────────────
FEATURE_DESC = {
    'xyz_raw': tf.io.FixedLenFeature([], tf.string),
    'ctrl_raw': tf.io.FixedLenFeature([], tf.string)
}

def _parse_fn(ex):
    p = tf.io.parse_single_example(ex, FEATURE_DESC)
    xyz = tf.io.parse_tensor(p['xyz_raw'], tf.float32)
    xyz = tf.reshape(xyz, (NUM_POINTS,3))
    ctrl = tf.io.parse_tensor(p['ctrl_raw'], tf.float32)
    ctrl = tf.reshape(ctrl, (MAX_CTRLS_U,MAX_CTRLS_V,3))
    return xyz, ctrl


def make_tf_dataset(files, shuffle=True, repeat=True):
    ds = tf.data.TFRecordDataset(files)
    ds = ds.map(_parse_fn, AUTOTUNE)
    if shuffle: ds = ds.shuffle(100000)
    if repeat: ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    return ds.prefetch(AUTOTUNE)

# ──────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────
def numpy_to_torch(batch):
    xyz_np, ctrl_np = batch
    return (
        torch.from_numpy(xyz_np).to(DEVICE).float(),
        torch.from_numpy(ctrl_np).to(DEVICE).float()
    )

@torch.no_grad()
def evaluate(model, it, steps):
    model.eval()
    loss_fn = TotalLoss()
    mse_fn  = nn.MSELoss()
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
    loss_fn = TotalLoss()
    mse_fn  = nn.MSELoss()
    running_loss, running_mse = 0.0, 0.0
    for step in range(1, steps+1):
        xyz, ctrl = numpy_to_torch(next(it))
        pred = model(xyz)
        total_loss = loss_fn(pred, ctrl)
        raw_mse    = mse_fn(pred, ctrl)
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        running_loss += total_loss.item()
        running_mse  += raw_mse.item()
        if step % 500 == 0 or step == steps:
            print(f"  [train] step {step}/{steps}  "
                  f"loss={running_loss/step:.4f}  "
                  f"mse={running_mse/step:.4f}")
    return running_loss/steps, running_mse/steps


def main(glob_pattern, epochs):
    files = tf.io.gfile.glob(glob_pattern)
    np.random.seed(42)
    np.random.shuffle(files)
    split = int(0.8 * len(files))
    train_f, val_f = files[:split], files[split:]
    train_ds = make_tf_dataset(train_f, True, True)
    val_ds   = make_tf_dataset(val_f, False, True)
    train_it, val_it = iter(train_ds.as_numpy_iterator()), iter(val_ds.as_numpy_iterator())

    model = SurfaceTransformer().to(DEVICE)
    print("Model architecture:")
    print(model)
    try:
        model_summary(model, input_size=(1, NUM_POINTS, 3))
    except Exception:
        print("Install torchinfo for detailed summary.")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    opt = AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
    warmup = LinearLR(opt, start_factor=1e-2, total_iters=10)
    cosine = CosineAnnealingLR(opt, T_max=epochs-10, eta_min=1e-6)
    sched = SequentialLR(opt, [warmup, cosine], [10])

    train_steps = 600_000 // BATCH_SIZE
    val_steps   = 100_000 // BATCH_SIZE
    best_val = float('inf')

    for epoch in range(1, epochs+1):
        t0 = time.time()
        sched.step()
        print(f"Epoch {epoch}/{epochs}")
        train_loss, train_mse = train_epoch(model, train_it, opt, train_steps)
        val_loss, val_mse     = evaluate(model, val_it, val_steps)
        print(f"Epoch {epoch} summary: train_loss={train_loss:.4f} mse={train_mse:.4f} "
              f"val_loss={val_loss:.4f} mse={val_mse:.4f} time={time.time()-t0:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "models/best_transformer.pt")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    torch.save(model.state_dict(), f"models/final_transformer_{stamp}_{epochs}ep.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_glob', required=True)
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()
    main(args.tfrecord_glob, args.epochs)

