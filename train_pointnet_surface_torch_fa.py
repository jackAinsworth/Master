#!/usr/bin/env python3
"""
train_pointnet_surface_torch.py

PyTorch re‑implementation of *train_pointnet_surface.py* that:
  • Builds a PointNet++ surface regressor (control‑net output) **in PyTorch**.
  • Consumes TFRecord shards produced from raw point‑cloud data (unchanged pipeline).
  • Trains on 1ormore GPUs via `torch.nn.DataParallel`.

Only the minimal set of changes required to drop the custom TensorFlow
PointNet++ ops have been made; everything else (argument names, config
constants, TFRecord parsing logic, overall training procedure, on‑disk
artifacts) remains identical so existing tooling & logs keep working.

Example:
  python train_pointnet_surface_torch.py \
      --tfrecord_glob "./tfrecords_pointcloud/data_*.tfrecord" \
      --epochs 100


      LOG=logs/pointnet_surface_$(date +%Y%m%d_%H%M).log   # e.g. pointnet_surface_20250618_1423.log
CUDA_VISIBLE_DEVICES=1 \

nohup python -u train_pointnet_surface_torch_fa.py \
      --tfrecord_glob "./tfrecords_pointcloud/data_*.tfrecord" \
      --epochs 100 \
      > "$LOG" 2>&1 &
echo "Started PID $! – tail -f $LOG to follow progress"
"""
import os, argparse, datetime, glob, math, random, itertools, time
from pathlib import Path

import numpy as np
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # quieter TF logs
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import tensorflow as tf  # 🍭 only used for TFRecord decoding (kept unchanged)
#gpus = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
#tf.config.experimental.set_memory_growth(gpus[1], True)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.profiler import profile, schedule, ProfilerActivity
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from pytorch3d.ops import sample_farthest_points, ball_query

# ──────────────────────────────────────────────────────────────────────────
# Configuration — identical to the original script
# ──────────────────────────────────────────────────────────────────────────
NUM_POINTS   = 35 * 35      # points per cloud
MAX_CTRLS_U  = 10
MAX_CTRLS_V  = 10
BATCH_SIZE   =  256 #128            # tune to your GPU memory
SHUFFLE_BUF  =   250_000      # was 50_000 in raw script (kept small for GPU)
AUTOTUNE     = tf.data.AUTOTUNE
NUM_EXAMPLES = 1_660_000

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────
# Laplacian (channel‑wise 4‑neighbour) & loss – ported to PyTorch
# ──────────────────────────────────────────────────────────────────────────
_LAP_K = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)


def laplacian(grid: torch.Tensor) -> torch.Tensor:
    """Channel‑wise 4‑neighbour Laplacian on a (B,H,W,C) tensor."""
    # Convert to NCHW for grouped conv
    grid_nchw = grid.permute(0, 3, 1, 2)  # (B, C, H, W)
    B, C, H, W = grid_nchw.shape
    kernel = _LAP_K.to(grid_nchw.device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    lap = F.conv2d(grid_nchw, kernel, padding=1, groups=C)
    return lap.permute(0, 2, 3, 1)  # back to (B,H,W,C)


class TotalLoss(nn.Module):
    """MSE + w_lap * Laplacian loss (identical maths to TF version)."""

    def __init__(self, w_lap: float = 0.10):
        super().__init__()
        self.w_lap = w_lap

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mse = F.mse_loss(pred, target)
        lap = F.mse_loss(laplacian(pred), laplacian(target))
        return mse + self.w_lap * lap


# ──────────────────────────────────────────────────────────────────────────
# PointNet++ encoder → dense head (PyTorch) – uses utils provided by user
# ──────────────────────────────────────────────────────────────────────────
from pointnet2_utils import (
    PointNetSetAbstractionMsg, PointNetFeaturePropagation
)
from pointnet2_utils import (
    farthest_point_sample,
    index_points,
)


# Helper for pairwise squared distances
def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    # src: (B, M, C), dst: (B, N, C)
    B, M, C = src.shape
    _, N, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(2, 1))  # (B, M, N)
    dist += torch.sum(src ** 2, dim=-1).view(B, M, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, N)
    return dist




class PointNetSetAbstractionKNN(nn.Module):
    """Set abstraction using k-nearest neighbors instead of radius grouping"""

    def __init__(
        self,
        npoint: int,
        k: int,
        in_channel: int,
        mlp_list: list[int],
    ):
        super().__init__()
        self.npoint = npoint
        self.k = k
        self.mlp_convs = nn.ModuleList()
        last_channel = in_channel + 3  # include relative xyz
        for out_ch in mlp_list:
            conv = nn.Sequential(
                nn.Conv2d(last_channel, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            self.mlp_convs.append(conv)
            last_channel = out_ch

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        # xyz: (B, 3, N), points: (B, D, N) or None
        B, _, N = xyz.shape
        # 1. sample points via FPS using pointnet2_utils
        # furthest_point_sample expects input (B, N, C)
        fps_idx = farthest_point_sample(xyz.permute(0, 2, 1), self.npoint)  # (B, npoint)
        new_xyz = index_points(xyz.permute(0, 2, 1), fps_idx)               # (B, npoint, 3)

        # 2. find kNN in the original cloud
        dist = square_distance(new_xyz, xyz.permute(0, 2, 1))              # (B, npoint, N)
        idx = dist.argsort(dim=-1)[:, :, : self.k]                        # (B, npoint, k)

        # 3. group points and relative coords
        grouped_xyz = index_points(xyz.permute(0, 2, 1), idx)             # (B, npoint, k, 3)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)                  # (B, npoint, k, 3)

        if points is not None:
            grouped_points = index_points(points.permute(0, 2, 1), idx)   # (B, npoint, k, D)
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1) # (B, npoint, k, C+D)
        else:
            new_points = grouped_xyz

        # 4. apply MLPs across channels
        new_points = new_points.permute(0, 3, 1, 2)  # (B, C+D, npoint, k)
        for conv in self.mlp_convs:
            new_points = conv(new_points)

        # 5. max pooling over k neighbors
        new_points = torch.max(new_points, dim=-1)[0]   # (B, out_ch, npoint)
        # return in same format as PointNetSetAbstractionMsg
        return new_xyz.permute(0, 2, 1), new_points      # (B,3,npoint),(B,out_ch,npoint)





# ──────────────────────────────────────────────────────────────────────────
# PointNet++ encoder → FP decoder → dense head (PyTorch)
# ──────────────────────────────────────────────────────────────────────────
class PointNet2SurfaceRegressor(nn.Module):
    """PointNet++ with KNN Set Abstraction + Feature Propagation (U‑Net style)"""
    def __init__(self,
                 num_points: int = NUM_POINTS,
                 output_ctrlpts_u: int = MAX_CTRLS_U,
                 output_ctrlpts_v: int = MAX_CTRLS_V):
        super().__init__()
        self.output_u = output_ctrlpts_u
        self.output_v = output_ctrlpts_v

        # (0) Embed raw XYZ -> low-level features
        self.input_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # (1) Encoder: four SA-KNN blocks (same as before)
        self.sa1 = PointNetSetAbstractionKNN(num_points // 2,  32, 64,  [64, 64, 128])   # -> 128 @ N/2
        self.sa2 = PointNetSetAbstractionKNN(num_points // 4,  32, 128, [128, 128, 256]) # -> 256 @ N/4
        self.sa3 = PointNetSetAbstractionKNN(num_points // 16, 32, 256, [256, 256, 512]) # -> 512 @ N/16
        self.sa4 = PointNetSetAbstractionKNN(num_points // 32, 32, 512, [512, 512, 1024])# -> 1024 @ N/32

        # (2) Decoder: Feature Propagation (upsample and fuse skip connections)
        # in_channel = D_skip + D_coarse at each stage (see comments)
        self.fp4 = PointNetFeaturePropagation(in_channel=512 + 1024, mlp=[512, 512])   # (l3<-l4)
        self.fp3 = PointNetFeaturePropagation(in_channel=256 + 512,  mlp=[512, 256])   # (l2<-l3)
        self.fp2 = PointNetFeaturePropagation(in_channel=128 + 256,  mlp=[256, 128])   # (l1<-l2)
        self.fp1 = PointNetFeaturePropagation(in_channel=64  + 128,  mlp=[128, 128, 128]) # (l0<-l1)

        # (3) Dense head (input = 1024 + 128 + 128 = 1280)
        self.global_in_ch = 1024 + 128 + 128
        self.mlp_head = nn.Sequential(
            nn.Linear(self.global_in_ch, 1024), nn.ReLU(inplace=True), nn.BatchNorm1d(1024), nn.Dropout(0.10),
            nn.Linear(1024, 512),              nn.ReLU(inplace=True), nn.BatchNorm1d(512),  nn.Dropout(0.10),
        )
        self.out_linear = nn.Linear(512, output_ctrlpts_u * output_ctrlpts_v * 3)

    def forward(self, xyz_b_n3: torch.Tensor):
        # Input xyz: (B,N,3)  -> Transpose to (B,3,N)
        xyz = xyz_b_n3.permute(0, 2, 1)         # l0_xyz: (B,3,N)
        l0_points = self.input_mlp(xyz)         # (B,64,N)

        # Encoder (downsample)
        l1_xyz, l1_points = self.sa1(xyz,      l0_points)  # (B,3,N/2), (B,128,N/2)
        l2_xyz, l2_points = self.sa2(l1_xyz,   l1_points)  # (B,3,N/4), (B,256,N/4)
        l3_xyz, l3_points = self.sa3(l2_xyz,   l2_points)  # (B,3,N/16),(B,512,N/16)
        l4_xyz, l4_points = self.sa4(l3_xyz,   l3_points)  # (B,3,N/32),(B,1024,N/32)

        # Decoder (upsample with FP)
        l3_points_up = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B,512,N/16)
        l2_points_up = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points_up)  # (B,256,N/4)
        l1_points_up = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_up)  # (B,128,N/2)
        l0_points_up = self.fp1(xyz,    l1_xyz, l0_points, l1_points_up)  # (B,128,N)

        # Global descriptor: deep + full-res statistics
        g4_max  = torch.max(l4_points, dim=2)[0]   # (B,1024)
        g0_max  = torch.max(l0_points_up, dim=2)[0]# (B,128)
        g0_mean = torch.mean(l0_points_up, dim=2)  # (B,128)
        global_feat = torch.cat([g4_max, g0_max, g0_mean], dim=1)  # (B,1280)

        # Dense head → control net
        x = self.mlp_head(global_feat)             # (B,512)
        ctrl = self.out_linear(x).view(-1, self.output_u, self.output_v, 3)  # (B,U,V,3)
        return ctrl


# ──────────────────────────────────────────────────────────────────────────
# TFRecord → torch data pipeline (kept identical at the *TFR* level)
# ──────────────────────────────────────────────────────────────────────────
FEATURE_DESC = {
    "xyz_raw":  tf.io.FixedLenFeature([], tf.string),
    "ctrl_raw": tf.io.FixedLenFeature([], tf.string),
}


def _parse_function(example_proto):
    parsed = tf.io.parse_single_example(example_proto, FEATURE_DESC)
    xyz = tf.io.parse_tensor(parsed["xyz_raw"],  out_type=tf.float32)
    xyz = tf.reshape(xyz, (NUM_POINTS, 3))
    ctrl = tf.io.parse_tensor(parsed["ctrl_raw"], out_type=tf.float32)
    ctrl = tf.reshape(ctrl, (MAX_CTRLS_U, MAX_CTRLS_V, 3))
    return xyz, ctrl


def make_tf_dataset(filenames, shuffle=True, repeat=True):
    ds = tf.data.TFRecordDataset(filenames)
    ds = ds.map(_parse_function, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=SHUFFLE_BUF)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    return ds.prefetch(AUTOTUNE)


# ──────────────────────────────────────────────────────────────────────────
# Training / evaluation helpers
# ──────────────────────────────────────────────────────────────────────────

def numpy_to_torch(batch):
    xyz_np, ctrl_np = batch
    xyz = torch.from_numpy(xyz_np).to(_DEVICE).float()
    ctrl = torch.from_numpy(ctrl_np).to(_DEVICE).float()
    return xyz, ctrl


@torch.no_grad()
def evaluate(model, tf_dataset_iter, steps):
    """
    Validation loop that returns both total loss and raw MSE.
    """
    model.eval()
    loss_fn = TotalLoss()
    mse_fn  = nn.MSELoss()
    running_loss = 0.0
    running_mse  = 0.0

    for step in range(1, steps + 1):
        xyz, ctrl = numpy_to_torch(next(tf_dataset_iter))
        pred = model(xyz)

        running_loss += loss_fn(pred, ctrl).item()
        running_mse  += mse_fn(pred, ctrl).item()

        if step % 500 == 0 or step == steps:
            print(f"  [val] step {step}/{steps}  "
                  f"loss={running_loss/step:.4f}  "
                  f"mse={running_mse/step:.4f}")

    return running_loss / steps, running_mse / steps







def train_epoch(model, tf_dataset_iter, optimizer, steps):
    """
    One training epoch with step-wise prints of both total loss and raw MSE.
    """
    model.train()
    loss_fn = TotalLoss()
    mse_fn  = nn.MSELoss()
    running_loss = 0.0
    running_mse  = 0.0

    for step in range(1, steps + 1):
        xyz, ctrl = numpy_to_torch(next(tf_dataset_iter))
        pred = model(xyz)

        total_loss = loss_fn(pred, ctrl)
        raw_mse    = mse_fn(pred, ctrl)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        running_mse  += raw_mse.item()

        # Print every 500 steps (and on the last step)
        if step % 500 == 0 or step == steps:
            print(f"  [train] step {step}/{steps}  "
                  f"loss={running_loss/step:.4f}  "
                  f"mse={running_mse/step:.4f}")

    return running_loss / steps, running_mse / steps




# ──────────────────────────────────────────────────────────────────────────
# Main entry‑point – behaviour mirrors the original script
# ──────────────────────────────────────────────────────────────────────────
try:
    from torchinfo import summary as model_summary
except ImportError:
    model_summary = None

def main(tfrecord_glob: str, epochs: int):
    # ─── Shard split (identical to original) ─────────────────────
    all_files = tf.io.gfile.glob(tfrecord_glob)
    if not all_files:
        raise RuntimeError(f"No TFRecord files match {tfrecord_glob}")

    rng = np.random.default_rng(seed=42)
    rng.shuffle(all_files)
    num_train = int(0.8 * len(all_files))
    train_files, val_files = all_files[:num_train], all_files[num_train:]
    print(f"Train shards: {len(train_files)}  Val shards: {len(val_files)}")

    # ─── Datasets as numpy iterators ────────────────────────────
    train_ds = make_tf_dataset(train_files, shuffle=True)
    val_ds = make_tf_dataset(val_files, shuffle=False, repeat=True)
    train_iter = iter(train_ds.as_numpy_iterator())
    val_iter = iter(val_ds.as_numpy_iterator())

    # ─── Model & optimisers ─────────────────────────────────────
    model = PointNet2SurfaceRegressor().to(_DEVICE)

    #print(" Model Architecture: ")
    #print(model)

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if model_summary is not None:
    # torchinfo expects input_size matching forward signature: (batch_size, N, 3)
        print("Detailed summary with torchinfo:")
        model_summary(model, input_size=(1, NUM_POINTS, 3))
    else:
        print("Install 'torchinfo' for a detailed layer-by-layer summary.")

    print("device count ", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = AdamW(model.parameters(), lr=2e-2, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-6)

    # originally
    #total_train_examples = int(NUM_EXAMPLES * 0.8)
    #train_steps = total_train_examples // BATCH_SIZE  # 10 375

    # suggestion: see only 600 000 examples / epoch
    train_steps = 600_000 // BATCH_SIZE  # 4 687
    val_steps = 100_000 // BATCH_SIZE
    best_val = float('inf')

    #train_steps = 60_000 // BATCH_SIZE  # 4 687
    #val_steps = 10_000 // BATCH_SIZE


    print(f"start training for {epochs} epochs")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Profile the training epoch (no schedule, records entire block)
        """  with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True) as prof:
            train_loss, train_mse = train_epoch(model, train_iter, optimizer, train_steps)
            print("test 1")
            """

        train_loss, train_mse = train_epoch(model, train_iter, optimizer, train_steps)

        # Print the top 5 CUDA kernels by total time
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))

        # Run validation
        val_loss, val_mse = evaluate(model, val_iter, val_steps)
        scheduler.step(val_loss)

        # Summary printout
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:03d}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_mse={train_mse:.4f}  "
              f"val_loss={val_loss:.4f}  val_mse={val_mse:.4f}  "
              f"time={epoch_time:.1f}s")

        ckpt_path = f"models/best_pointnet_surface_{epochs}.weights.h5"
        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ New best model saved to {ckpt_path} (val_loss={val_loss:.4f})")

    # ─── Final full-model export (.pt eager module) ─────────────
    # Generate a timestamp and filename
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_dir = Path("models")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    final_name = ckpt_dir / f"pointnet_surface_{stamp}_{epochs}ep.pt"

    # Save the model’s state_dict
    torch.save(model.state_dict(), final_name)
    print(f"Finished training → saved {final_name}")







# ──────────────────────────────────────────────────────────────────────────
# CLI glue – mirrors original argument names
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # unchanged default; can override via CLI
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description="Train PointNet++ surface regressor (PyTorch)")
    parser.add_argument("--tfrecord_glob", required=True,
                        help="Glob pattern for TFRecord shards, e.g. '/data/pc_*.tfrecord'")
    parser.add_argument("--epochs", type=int, default=250)
    args = parser.parse_args()

    # Quick device / build info (parity with original printouts)
    print("Torch:", torch.__version__, " CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    print("TensorFlow (for TFRecord I/O only):", tf.__version__)

    main(args.tfrecord_glob, args.epochs)
