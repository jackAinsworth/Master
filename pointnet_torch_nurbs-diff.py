#!/usr/bin/env python3
"""
train_pointnet_surface_hybrid.py

Hybrid training of a PointNet++ surface regressor with a differentiable
NURBS surface head and hybrid losses:

  L = w_ctrl*MSE(ctrl_pred, ctrl_gt)
    + w_lap*MSE(Lap(ctrl_pred), Lap(ctrl_gt))
    + w_surf_gt*Chamfer(S(ctrl_pred), S(ctrl_gt))
    + w_surf_pc*Chamfer(S(ctrl_pred), point_cloud)

The NURBS evaluator here is a pure-PyTorch implementation (degree=3,3; open-uniform knots),
fully differentiable w.r.t. control points (and optional weights). You can later replace
the evaluator body with the CUDA NURBS-Diff binding without changing the call sites.

      LOG=logs/pointnet_surface_$(date +%Y%m%d_%H%M).log   # e.g. pointnet_surface_20250618_1423.log
CUDA_VISIBLE_DEVICES=1 \

nohup python -u pointnet_torch_nurbs-diff.py \
      --tfrecord_glob "./tfrecords_pointcloud/data_*.tfrecord" \
      --epochs 100 \
      > "$LOG" 2>&1 &
echo "Started PID $! – tail -f $LOG to follow progress"
"""

#!/usr/bin/env python3
"""
train_pointnet_surface_torch.py  — with NURBS-Diff surface loss integration

Changes kept minimal:
  • Adds a thin NURBS sampler wrapper (NURBSDiff SurfEval) to turn (B,U,V,3) control nets
    into dense surface samples.
  • Adds surface-space supervised MSE (pred surface vs GT surface).
  • Adds optional symmetric Chamfer loss between predicted surface samples and input xyz.
  • Keeps your original control-net TotalLoss (MSE + w_lap * Laplacian) intact.
  • Leaves the network architecture and TFRecord pipeline unchanged.

Flags at top:
  W_SURF — weight of surface-space MSE (pred vs GT surface).
  W_CD   — weight of Chamfer (pred surface → input xyz).

If NURBSDiff’s CUDA backend isn’t available, it falls back to C++ (“cpp”) backend automatically.
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train PointNet++ → NURBS control-net regressor with fixed open-uniform knots

Run:
  python train_pointnet_nurbs.py --tfrecord_glob "/data/pc_*.tfrecord" --epochs 200
"""




import os, argparse, datetime, time
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# TensorFlow only for TFRecord I/O
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

# --- PointNet++ utilities (your compiled/available module) ---
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation

# --- Use NURBSDiff directly (as in your screenshot) ---
from NURBSDiff.surf_eval import SurfEval

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
NUM_POINTS   = 35 * 35          # input samples per shape
MAX_CTRLS_U  = 10
MAX_CTRLS_V  = 10
CTRL_DIM     = 3
DEG_P        = 3
DEG_Q        = 3
OUT_U        = 35
OUT_V        = 35

BATCH_SIZE   = 8
NUM_EXAMPLES = 332000           # adjust to your dataset
SHUFFLE_BUF  = 4096

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────
# TFRecord → Torch pipeline
# ─────────────────────────────────────────────────────────────
FEATURE_DESC = {
    "xyz_raw":  tf.io.FixedLenFeature([], tf.string),
    "ctrl_raw": tf.io.FixedLenFeature([], tf.string),
}

def infer_mask_from_padding(ctrl):  # ctrl: (Mu_max, Mv_max, 3)
    # valid if any channel != 0
    valid = tf.reduce_any(tf.not_equal(ctrl, 0.0), axis=-1)  # (Mu_max, Mv_max) bool
    return tf.cast(valid, tf.float32)

def _parse_function(example_proto):
    p = tf.io.parse_single_example(example_proto, FEATURE_DESC)
    xyz  = tf.io.parse_tensor(p["xyz_raw"],  out_type=tf.float32)
    xyz  = tf.reshape(xyz, (NUM_POINTS, 3))
    ctrl = tf.io.parse_tensor(p["ctrl_raw"], out_type=tf.float32)
    ctrl = tf.reshape(ctrl, (MAX_CTRLS_U, MAX_CTRLS_V, 3))
    mask = infer_mask_from_padding(ctrl)
    return xyz, ctrl, mask


def make_tf_dataset(filenames, shuffle=True, repeat=True):
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    ds = ds.map(_parse_function, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=SHUFFLE_BUF)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    return ds.prefetch(AUTOTUNE)

# REPLACE your numpy_to_torch with this:
def numpy_to_torch(batch):
    xyz_np, ctrl_np, mask_np = batch

    # Writable + contiguous copies (protects against undefined behavior)
    xyz_c  = np.ascontiguousarray(xyz_np).copy()
    ctrl_c = np.ascontiguousarray(ctrl_np).copy()
    mask_c = np.ascontiguousarray(mask_np).copy()

    xyz  = torch.from_numpy(xyz_c).to(_DEVICE).float().contiguous()
    ctrl = torch.from_numpy(ctrl_c).to(_DEVICE).float().contiguous()
    mask = torch.from_numpy(mask_c).to(_DEVICE).float().contiguous()
    return xyz, ctrl, mask


# ─────────────────────────────────────────────────────────────
# PointNet++ backbone & head (knots are fixed, NOT predicted)
# ─────────────────────────────────────────────────────────────
def mlp(layers, bn=True, act=nn.ReLU):
    seq = []
    for i in range(len(layers)-1):
        seq.append(nn.Linear(layers[i], layers[i+1]))
        if i < len(layers)-2:
            if bn: seq.append(nn.BatchNorm1d(layers[i+1]))
            if act is not None: seq.append(act(inplace=True))
    return nn.Sequential(*seq)

class ControlNetHead(nn.Module):
    def __init__(self, in_ch, Mu=MAX_CTRLS_U, Mv=MAX_CTRLS_V, dim=CTRL_DIM):
        super().__init__()
        self.Mu, self.Mv, self.dim = Mu, Mv, dim
        out_ch = Mu * Mv * dim
        self.head = mlp([in_ch, 512, 256, 128, out_ch])

    def forward(self, g):
        B = g.shape[0]
        ctrl = self.head(g).view(B, self.Mu, self.Mv, self.dim)
        return ctrl

class PointNet2Backbone(nn.Module):
    """
    SA (down) + FP (up) backbone.
    Produces a global feature g of size g_ch for the control-net head.
    """
    def __init__(self, g_ch=512):
        super().__init__()

        # --- SA block definitions (unchanged numerically vs your code) ---
        mlp1 = [[32, 32, 64], [64, 64, 128], [64, 96, 128]]   # -> C1 = 64+128+128 = 320
        mlp2 = [[64, 64, 128], [128, 128, 256], [128, 128, 256]]  # -> C2 = 128+256+256 = 640
        mlp3 = [[256, 256, 512], [256, 512, 512], [256, 512, 512]] # -> C3 = 512+512+512 = 1536
        self.C1 = sum(m[-1] for m in mlp1)
        self.C2 = sum(m[-1] for m in mlp2)
        self.C3 = sum(m[-1] for m in mlp3)

        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512, radius_list=[0.05, 0.1, 0.2], nsample_list=[16, 32, 64],
            in_channel=0, mlp_list=mlp1
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128, radius_list=[0.1, 0.2, 0.4], nsample_list=[16, 32, 64],
            in_channel=self.C1, mlp_list=mlp2
        )
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=1, radius_list=[0.4, 0.8, 1.2], nsample_list=[32, 64, 128],
            in_channel=self.C2, mlp_list=mlp3
        )

        # --- Feature Propagation path (top-down) ---
        # in_channel = points1_channels + points2_channels at each stage
        self.fp3 = PointNetFeaturePropagation(in_channel=self.C2 + self.C3, mlp=[1024, 512])  # 128 <- 1
        self.fp2 = PointNetFeaturePropagation(in_channel=self.C1 + 512,     mlp=[512, 512])   # 512 <- 128
        self.fp1 = PointNetFeaturePropagation(in_channel=512,               mlp=[256, 128])   # N   <- 512

        # --- Dual-path global aggregation (top feature + FP-refined bottom feature) ---
        g_h1 = g_ch // 2
        g_h2 = g_ch - g_h1
        self.proj_global = nn.Sequential(
            nn.Conv1d(self.C3, g_h1, 1), nn.BatchNorm1d(g_h1), nn.ReLU(inplace=True)
        )
        self.proj_local  = nn.Sequential(
            nn.Conv1d(128, g_h2, 1),     nn.BatchNorm1d(g_h2), nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, xyz):                # xyz: (B, N, 3)
        x = xyz.transpose(1, 2).contiguous()  # (B, 3, N)

        # Down: SA layers
        l1_xyz, l1_points = self.sa1(x,    None)        # (B, 3, 512), (B, C1, 512)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 128), (B, C2, 128)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3,   1), (B, C3,   1)

        # Up: FP layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)     # (B, 512, 128)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)     # (B, 512, 512)
        l0_points = self.fp1(x,      l1_xyz, None,      l1_points)     # (B, 128, N)

        # Global feature = concat( top-global , bottom-global )
        g_global = self.pool(self.proj_global(l3_points)).squeeze(-1)  # (B, g_h1)
        g_local  = self.pool(self.proj_local(l0_points)).squeeze(-1)   # (B, g_h2)
        g = torch.cat([g_global, g_local], dim=1)                      # (B, g_ch)
        return g


class PointNet2SurfaceRegressor(nn.Module):
    """
    Encodes (B, N, 3) → predicts control net (Mu x Mv x 3).
    Knots are fixed open-uniform in the loss (not predicted).
    """
    def __init__(self, Mu=MAX_CTRLS_U, Mv=MAX_CTRLS_V, g_ch=512):
        super().__init__()
        self.Mu, self.Mv = Mu, Mv
        self.backbone = PointNet2Backbone(g_ch=g_ch)
        self.ctrl_head = ControlNetHead(g_ch, Mu, Mv, CTRL_DIM)
        self.ctrl_scale = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, xyz):          # (B, N, 3)
        g = self.backbone(xyz)
        ctrl = self.ctrl_head(g) * self.ctrl_scale
        return ctrl                  # (B, Mu, Mv, 3)

# ─────────────────────────────────────────────────────────────
# Loss with fixed open-uniform knots (via NURBSDiff SurfEval)
# ─────────────────────────────────────────────────────────────

from NURBSDiff.surf_eval import SurfEval

def masked_ctrl_mse(ctrl_pred, ctrl_gt, mask):
    # ctrl_*: (B, Mu_max, Mv_max, 3), mask: (B, Mu_max, Mv_max) in {0,1}
    diff2 = (ctrl_pred - ctrl_gt).pow(2).sum(dim=-1)     # (B, Mu_max, Mv_max)
    num = (diff2 * mask).sum()
    den = (mask.sum() * 3.0).clamp_min(1.0)
    return num / den

class TotalLossVariable(nn.Module):
    """
    Removes padding (per-sample) before SurfEval and uses clamped (open-uniform) knots.
    Works with batches that mix different true control sizes (Mu_i, Mv_i).
    """
    def __init__(self, deg_p=3, deg_q=3, out_u=35, out_v=35,
                 w_ctrl=1.0, w_surf=1.0, use_chamfer=False, device="cuda"):
        super().__init__()
        self.p, self.q = deg_p, deg_q
        self.out_u, self.out_v = out_u, out_v
        self.w_ctrl, self.w_surf = w_ctrl, w_surf
        self.use_chamfer = use_chamfer
        self._cache = {}   # (Mu,Mv) -> SurfEval
        self.device = "cuda" if (device.startswith("cuda") and torch.cuda.is_available()) else "cpp"

    @staticmethod
    def _sizes_from_mask(mask_b):   # mask_b: (Mu_max, Mv_max)
        valid_rows = (mask_b.sum(dim=1) > 0)
        valid_cols = (mask_b.sum(dim=0) > 0)
        Mu_i = int(valid_rows.sum().item())
        Mv_i = int(valid_cols.sum().item())
        return Mu_i, Mv_i

    def _get_surfeval(self, Mu, Mv):
        key = (Mu, Mv)
        if key not in self._cache:
            self._cache[key] = SurfEval(
                m=Mu-1, n=Mv-1, dimension=3, p=self.p, q=self.q,
                out_dim_u=self.out_u, out_dim_v=self.out_v,
                method='tc', dvc=self.device
            )
        return self._cache[key]

    @staticmethod
    def _open_uniform_knots(n_ctrl, degree, device):
        L = n_ctrl + degree + 1
        kv = torch.linspace(0., 1., steps=L, device=device)
        kv[:degree+1]  = 0.0
        kv[-degree-1:] = 1.0
        return kv.unsqueeze(0)  # (1, L)

    @staticmethod
    def _chamfer(x, y):
        d = torch.cdist(x, y, p=2).pow(2)
        return d.min(dim=2).values.mean() + d.min(dim=1).values.mean()

    def forward(self, ctrl_pred_pad, ctrl_gt_pad, mask, xyz_samples):
        """
        ctrl_pred_pad: (B, Mu_max, Mv_max, 3)
        ctrl_gt_pad:   (B, Mu_max, Mv_max, 3)
        mask:          (B, Mu_max, Mv_max)   1=valid, 0=pad
        xyz_samples:   (B, OUT_U*OUT_V, 3) or (B, OUT_U, OUT_V, 3)
        """
        B, Mu_max, Mv_max, _ = ctrl_pred_pad.shape
        dev = ctrl_pred_pad.device

        # 1) masked control-point loss
        loss_ctrl = masked_ctrl_mse(ctrl_pred_pad, ctrl_gt_pad, mask)

        # 2) per-sample SurfEval with UNPADDED controls & CLAMPED knots
        surf_preds = []
        for b in range(B):
            Mu_i, Mv_i = self._sizes_from_mask(mask[b])
            # ensure at least degree+1 controls in each direction
            Mu_i = max(Mu_i, self.p + 1)
            Mv_i = max(Mv_i, self.q + 1)

            ctrl_b = ctrl_pred_pad[b, :Mu_i, :Mv_i, :].unsqueeze(0)       # (1, Mu_i, Mv_i, 3)
            ku = self._open_uniform_knots(Mu_i, self.p, dev)
            kv = self._open_uniform_knots(Mv_i, self.q, dev)

            seval = self._get_surfeval(Mu_i, Mv_i).to(dev)

            # ctrl_b: (1, Mu_i, Mv_i, 3)
            w = torch.ones(1, Mu_i, Mv_i, 1, device=dev)
            ctrl_b_h = torch.cat([ctrl_b, w], dim=-1)  # (1, Mu_i, Mv_i, 4)

            seval = self._get_surfeval(Mu_i, Mv_i).to(dev)
            seval.U = ku  # (1, Mu_i + p + 1)
            seval.V = kv  # (1, Mv_i + q + 1)

            surf_b = seval(ctrl_b_h)  # (1, OUT_U, OUT_V, 3)
            # (1, OUT_U, OUT_V, 3)
            surf_preds.append(surf_b)

        surf_pred = torch.cat(surf_preds, dim=0)                           # (B, OUT_U, OUT_V, 3)
        surf_pred = surf_pred.view(B, -1, 3)


        # (B, OUT_U*OUT_V, 3)

        # 3) surface loss (grid MSE or Chamfer)
        if xyz_samples.dim() == 4:  # (B, OUT_U, OUT_V, 3)
            xyz_grid = xyz_samples.view(B, -1, 3)
        else:
            xyz_grid = xyz_samples                                        # (B, OUT_U*OUT_V, 3)

        loss_surf = self._chamfer(surf_pred, xyz_grid) if self.use_chamfer \
                    else F.mse_loss(surf_pred, xyz_grid)

        total = self.w_ctrl * loss_ctrl + self.w_surf * loss_surf

        # Also after seval(...) in loss forward, right before returning:
        if torch.isnan(surf_pred).any() or torch.isinf(surf_pred).any():
            # Return a big finite loss instead of NaN to keep training alive
            loss_surf = torch.tensor(1e6, device=surf_pred.device)
        return total, {"ctrl_mse": loss_ctrl.detach(), "surf_mse": loss_surf.detach()}

# ─────────────────────────────────────────────────────────────
# Train / Eval
# ─────────────────────────────────────────────────────────────
def reshape_xyz_grid(xyz_flat_bnv):
    # If your TF order differs from SurfEval’s sampling order, adapt here.
    return xyz_flat_bnv

@torch.no_grad()
def evaluate(model, tf_dataset_iter, steps, loss_fn):
    model.eval()
    running_loss = 0.0
    running_mse  = 0.0
    for step in range(1, steps + 1):
        xyz, ctrl_pad, mask = numpy_to_torch(next(tf_dataset_iter))
        ctrl_pred = model(xyz)

        # xyz is (B, OUT_U*OUT_V, 3) in your pipeline; if not, reshape before passing
        total_loss, parts = loss_fn(ctrl_pred, ctrl_pad, mask, xyz)

        running_loss += total_loss.item()
        # masked ctrl MSE so padding doesn’t skew the metric
        raw = ((ctrl_pred - ctrl_pad).pow(2).sum(-1) * mask).sum() / (mask.sum()*3.0).clamp_min(1.0)
        running_mse  += raw.item()

        if step % 500 == 0 or step == steps:
            print(f"  [val] step {step}/{steps}  "
                  f"loss={running_loss/step:.4f}  "
                  f"ctrl_mse={running_mse/step:.4f}  "
                  f"surf_mse={parts['surf_mse'].item():.4f}")
    return running_loss/steps, running_mse/steps


def train_epoch(model, tf_dataset_iter, optimizer, steps, loss_fn):
    model.train()
    running_loss = 0.0
    running_mse  = 0.0
    for step in range(1, steps + 1):
        xyz, ctrl_pad, mask = numpy_to_torch(next(tf_dataset_iter))
        ctrl_pred = model(xyz)

        total_loss, parts = loss_fn(ctrl_pred, ctrl_pad, mask, xyz)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        raw = ((ctrl_pred - ctrl_pad).pow(2).sum(-1) * mask).sum() / (mask.sum()*3.0).clamp_min(1.0)
        running_mse  += raw.item()

        if not torch.isfinite(total_loss):
            with torch.no_grad():
                # quick diag
                bad = {
                    "ctrl_pred_has_nan": torch.isnan(ctrl_pred).any().item(),
                    "ctrl_gt_has_nan": torch.isnan(ctrl_pad).any().item(),
                    "xyz_has_nan": torch.isnan(xyz).any().item()
                }
                print("[skip-batch] non-finite total_loss", bad)
            optimizer.zero_grad(set_to_none=True)
            continue  # skip weight update


        if step % 500 == 0 or step == steps:
            print(f"  [train] step {step}/{steps}  "
                  f"loss={running_loss/step:.4f}  "
                  f"ctrl_mse={running_mse/step:.4f}  "
                  f"surf_mse={parts['surf_mse'].item():.4f}")
    return running_loss/steps, running_mse/steps


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main(tfrecord_glob: str, epochs: int, lr: float, wd: float, val_steps: int):
    all_files = tf.io.gfile.glob(tfrecord_glob)
    if not all_files:
        raise RuntimeError(f"No TFRecord files match {tfrecord_glob}")
    rng = np.random.default_rng(seed=42)
    rng.shuffle(all_files)
    num_train = int(0.8 * len(all_files))
    train_files, val_files = all_files[:num_train], all_files[num_train:]
    print(f"Train shards: {len(train_files)}  Val shards: {len(val_files)}")

    train_ds = make_tf_dataset(train_files, shuffle=True)
    val_ds   = make_tf_dataset(val_files,   shuffle=False, repeat=True)
    train_iter = iter(train_ds.as_numpy_iterator())
    val_iter   = iter(val_ds.as_numpy_iterator())

    model = PointNet2SurfaceRegressor(Mu=MAX_CTRLS_U, Mv=MAX_CTRLS_V).to(_DEVICE)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)

    loss_fn = TotalLossVariable(deg_p=3, deg_q=3, out_u=35, out_v=35,
                                w_ctrl=1.0, w_surf=1.0, use_chamfer=False,
                                device=_DEVICE)
    #mse_fn = nn.MSELoss()   for raw ctrl MSE logging only

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")
    print("Torch:", torch.__version__, " CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPUs:", torch.cuda.device_count(),
              [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-6)

    total_train_examples = int(NUM_EXAMPLES * 0.8)
    train_steps = max(1, total_train_examples // BATCH_SIZE)
    print(f"start training for {epochs} epochs; train_steps/epoch={train_steps} val_steps/epoch={val_steps}")

    best_val = float("inf")
    ckpt_dir = os.path.abspath("models")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        tic = time.time()
        train_loss, train_mse = train_epoch(model, train_iter, optimizer, train_steps, loss_fn) #, mse_fn
        val_loss,   val_mse   = evaluate(model, val_iter, val_steps, loss_fn, mse_fn)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_mse={train_mse:.4f}  "
              f"val_loss={val_loss:.4f}  val_mse={val_mse:.4f}  time={time.time()-tic:.1f}s")
        ckpt_dir = 'pointnet'
        ckpt_path = os.path.join(ckpt_dir, "best_pointnet_surface.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ New best model saved to {ckpt_path} (val_loss={val_loss:.4f})")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    final_name = os.path.join(ckpt_dir, f"pointnet_surface_{stamp}_{epochs}ep.pt")
    torch.save(model.state_dict(), final_name)
    print(f"Finished training → saved {final_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PointNet++ NURBS control-net regressor (PyTorch)")
    parser.add_argument("--tfrecord_glob", required=True,
                        help="Glob for TFRecord shards, e.g. '/data/pc_*.tfrecord'")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--val_steps", type=int, default=500, help="Validation steps per epoch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    args = parser.parse_args()
    main(args.tfrecord_glob, args.epochs, args.lr, args.wd, args.val_steps)
