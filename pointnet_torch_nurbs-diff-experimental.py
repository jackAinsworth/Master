#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
      LOG=logs/pointnet_surface_$(date +%Y%m%d_%H%M).log   # e.g. pointnet_surface_20250618_1423.log
CUDA_VISIBLE_DEVICES=1 \

nohup python -u pointnet_torch_nurbs-diff.py \
      --tfrecord_glob "./tfrecords_pointcloud/data_*.tfrecord" \
      --epochs 100 \
      > "$LOG" 2>&1 &
echo "Started PID $! – tail -f $LOG to follow progress"
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from NURBSDiff import surf_eval  # your CUDA NURBS-Diff decoder

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = np.finfo(np.float32).eps


# ---------------------------------------------------------------------
# Encoder utils (vectorized kNN + edge features)
# ---------------------------------------------------------------------
@torch.no_grad()
def knn_fast(x, k):
    """
    x: (B, C, N)
    returns idx: (B, N, k) of nearest neighbors (including self if k>=1)
    """
    # pairwise distances (B, N, N)
    dists = torch.cdist(x.transpose(1, 2), x.transpose(1, 2))  # (B, N, N)
    idx = dists.topk(k, largest=False)[1]  # (B, N, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    x: (B, C, N)
    returns: edge features (B, 2C, N, k)
    """
    B, C, N = x.shape
    if idx is None:
        idx = knn_fast(x, k=k)  # (B, N, k)

    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N  # (B,1,1)
    idx = (idx + idx_base).reshape(-1)  # (B*N*k,)

    x_t = x.transpose(2, 1).contiguous()               # (B, N, C)
    features = x_t.reshape(B * N, C)[idx, :]           # (B*N*k, C)
    features = features.view(B, N, k, C)               # (B, N, k, C)

    x_central = x_t.unsqueeze(2).expand(-1, -1, k, -1) # (B, N, k, C)
    feature = torch.cat((features - x_central, x_central), dim=3)  # (B, N, k, 2C)
    feature = feature.permute(0, 3, 1, 2).contiguous()             # (B, 2C, N, k)
    return feature


# ---------------------------------------------------------------------
# DGCNN Control-Point Regressor
# ---------------------------------------------------------------------
class DGCNNControlPoints(nn.Module):
    def __init__(self, num_control_points, k_neighbors=20, mode=1):
        """
        Control points prediction network. Takes points as input (B,3,N)
        and outputs control points grid flattened to (B, M, 3), M=grid^2.
        """
        super().__init__()
        self.k = k_neighbors
        self.mode = mode
        self.controlpoints = num_control_points  # grid size (e.g., 20)

        if self.mode == 0:
            ch1, ch2, ch3, ch4 = 64, 64, 128, 256
        else:
            ch1, ch2, ch3, ch4 = 128, 256, 256, 512

        self.bn1 = nn.BatchNorm2d(ch1)
        self.bn2 = nn.BatchNorm2d(ch2)
        self.bn3 = nn.BatchNorm2d(ch3)
        self.bn4 = nn.BatchNorm2d(ch4)
        self.bn5 = nn.BatchNorm1d(1024)

        in1 = 6                     # edge feat 2*C with C=3 → 6
        in2 = ch1 * 2
        in3 = ch2 * 2
        in4 = ch3 * 2

        self.conv1 = nn.Sequential(nn.Conv2d(in1, ch1, 1, bias=False), self.bn1, nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(in2, ch2, 1, bias=False), self.bn2, nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(in3, ch3, 1, bias=False), self.bn3, nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(in4, ch4, 1, bias=False), self.bn4, nn.LeakyReLU(0.2))

        # feature fusion (x1,x2,x3,x4) → (B, ch1+ch2+ch3+ch4, N)
        self.conv5 = nn.Sequential(
            nn.Conv1d(ch1 + ch2 + ch3 + ch4, 1024, 1, bias=False),
            self.bn5,
            nn.LeakyReLU(0.2)
        )

        self.conv6 = nn.Conv1d(1024, 1024, 1)
        self.bn6 = nn.BatchNorm1d(1024)
        self.conv7 = nn.Conv1d(1024, 1024, 1)
        self.bn7 = nn.BatchNorm1d(1024)

        self.conv8 = nn.Conv1d(1024, 3 * (self.controlpoints ** 2), 1)
        self.drop = 0.0
        self.tanh = nn.Tanh()

    def forward(self, x, weights=None):
        """
        x: (B, 3, N)
        """
        B, _, N = x.shape

        f = get_graph_feature(x, k=self.k)      # (B, 6, N, k)
        f = self.conv1(f).max(dim=-1)[0]        # (B, ch1, N)
        x1 = f

        f = get_graph_feature(x1, k=self.k)
        f = self.conv2(f).max(dim=-1)[0]
        x2 = f

        f = get_graph_feature(x2, k=self.k)
        f = self.conv3(f).max(dim=-1)[0]
        x3 = f

        f = get_graph_feature(x3, k=self.k)
        f = self.conv4(f).max(dim=-1)[0]
        x4 = f

        f = torch.cat((x1, x2, x3, x4), dim=1)  # (B, ch1+ch2+ch3+ch4, N)
        f = self.conv5(f)                       # (B, 1024, N)

        if isinstance(weights, torch.Tensor):
            f = f * weights.reshape((B, 1, -1))

        f = F.adaptive_max_pool1d(f, 1).view(B, -1)  # (B, 1024)
        f = f.unsqueeze(2)                           # (B, 1024, 1)
        f = F.dropout(F.relu(self.bn6(self.conv6(f))), self.drop, training=self.training)
        f = F.dropout(F.relu(self.bn7(self.conv7(f))), self.drop, training=self.training)
        out = self.conv8(f)                          # (B, 3*M, 1)
        out = self.tanh(out[:, :, 0]).view(B, self.controlpoints * self.controlpoints, 3)
        return out


# ---------------------------------------------------------------------
# Laplacian regularizer on control nets (4-neighbour)
# ---------------------------------------------------------------------
def laplacian_4n(ctrl_grid_b_uv3: torch.Tensor) -> torch.Tensor:
    """
    ctrl_grid_b_uv3: (B, U, V, 3)
    returns Laplacian with same shape
    """
    kern = torch.tensor([[0, 1, 0],
                         [1,-4, 1],
                         [0, 1, 0]], dtype=ctrl_grid_b_uv3.dtype, device=ctrl_grid_b_uv3.device)
    x = ctrl_grid_b_uv3.permute(0, 3, 1, 2)  # (B, 3, U, V)
    k = kern.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    lap = F.conv2d(x, k, padding=1, groups=3)
    return lap.permute(0, 2, 3, 1)


# ---------------------------------------------------------------------
# Training script
# ---------------------------------------------------------------------
def main():
    # ---------------- data ----------------
    path = "closed_splines.h5"
    with h5py.File(path, "r") as hf:
        input_points = np.array(hf["points"]).astype(np.float32)          # (B, N, 3)
        input_control_points = np.array(hf["controlpoints"]).astype(np.float32)  # (B, 20, 20, 3)

    # simple train split
    train_data = input_points[:80]
    train_control_points = input_control_points[:80]

    num_epochs    = 1000
    batch_size    = 8
    learning_rate = 3e-4

    grid_size = 20     # 20x20 control net
    deg_u = deg_v = 2  # matches your decoder init
    out_u = out_v = 40 # surface samples

    # ---------------- model ----------------
    encoder = DGCNNControlPoints(grid_size, k_neighbors=20, mode=1).to(device)
    decoder = SurfEval(grid_size, grid_size, dimension=3,
                       p=deg_u, q=deg_v,
                       out_dim_u=out_u, out_dim_v=out_v,
                       method='tc', dvc='cuda' if device.type == 'cuda' else 'cpu').to(device)

    # only encoder has params; SurfEval is an op
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    # hybrid loss weights
    w_ctrl, w_lap, w_surf_gt, w_surf_pc = 1.0, 0.10, 1.0, 1.0
    mse = nn.MSELoss()

    # ---------------- training ----------------
    encoder.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        epoch_loss = 0.0

        for i in range(0, len(train_data), batch_size):
            optimizer.zero_grad()

            pts_np   = train_data[i:i+batch_size]                 # (B,N,3)
            ctrl_np  = train_control_points[i:i+batch_size]       # (B,20,20,3)

            # torch tensors
            pts  = torch.from_numpy(pts_np).to(device)            # (B,N,3)
            ctrl = torch.from_numpy(ctrl_np).to(device)           # (B,20,20,3)

            B, N, _ = pts.shape
            # encoder expects (B,3,N); if you want to cap N (as your original 0:700), do it safely:
            max_n = min(N, 700)
            x_enc = pts[:, :max_n, :].permute(0, 2, 1).contiguous()  # (B,3,max_n)

            # ----- forward -----
            pred_ctrl_flat = encoder(x_enc)                 # (B, 400, 3)
            gt_ctrl_flat   = ctrl.view(B, -1, 3)            # (B, 400, 3)

            # control-net losses
            L_ctrl = mse(pred_ctrl_flat, gt_ctrl_flat)
            L_lap  = mse(
                laplacian_4n(pred_ctrl_flat.view(B, grid_size, grid_size, 3)),
                laplacian_4n(gt_ctrl_flat.view(B, grid_size, grid_size, 3))
            )

            # homogeneous coordinates (w=1, constant)
            ones = torch.ones((B, grid_size, grid_size, 1), device=device)
            pred_ctrl_h = torch.cat([pred_ctrl_flat.view(B, grid_size, grid_size, 3), ones], dim=-1)
            gt_ctrl_h   = torch.cat([gt_ctrl_flat.view(B, grid_size, grid_size, 3), ones], dim=-1)

            # NURBS-Diff decode to surfaces (B, out_u*out_v, 3)
            S_pred = decoder(pred_ctrl_h).view(B, out_u * out_v, 3)
            S_gt   = decoder(gt_ctrl_h).view(B, out_u * out_v, 3)

            # Chamfer vs GT surface
            cd_gt, _ = chamfer_distance(S_pred, S_gt)

            # Chamfer vs input point cloud
            cd_pc, _ = chamfer_distance(S_pred, pts)  # pts is (B,N,3)

            # total hybrid loss
            loss = w_ctrl*L_ctrl + w_lap*L_lap + w_surf_gt*cd_gt + w_surf_pc*cd_pc

            # ----- backward -----
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss)

            if (i // batch_size) % 10 == 0:
                print(f"  step {i:03d}  "
                      f"loss={loss.item():.4f}  "
                      f"ctrl={L_ctrl.item():.4f}  lap={L_lap.item():.4f}  "
                      f"cd_gt={cd_gt.item():.4f}  cd_pc={cd_pc.item():.4f}")

        print(f"Epoch {epoch} avg loss = {epoch_loss / max(1, len(train_data)//batch_size):.4f}")


if __name__ == "__main__":
    main()
