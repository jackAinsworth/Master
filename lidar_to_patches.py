#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiDAR point cloud → 32 patches of shape (50, 50, 3) with UV grids and AABB normalization.

All user-tunable variables are declared in UPPERCASE below.
This script can read: .laz/.las (via laspy + lazrs/laszip), .csv, .npy, .npz

Outputs:
- RAW NPZ:    per-patch raw XYZ grids    -> patches_xyz (32, 50, 50, 3)
- NORM NPZ:   per-patch AABB-normalized  -> patches_norm (32, 50, 50, 3)
- Both NPZs also include: uv_grids, tile_bounds, patch_grid, grid_shape, points_per_tile
- NORM NPZ additionally includes: aabbs (per-patch [min,max] in XYZ)

Usage (example):
    python lidar_to_patches.py

If you want to specify a different input or change behavior, edit the CAPS variables below.
"""

from __future__ import annotations

import os
import sys
import math
import numpy as np
from typing import Tuple, Optional

# =============================
# ======= USER SETTINGS =======
# =============================

# Input file path (supports .laz/.las/.csv/.npy/.npz)
INPUT_PATH: str = "/home/ainsworth/master/lidar/points (2).laz"

# Output files (NPZ)
OUTPUT_RAW_NPZ: str = "/home/ainsworth/master/lidar/patches/raw/patches_50x50_RAW.npz"
OUTPUT_NORM_NPZ: str = "/home/ainsworth/master/lidar/patches/patches_50x50_AABB.npz"

# Patch grid & resampling
NUM_PATCHES: int = 32           # total patches (factored into NX x NY automatically if GRID_MODE='auto')
GRID_SIZE: int = 50             # 50 -> each patch is 50x50 cells (downsampling resolution)

# Choose how to split into tiles
# 'auto' -> choose (NX, NY) to match XY aspect; 'manual' -> use NX_MANUAL, NY_MANUAL below
GRID_MODE: str = "auto"         # "auto" or "manual"
NX_MANUAL: int = 8              # used only if GRID_MODE == "manual"
NY_MANUAL: int = 4              # used only if GRID_MODE == "manual"

# Aggregation of points per cell
# For x,y we always use mean; for z you can choose "mean" or "median"
AGGREGATION_Z: str = "mean"     # "mean" or "median"

# Empty-cell Z fill strategy
EMPTY_Z_FILL: str = "nearest"   # "nearest" or "global_mean"

# Normalization mode
PER_PATCH_AABB: bool = True     # True = per-patch AABB to [0,1]; False = global AABB across all patches

# CSV reader options (only used when INPUT_PATH ends with .csv)
CSV_DELIMITER: str = ","
CSV_HAS_HEADER: bool = True     # If True, the first line is a header; we will try to auto-detect X/Y/Z columns by name
CSV_X_NAME_CANDIDATES = ("x","X","X_coordinate","X_COORDINATE")
CSV_Y_NAME_CANDIDATES = ("y","Y","Y_coordinate","Y_COORDINATE")
CSV_Z_NAME_CANDIDATES = ("z","Z","Z_coordinate","Z_COORDINATE")
# If headerless or names not found, fallback to these indices (0-based):
CSV_XYZ_COLS_FALLBACK = (0, 1, 2)

# Save toggles (usually leave True)
SAVE_RAW_PATCHES: bool = True
SAVE_NORMALIZED_PATCHES: bool = True

# =============================
# ===== END USER SETTINGS =====
# =============================


def _try_import_laspy():
    try:
        import laspy  # type: ignore
        return laspy, None
    except Exception as e:
        return None, e


def _read_laz_or_las(path: str) -> np.ndarray:
    """Read .laz/.las using laspy (scaled coordinates)."""
    laspy, err = _try_import_laspy()
    if laspy is None:
        raise RuntimeError(
            "Cannot import laspy. Install with: pip install 'laspy[lazrs]'\n"
            f"Original import error: {err}"
        )
    try:
        las = laspy.read(path)
    except Exception as e:
        raise RuntimeError(
            "Failed to read .laz/.las. For .laz you need 'lazrs' or 'laszip' backends.\n"
            "Try: pip install 'laspy[lazrs]'\n"
            f"Reader error: {e}"
        )
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    return np.stack([x, y, z], axis=1)


def _infer_csv_xyz_indices_from_header(header_line: str, delimiter: str) -> Tuple[int,int,int]:
    cols = [c.strip() for c in header_line.strip().split(delimiter)]
    lower = [c.lower() for c in cols]
    def find_any(cands):
        for name in cands:
            name_l = name.lower()
            if name_l in lower:
                return lower.index(name_l)
        return None
    ix = find_any(CSV_X_NAME_CANDIDATES)
    iy = find_any(CSV_Y_NAME_CANDIDATES)
    iz = find_any(CSV_Z_NAME_CANDIDATES)
    # If any missing, fallback to default (0,1,2)
    if ix is None or iy is None or iz is None:
        return CSV_XYZ_COLS_FALLBACK
    return (ix, iy, iz)


def _read_csv_xyz(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        if CSV_HAS_HEADER:
            ix, iy, iz = _infer_csv_xyz_indices_from_header(first, CSV_DELIMITER)
            data = np.genfromtxt(
                f, delimiter=CSV_DELIMITER, dtype=np.float64, autostrip=True
            )
        else:
            # Rewind to include first line in data
            f.seek(0)
            ix, iy, iz = CSV_XYZ_COLS_FALLBACK
            data = np.loadtxt(f, delimiter=CSV_DELIMITER, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    xyz = data[:, [ix, iy, iz]]
    return xyz


def _read_npy_npz(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("Expected .npy shaped (N,3+) with columns [x,y,z,...].")
        return np.asarray(arr[:, :3], dtype=np.float64)
    else:
        # .npz
        dat = np.load(path)
        # Try common keys
        for key in ("xyz", "points", "XY Z".replace(" ", ""), "data"):
            if key in dat:
                arr = dat[key]
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    return np.asarray(arr[:, :3], dtype=np.float64)
        # Fallback: first array-like entry
        for key in dat.files:
            arr = dat[key]
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 3:
                return np.asarray(arr[:, :3], dtype=np.float64)
        raise ValueError("Could not find an (N,3+) array inside the .npz")


def load_points(path: str) -> np.ndarray:
    p = path.lower()
    if p.endswith(".laz") or p.endswith(".las"):
        return _read_laz_or_las(path)
    elif p.endswith(".csv"):
        return _read_csv_xyz(path)
    elif p.endswith(".npy") or p.endswith(".npz"):
        return _read_npy_npz(path)
    else:
        raise ValueError(f"Unsupported extension: {os.path.splitext(path)[1]}")


def choose_grid(num_patches: int, x_range: float, y_range: float) -> Tuple[int, int]:
    """Select (nx, ny) factorization of num_patches to match XY aspect."""
    aspect = (x_range / y_range) if y_range != 0 else 1.0
    candidates = []
    for nx in range(1, num_patches + 1):
        if num_patches % nx == 0:
            ny = num_patches // nx
            candidates.append((nx, ny))
    # Consider both orientations
    candidates = list(set(candidates + [(ny, nx) for (nx, ny) in candidates]))
    best = min(candidates, key=lambda p: abs((p[0] / p[1]) - aspect))
    return best


def _aggregate_mean(ii, jj, x, y, z, S):
    """Mean aggregator via bincount for speed."""
    lin = ii * S + jj
    M = S * S

    counts = np.bincount(lin, minlength=M).astype(np.int64)
    sum_x = np.bincount(lin, weights=x, minlength=M)
    sum_y = np.bincount(lin, weights=y, minlength=M)
    sum_z = np.bincount(lin, weights=z, minlength=M)

    counts = counts.reshape(S, S)
    sum_x = sum_x.reshape(S, S)
    sum_y = sum_y.reshape(S, S)
    sum_z = sum_z.reshape(S, S)

    occupied = counts > 0
    xg = np.zeros_like(sum_x, dtype=np.float64)
    yg = np.zeros_like(sum_y, dtype=np.float64)
    zg = np.full_like(sum_z, np.nan, dtype=np.float64)

    xg[occupied] = sum_x[occupied] / counts[occupied]
    yg[occupied] = sum_y[occupied] / counts[occupied]
    zg[occupied] = sum_z[occupied] / counts[occupied]
    return xg, yg, zg, occupied


def _aggregate_median(ii, jj, x, y, z, S):
    """Median for Z, mean for X/Y. Slower but robust for Z."""
    lin = ii * S + jj
    M = S * S
    # Mean for x/y
    counts = np.bincount(lin, minlength=M).astype(np.int64)
    sum_x = np.bincount(lin, weights=x, minlength=M)
    sum_y = np.bincount(lin, weights=y, minlength=M)

    counts = counts.reshape(S, S)
    sum_x = sum_x.reshape(S, S)
    sum_y = sum_y.reshape(S, S)

    occupied = counts > 0
    xg = np.zeros_like(sum_x, dtype=np.float64)
    yg = np.zeros_like(sum_y, dtype=np.float64)
    xg[occupied] = sum_x[occupied] / counts[occupied]
    yg[occupied] = sum_y[occupied] / counts[occupied]

    # Median for z (group by lin)
    # Build lists of z per cell index (vectorized grouping is non-trivial; this is acceptable at 50x50)
    z_lists = [[] for _ in range(M)]
    for k, l in enumerate(lin):
        z_lists[l].append(float(z[k]))
    zg = np.full((S, S), np.nan, dtype=np.float64)
    for idx_lin, lst in enumerate(z_lists):
        if lst:
            i = idx_lin // S
            j = idx_lin % S
            zg[i, j] = float(np.median(np.array(lst, dtype=np.float64)))
    return xg, yg, zg, occupied


def bin_points_to_grid(x, y, z, x0, x1, y0, y1, grid_size, global_mean_z):
    """
    Aggregate points within the tile [x0,x1]x[y0,y1] onto a grid_size x grid_size grid.
    Returns (xg, yg, zg) each (S, S) where S=grid_size.
    - xg, yg are mean XY per occupied cell; empty cells get cell centers.
    - zg is mean/median Z; empty cells filled by nearest occupied cell or global_mean_z.
    """
    S = grid_size
    dx = (x1 - x0) / S if S > 0 else 0.0
    dy = (y1 - y0) / S if S > 0 else 0.0

    # Cell centers
    xc = x0 + (np.arange(S, dtype=np.float64) + 0.5) * dx
    yc = y0 + (np.arange(S, dtype=np.float64) + 0.5) * dy
    xg_centers, yg_centers = np.meshgrid(xc, yc, indexing='ij')

    if dx == 0 or dy == 0 or x.size == 0:
        # Degenerate or empty: flat fill at global mean Z
        zg = np.full((S, S), global_mean_z, dtype=np.float64)
        return xg_centers, yg_centers, zg

    # Map each point into [0, S-1] cell indices
    ii = np.floor((x - x0) / dx).astype(np.int64)
    jj = np.floor((y - y0) / dy).astype(np.int64)
    ii = np.clip(ii, 0, S - 1)
    jj = np.clip(jj, 0, S - 1)

    # Aggregate
    if AGGREGATION_Z.lower() == "median":
        xg, yg, zg, occupied = _aggregate_median(ii, jj, x, y, z, S)
    else:
        xg, yg, zg, occupied = _aggregate_mean(ii, jj, x, y, z, S)

    # Initialize outputs with centers; overwrite occupied
    xg_out = xg_centers.copy()
    yg_out = yg_centers.copy()
    xg_out[occupied] = xg[occupied]
    yg_out[occupied] = yg[occupied]

    # Fill empty Z
    empty = ~np.isfinite(zg)
    if np.any(empty):
        if EMPTY_Z_FILL.lower() == "nearest":
            occ_idx = np.argwhere(~empty)  # occupied cells
            if occ_idx.size == 0:
                zg[empty] = global_mean_z
            else:
                emp_idx = np.argwhere(empty)
                diffs = emp_idx[:, None, :] - occ_idx[None, :, :]
                d2 = (diffs[..., 0] * diffs[..., 0] + diffs[..., 1] * diffs[..., 1]).astype(np.float64)
                nearest = np.argmin(d2, axis=1)
                nearest_cells = occ_idx[nearest]
                zg[emp_idx[:, 0], emp_idx[:, 1]] = zg[nearest_cells[:, 0], nearest_cells[:, 1]]
        else:
            zg[empty] = global_mean_z

    return xg_out, yg_out, zg


def normalize_aabb(pts_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize (S,S,3) to [0,1] per axis. Returns (norm, min, max)."""
    p = pts_xyz.reshape(-1, 3)
    aabb_min = np.nanmin(p, axis=0)
    aabb_max = np.nanmax(p, axis=0)
    scale = np.where(aabb_max > aabb_min, (aabb_max - aabb_min), 1.0)
    norm = (pts_xyz - aabb_min) / scale
    return norm, aabb_min, aabb_max


import numpy as np
from typing import Tuple

def normalize_aabb_global(pts_xyz: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalize a patch (S,S,3) to the global [0,1] range based on all XYZ values.

    Parameters
    ----------
    pts_xyz : np.ndarray
        Shape (S,S,3), raw XYZ coordinates.

    Returns
    -------
    norm : np.ndarray
        Normalized array in [0,1] range (same shape as input).
    vmin : float
        Minimum value before normalization.
    vmax : float
        Maximum value before normalization.
    """
    p = pts_xyz.reshape(-1, 3)
    vmin = float(np.nanmin(p))
    vmax = float(np.nanmax(p))
    scale = (vmax - vmin) if vmax > vmin else 1.0
    norm = (pts_xyz - vmin) / scale
    return norm, vmin, vmax



def main():
    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Input file not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    # Load points
    xyz = load_points(INPUT_PATH)
    if xyz.shape[1] != 3:
        xyz = xyz[:, :3]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # Global stats
    xmin, ymin, zmin = float(x.min()), float(y.min()), float(z.min())
    xmax, ymax, zmax = float(x.max()), float(y.max()), float(z.max())
    x_range = xmax - xmin
    y_range = ymax - ymin
    global_mean_z = float(z.mean())

    # Decide patch grid (nx, ny)
    if GRID_MODE.lower() == "manual":
        nx, ny = int(NX_MANUAL), int(NY_MANUAL)
        if nx * ny != NUM_PATCHES:
            print(f"[WARN] NX_MANUAL*NY_MANUAL ({nx*ny}) != NUM_PATCHES ({NUM_PATCHES}). Using manual anyway.")
    else:
        nx, ny = choose_grid(NUM_PATCHES, x_range, y_range)

    # Tile edges
    x_edges = np.linspace(xmin, xmax, nx + 1, dtype=np.float64)
    y_edges = np.linspace(ymin, ymax, ny + 1, dtype=np.float64)

    S = int(GRID_SIZE)
    num_tiles = nx * ny
    patches_raw = np.empty((num_tiles, S, S, 3), dtype=np.float32)
    patches_norm = np.empty_like(patches_raw)
    uv_grids = np.empty((num_tiles, S, S, 2), dtype=np.float32)
    tile_bounds = np.empty((num_tiles, 2, 2), dtype=np.float64)
    patch_aabbs = np.empty((num_tiles, 2, 3), dtype=np.float64)
    points_per_tile = np.zeros((num_tiles,), dtype=np.int64)

    # UV grid in [0,1]
    u = (np.arange(S, dtype=np.float64) + 0.5) / S
    v = (np.arange(S, dtype=np.float64) + 0.5) / S
    Ug, Vg = np.meshgrid(u, v, indexing='ij')

    # Process tiles
    idx = 0
    for iy in range(ny):
        y0, y1 = y_edges[iy], y_edges[iy + 1]
        y_mask = (y >= y0) & ((y < y1) if iy < ny - 1 else (y <= y1))
        for ix in range(nx):
            x0, x1 = x_edges[ix], x_edges[ix + 1]
            x_mask = (x >= x0) & ((x < x1) if ix < nx - 1 else (x <= x1))
            mask = x_mask & y_mask
            pts = xyz[mask]
            points_per_tile[idx] = pts.shape[0]

            tile_bounds[idx, 0, :] = [x0, y0]
            tile_bounds[idx, 1, :] = [x1, y1]

            if pts.shape[0] == 0:
                dx = (x1 - x0) / S if S > 0 else 0.0
                dy = (y1 - y0) / S if S > 0 else 0.0
                xc = x0 + (np.arange(S, dtype=np.float64) + 0.5) * dx
                yc = y0 + (np.arange(S, dtype=np.float64) + 0.5) * dy
                xg, yg = np.meshgrid(xc, yc, indexing='ij')
                zg = np.full((S, S), global_mean_z, dtype=np.float64)
            else:
                xg, yg, zg = bin_points_to_grid(pts[:, 0], pts[:, 1], pts[:, 2],
                                                x0, x1, y0, y1, S, global_mean_z)

            patch = np.stack([xg, yg, zg], axis=-1).astype(np.float32)
            patches_raw[idx] = patch

            if PER_PATCH_AABB:
                patch_norm, aabb_min, aabb_max = normalize_aabb(patch)
            else:
                # Global AABB normalization
                aabb_min = np.array([xmin, ymin, zmin], dtype=np.float64)
                aabb_max = np.array([xmax, ymax, zmax], dtype=np.float64)
                scale = np.where(aabb_max > aabb_min, (aabb_max - aabb_min), 1.0)
                patch_norm = (patch - aabb_min) / scale

            patches_norm[idx] = patch_norm.astype(np.float32)
            patch_aabbs[idx, 0, :] = aabb_min
            patch_aabbs[idx, 1, :] = aabb_max

            uv_grids[idx, :, :, 0] = Ug.astype(np.float32)
            uv_grids[idx, :, :, 1] = Vg.astype(np.float32)

            idx += 1

    # Save
    if SAVE_RAW_PATCHES:
        np.savez(
            OUTPUT_RAW_NPZ,
            patches_xyz=patches_raw,
            uv_grids=uv_grids,
            tile_bounds=tile_bounds,
            grid_shape=np.array([S, S], dtype=np.int32),
            patch_grid=np.array([nx, ny], dtype=np.int32),
            points_per_tile=points_per_tile,
        )
        print(f"[OK] Wrote RAW patches: {OUTPUT_RAW_NPZ}")

    if SAVE_NORMALIZED_PATCHES:
        np.savez(
            OUTPUT_NORM_NPZ,
            patches_norm=patches_norm,
            uv_grids=uv_grids,
            tile_bounds=tile_bounds,
            aabbs=patch_aabbs,
            grid_shape=np.array([S, S], dtype=np.int32),
            patch_grid=np.array([nx, ny], dtype=np.int32),
            points_per_tile=points_per_tile,
        )
        print(f"[OK] Wrote normalized patches: {OUTPUT_NORM_NPZ}")

    # Summary
    print("[SUMMARY]")
    print(f"  Input:        {INPUT_PATH}")
    print(f"  Points:       {xyz.shape[0]:,}")
    print(f"  Bounds X:     [{xmin:.3f}, {xmax:.3f}]  (Δ={x_range:.3f})")
    print(f"  Bounds Y:     [{ymin:.3f}, {ymax:.3f}]  (Δ={y_range:.3f})")
    print(f"  Bounds Z:     [{zmin:.3f}, {zmax:.3f}]")
    print(f"  Patch grid:   {nx} x {ny}  (total {nx*ny})")
    print(f"  Patch size:   {GRID_SIZE} x {GRID_SIZE}  → (50,50,3) tensors")
    print(f"  Z agg:        {AGGREGATION_Z} ; empty fill: {EMPTY_Z_FILL}")
    print(f"  Normalize:    {'per-patch AABB' if PER_PATCH_AABB else 'global AABB'}")
    print(f"  Min/Max pts per tile: {points_per_tile.min()} / {points_per_tile.max()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(2)
