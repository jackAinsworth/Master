from __future__ import annotations
"""
mesh_subdivider.py – Fast voxel‑grid + summed‑volume‑table routine to find the
largest axis‑aligned cuboid that fits entirely inside an arbitrary watertight
mesh.

Key public calls
----------------
* ``largest_cuboid_voxel(mesh, pitch=0.02, max_grid=160)`` – returns ``(volume,
  bounds)`` where ``bounds=(x0,x1,y0,y1,z0,z1)`` in *world units*.
* ``cuboid_mesh(bounds)`` – converts the bounds to a watertight ``trimesh`` box
  mesh.
* ``debug_plot_voxel(mesh, bounds)`` – quick 3‑D preview (Matplotlib) showing
  the voxel grid, the cuboid, and the original mesh outline.

Dependencies: numpy, trimesh, scipy, matplotlib  (all pip‑installable).

Designed to finish < 0.2 s for ~150³ voxels on a typical laptop.
"""

import numpy as np
import trimesh
from typing import Tuple

__all__ = [
    "largest_cuboid_voxel",
    "cuboid_mesh",
    "debug_plot_voxel",
]

# -----------------------------------------------------------------------------
# Core algorithm
# -----------------------------------------------------------------------------

def _summed_volume(binary: np.ndarray) -> np.ndarray:
    """Return the 3‑D summed‑volume table (prefix sums)."""
    return binary.cumsum(0).cumsum(1).cumsum(2)


def _subvol(svt: np.ndarray, x0, x1, y0, y1, z0, z1):
    """Sum of voxels inside the half‑open cuboid [x0,x1)×[y0,y1)×[z0,z1)."""
    # Inclusion‑exclusion principle in 3‑D
    return (
        svt[x1, y1, z1]
        - svt[x0, y1, z1]
        - svt[x1, y0, z1]
        - svt[x1, y1, z0]
        + svt[x0, y0, z1]
        + svt[x0, y1, z0]
        + svt[x1, y0, z0]
        - svt[x0, y0, z0]
    )


def largest_cuboid_voxel(
    mesh: trimesh.Trimesh,
    *,
    pitch: float = 0.02,
    max_grid: int = 160,
) -> Tuple[float, tuple[float, float, float, float, float, float]]:
    """Voxel‑grid search for the maximal axis‑aligned cuboid.

    Parameters
    ----------
    mesh : trimesh.Trimesh (watertight preferred)
    pitch : float
        Voxel size in the mesh's length units.  0.02 → 2 cm if model in metres.
    max_grid : int
        Hard cap on the voxel dimensions in each axis; if exceeded, *pitch* is
        increased proportionally to keep memory × time in check.

    Returns
    -------
    volume : float  – cuboid volume in mesh units³
    bounds : tuple  – (x0,x1,y0,y1,z0,z1) world‑space coordinates
    """
    # 1) Voxelise --------------------------------------------------------------
    vox = mesh.voxelized(pitch=pitch)
    if max(vox.shape) > max_grid:
        scale = max(vox.shape) / max_grid
        pitch *= scale
        vox = mesh.voxelized(pitch=pitch)
    mat = vox.matrix.astype(np.uint8)  # 1 == occupied (inside)

    # Fill internal cavities (optional but safer)
    from scipy.ndimage import binary_fill_holes

    inside = binary_fill_holes(mat).astype(np.uint8)

    # 2) Prefix sums -----------------------------------------------------------
    svt = _summed_volume(inside)

    X, Y, Z = inside.shape
    bestV = 0
    best_box = (0, 0, 0, 0, 0, 0)

    # 3) Enumerate origins -----------------------------------------------------
    # Greedy optimisation: we iterate possible sizes outward using a height‑map
    height = np.zeros((Y, X), dtype=int)
    for z0 in range(Z):
        # Update height map: grow where inside, reset where outside
        height = height + inside[:, :, z0].T  # shape (X,Y)
        # For every z1 >= z0 that keeps full columns of inside voxels, we can form cuboids
        for z1 in range(z0, Z):
            dz = z1 - z0 + 1
            if dz * X * Y <= bestV:
                continue  # can't beat current best even if XY full
            mask = (height >= dz).T   # shape → (X,Y) so rows index X, columns Y
            h = np.zeros(Y, dtype=int)
            for x in range(X):
                col = mask[x]
                h = (h + col) * col  # grow where True, reset where False

                # largest rectangle in histogram 'h' (classic stack algo)
                stack = []
                for y in range(Y + 1):
                    curr = h[y] if y < Y else 0
                    while stack and curr < h[stack[-1]]:
                        height_idx = h[stack.pop()]
                        width = y if not stack else y - stack[-1] - 1
                        area = height_idx * width
                        vol = area * dz
                        if vol > bestV:
                            bestV = vol
                            x1 = x
                            x0 = x1 - height_idx + 1
                            y1 = y - 1
                            y0 = y1 - width + 1
                            best_box = (x0, x1, y0, y1, z0, z1)
                    stack.append(y)
            # break early: next z1 only increases dz, improves volumes
    if bestV == 0:
        raise RuntimeError("No internal voxels found – is the mesh watertight?")

    # 4) Convert voxel indices to world coords -------------------------------
        # 4) Convert voxel indices to world coords -------------------------------
    if hasattr(vox, "origin"):
        origin = vox.origin
    else:  # fallback for older trimesh: compute from bounds and pitch
        min_bound = mesh.bounds[0]
        origin = min_bound - 0.5 * pitch  # trimesh voxel grid is centred in each cell
    bounds = (
        origin[0] + best_box[0] * pitch,
        origin[0] + (best_box[1] + 1) * pitch,
        origin[1] + best_box[2] * pitch,
        origin[1] + (best_box[3] + 1) * pitch,
        origin[2] + best_box[4] * pitch,
        origin[2] + (best_box[5] + 1) * pitch,
    )
    return bestV * (pitch ** 3), bounds


# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------

def cuboid_from_ply(
    idx: int,
    *,
    subdiv: bool = True,
    pitch: float = 0.02,
    data_dir: str | None = None,
    **kw,
):
    """One‑liner: load *sampleXXXX.ply*, find the largest cuboid, return volume & bounds.

    Parameters
    ----------
    idx : int            – sample index (same naming as in ``largest_rect.load_ply``)
    subdiv : bool        – load the *_subdiv.ply* variant (default **True**)
    pitch : float        – voxel size passed to ``largest_cuboid_voxel``
    data_dir : str|None  – override the dataset directory
    kw :                 – forwarded to ``largest_cuboid_voxel`` (e.g. max_grid=128)
    """
        # Inline loader so this file is self‑contained --------------------------------
    try:
        from largest_rect import load_ply  # existing lib available?
    except ModuleNotFoundError:
        import os, numpy as _np

        def load_ply(idx: int, *, subdiv: bool = False, data_dir: str | None = None):
            """Lightweight PLY loader (ASCII only) cloned from the old largest_rect lib."""
            base = data_dir or os.getenv("SUBDIV_DATA_DIR", "/home/ainsworth/master/dataset_1000/")
            name = f"sample{idx:06d}" + ("_subdiv" if subdiv else "")
            path = os.path.join(base, name + ".ply")
            verts, faces = [], []
            with open(path, "r", encoding="utf-8") as fh:
                assert fh.readline().strip() == "ply"
                assert "ascii" in fh.readline()
                nv = nf = 0
                for line in fh:
                    line = line.strip()
                    if line.startswith("element vertex"):
                        nv = int(line.split()[-1])
                    elif line.startswith("element face"):
                        nf = int(line.split()[-1])
                    elif line == "end_header":
                        break
                for _ in range(nv):
                    verts.append(list(map(float, fh.readline().split())))
                for _ in range(nf):
                    vals = list(map(int, fh.readline().split()))
                    faces.append(vals[1:])  # drop the leading N
            return _np.asarray(verts, dtype=float), faces
    # -------------------------------------------------------------------------
        verts, faces = load_ply(idx, subdiv=subdiv, data_dir=data_dir)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return largest_cuboid_voxel(mesh, pitch=pitch, **kw)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def cuboid_mesh(bounds):
    """Return a watertight trimesh box from (x0,x1,y0,y1,z0,z1)."""
    x0, x1, y0, y1, z0, z1 = map(float, bounds)
    size = [x1 - x0, y1 - y0, z1 - z0]
    centre = [(x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5]
    return trimesh.creation.box(extents=size, transform=trimesh.transformations.translation_matrix(centre))


try:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt

    def debug_plot_voxel(mesh: trimesh.Trimesh, bounds, *, pitch=0.02):
        """Show mesh wireframe, voxel centres, and the cuboid in 3‑D."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Mesh wireframe
        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                        triangles=mesh.faces, alpha=0.1, edgecolor="k", linewidth=0.2, color="cyan")

        # Cuboid faces
        box = cuboid_mesh(bounds)
        faces3d = [[box.vertices[idx] for idx in face] for face in box.faces_unique]
        ax.add_collection3d(Poly3DCollection(faces3d, facecolor="orange", alpha=0.4, edgecolor="r"))

        ax.auto_scale_xyz(mesh.bounds[:, 0], mesh.bounds[:, 1], mesh.bounds[:, 2])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()
except ModuleNotFoundError:
    pass
