"""
surface_compare.py

A collection of functions for comparing the original surface with the predicted surface
generated from a predicted control net. Both surfaces are computed using a NURBS-based
approach (as used in the nurbs_vis library and data generator), so that the evaluation is
consistent.

Usage:
  Import this module and call the desired function, e.g.:
      import surface_compare
      surface_compare.compare_surfaces(original_entry, predicted_ctrl_net)
      surface_compare.compare_control_nets_in_one_plot(original_cn, predicted_cn)
      surface_compare.compare_approximated_control_nets(original_entry, predicted_ctrl_net)
"""

import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Ensures 3D plotting
from geomdl import BSpline, NURBS, utilities, fitting  # For creating and evaluating NURBS surfaces
import matplotlib.cm as cm




# ---------------- Schema helpers ----------------

from typing import Any, Optional, Sequence, Literal

Schema = Literal["rotated", "coons", "auto"]

# Logical → possible concrete keys in priority order
KEY_ALIASES = {
    "surface_points": ["rotated_noisy_points", "points"],
    "control_net":    ["rotated_control_net", "control_net"],
    # Optional per-entry knots (when present)
    "knotvector_u":   ["knotvector_u"],
    "knotvector_v":   ["knotvector_v"],
}

def pick(entry: dict, logical_key: str, *, candidates: Optional[Sequence[str]] = None) -> Any:
    """
    Return the first present value for `logical_key` using KEY_ALIASES
    (or a custom `candidates` list). Raises KeyError if none found.
    """
    keys = list(candidates) if candidates is not None else KEY_ALIASES.get(logical_key, [])
    for k in keys:
        if k in entry:
            return entry[k]
    # Fall back: if the logical key itself exists, allow it.
    if logical_key in entry:
        return entry[logical_key]
    raise KeyError(f"None of the expected keys for '{logical_key}' found: {keys or [logical_key]}")

def detect_schema(entry: dict) -> Literal["rotated", "coons"]:
    """Heuristic to classify the entry."""
    if "rotated_noisy_points" in entry or "rotated_control_net" in entry:
        return "rotated"
    if "points" in entry or "control_net" in entry:
        return "coons"
    # Default to rotated if ambiguous, but you can change this.
    return "rotated"

def aliases_for(logical_key: str, schema: Schema) -> Sequence[str]:
    """
    Return the concrete key list for a given logical key and schema.
    If schema='auto', we use both aliases in order.
    If schema is explicit, we return only that schema's key first.
    """
    both = KEY_ALIASES[logical_key]
    if schema == "auto":
        return both
    if logical_key == "surface_points":
        return ["rotated_noisy_points"] if schema == "rotated" else ["points"]
    if logical_key == "control_net":
        return ["rotated_control_net"] if schema == "rotated" else ["control_net"]
    # knots don't differ by schema in your data, but keep function generic
    return both








# -----------------------------------------------------------------------------
# Utility: Set axes equal (for 3D plots)
def set_axes_equal(ax):
    """
    Set 3D plot axes to equal scale so that objects appear with correct proportions.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# -----------------------------------------------------------------------------
# Functions to remove padding from a control net

def remove_padding_and_shrink_control_net(ctrl_net, threshold=1e-1, padding_value=-10, padding_tol=1e-3):
    """
    Remove padded control points by dropping entire rows or columns where all points
    are considered padded.

    A control point is considered padded if either:
      1. Its L2 norm is below the given threshold, or
      2. All its channel values are within a small tolerance of the specified padding_value.

    This function removes rows if every control point in that row is padded,
    and similarly for columns. This will change the shape of the control net.

    Args:
        ctrl_net (np.ndarray): Control net array of shape (u, v, 3) or (1, u, v, 3).
        threshold (float): The L2 norm threshold below which a control point is considered padded.
        padding_value (float): The value that indicates a padded control point (default: -10).
        padding_tol (float): Tolerance for comparing a control point to the padding_value.

    Returns:
        np.ndarray: A new control net array with padded rows and columns removed.
    """
    # Remove extra dimension if present.
    if ctrl_net.ndim == 4:
        ctrl_net = ctrl_net[0]

    # Compute the L2 norm for each control point.
    norms = np.linalg.norm(ctrl_net, axis=-1)  # shape: (u, v)

    # Create a mask for control points that are nearly equal to the padding value.
    padding_mask = np.all(np.abs(ctrl_net - padding_value) < padding_tol, axis=-1)

    # A control point is considered padded if it is either close to the padding value
    # or its norm is below the threshold.
    padded = padding_mask | (norms < threshold)

    # Identify rows and columns that have at least one non-padded control point.
    rows_to_keep = ~np.all(padded, axis=1)
    cols_to_keep = ~np.all(padded, axis=0)

    # Select and return only the non-padded rows and columns.
    new_cn = ctrl_net[rows_to_keep][:, cols_to_keep, :]
    return new_cn


def remove_padding_from_control_net(ctrl_net, threshold=1e-1):
    """
    Remove (or erase) padded control points from a predicted control net by replacing
    their values with np.nan.

    The function assumes that padded control points have coordinate values close to zero
    (within the given threshold) and replaces them with np.nan. This is useful when you
    want to ignore these points during plotting without changing the net's shape.

    Args:
        ctrl_net (np.ndarray): Control net array of shape (u, v, 3) or (1, u, v, 3).
        threshold (float): Threshold value for the L2 norm below which a control point is considered padding.

    Returns:
        np.ndarray: A new control net array with padded control points replaced by np.nan.
    """
    if ctrl_net.ndim == 4:
        ctrl_net = ctrl_net[0]
    new_cn = np.copy(ctrl_net)
    norms = np.linalg.norm(new_cn, axis=-1)
    padding_mask = norms < threshold
    new_cn[padding_mask] = np.nan
    return new_cn




import numpy as np

def remove_negative_and_shrink_control_net(ctrl_net):
    """
    Remove rows and columns from the control net where *all* control points
    have at least one negative coordinate.

    Args:
        ctrl_net (np.ndarray): Control net array of shape (u, v, 3) or (1, u, v, 3).

    Returns:
        np.ndarray: A new control net array with rows/columns below 0 removed.
    """
    # Remove extra dimension if present.
    if ctrl_net.ndim == 4:
        ctrl_net = ctrl_net[0]

    # A control point is considered "negative" if any of its coordinates < 0
    negative_mask = np.any(ctrl_net < 0, axis=-1)  # shape: (u, v)

    # Keep rows/cols if they have at least one non-negative control point
    rows_to_keep = ~np.all(negative_mask, axis=1)
    cols_to_keep = ~np.all(negative_mask, axis=0)

    # Select and return only the non-negative rows and columns
    new_cn = ctrl_net[rows_to_keep][:, cols_to_keep, :]
    return new_cn




def compare_control_net_padding(ctrl_net,
                                threshold=1e-1,
                                padding_value=-10.0,
                                padding_tol=1e-3,
                                title="Control Net: before vs after padding removal"):
    """
    Quick visual + summary of a control net before/after removing padding.

    - Left: original net; padded points highlighted.
    - Right: shrunken net (rows/cols where all points are padded are dropped).

    Args:
        ctrl_net (np.ndarray): (u, v, 3) or (1, u, v, 3)
        threshold (float): L2-norm below which a point is considered padded.
        padding_value (float): Explicit padding sentinel value (e.g., -10).
        padding_tol (float): Tolerance to treat values as padding_value.

    Returns:
        dict with:
            "clean_net": np.ndarray of shape (u', v', 3)
            "orig_shape": tuple
            "clean_shape": tuple
            "num_padded_points": int
    """
    # --- normalize shape ---
    orig = ctrl_net[0] if ctrl_net.ndim == 4 else ctrl_net
    u, v, _ = orig.shape

    # --- classify padded points on the original net ---
    norms = np.linalg.norm(orig, axis=-1)                     # (u, v)
    is_padding_val = np.all(np.abs(orig - padding_value) < padding_tol, axis=-1)
    is_padded = is_padding_val | (norms < threshold)
    num_padded = int(is_padded.sum())

    # --- build the cleaned net (drop padded-only rows/cols) ---
    clean = remove_padding_and_shrink_control_net(
        ctrl_net, threshold=threshold, padding_value=padding_value, padding_tol=padding_tol
    )

    # --- plotting ---
    fig = plt.figure(figsize=(14, 6))

    # Left: original with padded points marked
    ax1 = fig.add_subplot(121, projection='3d')
    # draw grid lines (original)
    for i in range(u):
        ax1.plot(orig[i, :, 0], orig[i, :, 1], orig[i, :, 2],
                 color='tab:blue', linestyle='--', marker='o', markersize=4, alpha=0.9)
    for j in range(v):
        ax1.plot(orig[:, j, 0], orig[:, j, 1], orig[:, j, 2],
                 color='tab:blue', linestyle='--', marker='o', markersize=4, alpha=0.9)
    # overlay padded points
    pad_idx = np.argwhere(is_padded)
    if pad_idx.size > 0:
        ax1.scatter(orig[is_padded, 0], orig[is_padded, 1], orig[is_padded, 2],
                    marker='x', s=40, linewidths=1.5, color='tab:red', label='padded')
        ax1.legend(loc='best')
    ax1.set_title(f"Original (u,v)=({u},{v})  |  padded pts: {num_padded}")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    set_axes_equal(ax1)

    # Right: cleaned/shrunken net
    cu, cv, _ = clean.shape
    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(cu):
        ax2.plot(clean[i, :, 0], clean[i, :, 1], clean[i, :, 2],
                 color='tab:green', linestyle='--', marker='^', markersize=4, alpha=0.95)
    for j in range(cv):
        ax2.plot(clean[:, j, 0], clean[:, j, 1], clean[:, j, 2],
                 color='tab:green', linestyle='--', marker='^', markersize=4, alpha=0.95)
    ax2.set_title(f"After removal (u,v)=({cu},{cv})")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    set_axes_equal(ax2)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

    return {
        "clean_net": clean,
        "orig_shape": (u, v, 3),
        "clean_shape": clean.shape,
        "num_padded_points": num_padded,
    }



def compare_control_net_negative_padding(ctrl_net,
                                         title="Control Net: before vs after negative-padding removal"):
    """
    Visual + summary using negative-based padding:
      - A control point is 'negative' if ANY coordinate < 0.
      - Rows/cols are dropped if ALL points in that row/col are negative.

    Left: original net (negative points highlighted).
    Right: shrunken net after remove_negative_and_shrink_control_net(...).

    Args:
        ctrl_net (np.ndarray): (u, v, 3) or (1, u, v, 3)

    Returns:
        dict with:
            "clean_net": np.ndarray of shape (u', v', 3)
            "orig_shape": tuple
            "clean_shape": tuple
            "num_negative_points": int
    """
    # --- normalize shape ---
    orig = ctrl_net[0] if ctrl_net.ndim == 4 else ctrl_net
    u, v, _ = orig.shape

    # --- classify negatives (ANY coord < 0) ---
    is_negative = np.any(orig < 0, axis=-1)  # (u, v)
    num_negative = int(is_negative.sum())

    # --- shrink using your helper ---
    clean = remove_negative_and_shrink_control_net(ctrl_net)
    cu, cv, _ = clean.shape

    # --- plotting ---
    fig = plt.figure(figsize=(14, 6))

    # Left: original with negative points marked
    ax1 = fig.add_subplot(121, projection='3d')
    # grid lines
    for i in range(u):
        ax1.plot(orig[i, :, 0], orig[i, :, 1], orig[i, :, 2],
                 color='tab:blue', linestyle='--', marker='o', markersize=4, alpha=0.9)
    for j in range(v):
        ax1.plot(orig[:, j, 0], orig[:, j, 1], orig[:, j, 2],
                 color='tab:blue', linestyle='--', marker='o', markersize=4, alpha=0.9)
    # overlay negatives
    if num_negative > 0:
        ax1.scatter(orig[is_negative, 0], orig[is_negative, 1], orig[is_negative, 2],
                    marker='x', s=40, linewidths=1.5, color='tab:red', label='negative (<0)')
        ax1.legend(loc='best')
    ax1.set_title(f"Original (u,v)=({u},{v})  |  negative pts: {num_negative}")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    set_axes_equal(ax1)

    # Right: cleaned/shrunken net
    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(cu):
        ax2.plot(clean[i, :, 0], clean[i, :, 1], clean[i, :, 2],
                 color='tab:green', linestyle='--', marker='^', markersize=4, alpha=0.95)
    for j in range(cv):
        ax2.plot(clean[:, j, 0], clean[:, j, 1], clean[:, j, 2],
                 color='tab:green', linestyle='--', marker='^', markersize=4, alpha=0.95)
    ax2.set_title(f"After removal (u,v)=({cu},{cv})")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    set_axes_equal(ax2)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

    return {
        "clean_net": clean,
        "orig_shape": (u, v, 3),
        "clean_shape": clean.shape,
        "num_negative_points": num_negative,
    }




# -----------------------------------------------------------------------------
# NURBS surface reconstruction functions

    # ------------------------------------------------------------------
    # helper: approximate a control-net from a noisy surface the *right* way
    # ------------------------------------------------------------------
def approximate_control_net(surface, degree_u, degree_v, ctrlpts_size_u, ctrlpts_size_v, centri = True):
        """Fit a NURBS surface directly to the full `surface` data; return the fitted surface and its control net."""
        # surface: ndarray of shape (num_u, num_v, dim)
        num_u, num_v, dim = surface.shape

        # Flatten the full grid of data points
        flat_points = surface.reshape(-1, dim).tolist()

        # Approximate surface, specifying desired control-net size
        nurbs_surf_approx = fitting.approximate_surface(
            flat_points,
            size_u=num_u,
            size_v=num_v,
            degree_u=degree_u,
            degree_v=degree_v,
            ctrlpts_size_u=ctrlpts_size_u,
            ctrlpts_size_v=ctrlpts_size_v,
            centripetal=centri
        )

        # Extract actual control-net dimensions
        try:
            size_u_a, size_v_a = nurbs_surf_approx.ctrlpts_size
        except (TypeError, AttributeError):
            size_u_a = nurbs_surf_approx.ctrlpts_size_u
            size_v_a = nurbs_surf_approx.ctrlpts_size_v

        # Reshape control points into a net array
        approx_cn = np.array(nurbs_surf_approx.ctrlpts).reshape((size_u_a, size_v_a, dim))

        return nurbs_surf_approx, approx_cn

def rebuild_nurbs_surface_from_control_net(
        ctrl_net,
        degree_u=3,
        degree_v=3,
        knotvector_u=None,
        knotvector_v=None
    ):
    """
    Rebuild a NURBS surface from a given control net using either
    provided knot vectors or uniform (clamped) ones.

    Args:
        ctrl_net (np.ndarray): Control net array of shape (u, v, 3) or (1, u, v, 3).
        degree_u (int): Degree in the u-direction.
        degree_v (int): Degree in the v-direction.
        knotvector_u (list, optional): Precomputed knot vector for u. If None, a uniform
                                       clamped knot vector is generated.
        knotvector_v (list, optional): Precomputed knot vector for v. If None, a uniform
                                       clamped knot vector is generated.

    Returns:
        BSpline.Surface: A NURBS surface object with control points from ctrl_net.
    """
    if ctrl_net.ndim == 4:
        ctrl_net = ctrl_net[0]
    u_count, v_count, _ = ctrl_net.shape

    # Create the surface
    surf = BSpline.Surface()
    surf.degree_u = degree_u
    surf.degree_v = degree_v
    surf.ctrlpts_size_u = u_count
    surf.ctrlpts_size_v = v_count
    surf.ctrlpts = ctrl_net.reshape(-1, 3).tolist()

    # U knot vector
    if knotvector_u is None:
        # uniform clamped
        tmp_u = np.linspace(0, 1, u_count - degree_u + 1)
        kv_u = [0.0]*(degree_u+1) + list(tmp_u[1:-1]) + [1.0]*(degree_u+1)
    else:
        kv_u = knotvector_u

    # V knot vector
    if knotvector_v is None:
        tmp_v = np.linspace(0, 1, v_count - degree_v + 1)
        kv_v = [0.0]*(degree_v+1) + list(tmp_v[1:-1]) + [1.0]*(degree_v+1)
    else:
        kv_v = knotvector_v

    surf.knotvector_u = kv_u
    surf.knotvector_v = kv_v

    return surf



def sample_nurbs_surface_with_control_net(surf, num_samples_u=50, num_samples_v=50):
    """
    Uniformly sample a NURBS surface.

    Args:
        surf (BSpline.Surface): A NURBS surface object.
        num_samples_u (int): Number of samples along the u-direction.
        num_samples_v (int): Number of samples along the v-direction.

    Returns:
        tuple: (pts, control_net) where:
            pts is an array of shape (num_samples_u, num_samples_v, 3) of sampled surface points,
            and control_net is an array of shape (size_u, size_v, 3) extracted from surf.ctrlpts.
    """
    pts = np.empty((num_samples_u, num_samples_v, 3))
    u_start = surf.knotvector_u[surf.degree_u]
    u_end = surf.knotvector_u[-(surf.degree_u+1)]
    v_start = surf.knotvector_v[surf.degree_v]
    v_end = surf.knotvector_v[-(surf.degree_v+1)]
    u_vals = np.linspace(u_start, u_end, num_samples_u)
    v_vals = np.linspace(v_start, v_end, num_samples_v)
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            pts[i, j, :] = np.array(surf.evaluate_single([u, v]))
    try:
        size_u, size_v = surf.ctrlpts_size
    except TypeError:
        size_u = surf.ctrlpts_size_u
        size_v = surf.ctrlpts_size_v
    control_net = np.array(surf.ctrlpts).reshape((size_u, size_v, 3))
    return pts, control_net


def reconstruct_surface_from_ctrlnet(ctrl_net, num_samples_u=50, num_samples_v=50, degree_u=3, degree_v=3,
        knotvector_u=None,
        knotvector_v=None):
    """
    Reconstruct a surface by first rebuilding a NURBS surface from the control net
    and then sampling it uniformly.

    Args:
        ctrl_net (np.ndarray): Control net array of shape (u, v, 3) or (1, u, v, 3).
        num_samples_u (int): Number of samples along the u-direction.
        num_samples_v (int): Number of samples along the v-direction.
        degree_u (int): Degree in the u-direction for the NURBS surface.
        degree_v (int): Degree in the v-direction for the NURBS surface.

    Returns:
        np.ndarray: Reconstructed surface points of shape (num_samples_u, num_samples_v, 3).
    """


    surf = rebuild_nurbs_surface_from_control_net(ctrl_net, degree_u=degree_u, degree_v=degree_v, knotvector_u=knotvector_u, knotvector_v=knotvector_v)
    pts, _ = sample_nurbs_surface_with_control_net(surf, num_samples_u, num_samples_v)
    return pts


# -----------------------------------------------------------------------------
# Comparison functions

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree


def chamfer_distance(points_src, points_tgt):
    """
    Compute the symmetric Chamfer Distance between two point clouds.
    """
    # Build KD-trees
    tree_src = cKDTree(points_src)
    tree_tgt = cKDTree(points_tgt)

    # For each point in src, find closest in tgt
    d_src, _ = tree_src.query(points_tgt)
    d_tgt, _ = tree_tgt.query(points_src)

    # Chamfer = average nearest neighbor squared distances both ways
    return np.mean(d_src ** 2) + np.mean(d_tgt ** 2)


def compare_surfaces(
        original_entry,
        predicted_ctrl_net,
        num_samples=35,
        show_control_net=True,
        degree_u=3,
        degree_v=3
):
    """
    Compare original vs predicted surfaces and control nets, coloring the
    predicted surface by the 3D Euclidean distance to the original.
    """
    # --- Reconstruct surfaces ---
    pred_surf = reconstruct_surface_from_ctrlnet(
        predicted_ctrl_net, num_samples, num_samples, degree_u, degree_v
    )
    orig_surf = np.array(original_entry["rotated_noisy_points"])
    if orig_surf.shape != pred_surf.shape:
        pred_surf = reconstruct_surface_from_ctrlnet(
            predicted_ctrl_net,
            orig_surf.shape[0], orig_surf.shape[1],
            degree_u, degree_v
        )

    # --- Compute per‐point Euclidean error ---
    # error_map[i,j] = || orig_surf[i,j,:] - pred_surf[i,j,:] ||
    error_map = np.linalg.norm(orig_surf - pred_surf, axis=2)

    # --- Overall metrics ---
    mse_surface = np.mean((orig_surf - pred_surf)**2)
    pts_orig = orig_surf.reshape(-1,3)
    pts_pred = pred_surf.reshape(-1,3)
    chamfer_surf = chamfer_distance(pts_orig, pts_pred)

    # --- Control net metrics (unchanged) ---
    orig_cn = np.array(original_entry["rotated_control_net"])
    if predicted_ctrl_net.ndim == 4:
        pred_cn = predicted_ctrl_net[0]
    else:
        pred_cn = predicted_ctrl_net
    pred_cn_clean = remove_padding_and_shrink_control_net(pred_cn, threshold=1e-1)

    min_u = min(orig_cn.shape[0], pred_cn_clean.shape[0])
    min_v = min(orig_cn.shape[1], pred_cn_clean.shape[1])
    o_cn = orig_cn[:min_u, :min_v, :].reshape(-1,3)
    p_cn = pred_cn_clean[:min_u, :min_v, :].reshape(-1,3)
    mse_cn = np.mean((o_cn - p_cn)**2)
    chamfer_cn = chamfer_distance(o_cn, p_cn)

    # --- Plotting ---
    fig = plt.figure(figsize=(16,8))

    # Original surface in solid gray
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(
        orig_surf[:,:,0], orig_surf[:,:,1], orig_surf[:,:,2],
        color='purple', edgecolor='none', alpha=0.9
    )
    ax1.set_title("Original Surface")
    if show_control_net:
        u_o, v_o, _ = orig_cn.shape
        for i in range(u_o):
            ax1.plot(orig_cn[i,:,0], orig_cn[i,:,1], orig_cn[i,:,2],
                     'r--o', markersize=4)
        for j in range(v_o):
            ax1.plot(orig_cn[:,j,0], orig_cn[:,j,1], orig_cn[:,j,2],
                     'r--o', markersize=4)

    # Predicted surface colored by Euclidean distance
    fig2 = plt.figure(figsize=(18, 8))
    ax2 = fig2.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(
        pred_surf[:,:,0], pred_surf[:,:,1], pred_surf[:,:,2],
        facecolors=cm.inferno(error_map / error_map.max()),
        rcount=num_samples, ccount=num_samples,
        linewidth=0, antialiased=False
    )
    ax2.set_title(
        f"Predicted Surface\nMSE={mse_surface:.4f}, Chamfer={chamfer_surf:.4f}"
    )
    m = cm.ScalarMappable(cmap=cm.inferno)
    m.set_array(error_map)
    fig2.colorbar(m, ax=ax2, shrink=0.5, label="3D Euclidean error")

    if show_control_net:
        u_p, v_p, _ = pred_cn_clean.shape
        for i in range(u_p):
            ax2.plot(pred_cn_clean[i,:,0], pred_cn_clean[i,:,1], pred_cn_clean[i,:,2],
                     'r--o', markersize=4)
        for j in range(v_p):
            ax2.plot(pred_cn_clean[:,j,0], pred_cn_clean[:,j,1], pred_cn_clean[:,j,2],
                     'r--o', markersize=4)

    for ax in (ax1, ax2):
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        set_axes_equal(ax)

    #plt.tight_layout()
    plt.show()

    return {
        "mse_surface": mse_surface,
        "chamfer_surface": chamfer_surf,
        "mse_ctrl_net": mse_cn,
        "chamfer_ctrl_net": chamfer_cn
    }


def compare_control_nets_in_one_plot(original_cn, predicted_cn):
    """
    Compare the original and predicted control nets in a single 3D plot.

    Args:
        original_cn (np.ndarray): Original control net of shape (u, v, 3) or (1, u, v, 3).
        predicted_cn (np.ndarray): Predicted control net of shape (u, v, 3) or (1, u, v, 3).

    This function plots both control nets for direct visual comparison.
    """
    if original_cn.ndim == 4:
        original_cn = original_cn[0]
    if predicted_cn.ndim == 4:
        predicted_cn = predicted_cn[0]
    predicted_cn = remove_padding_and_shrink_control_net(predicted_cn, threshold=1e-1)

    if original_cn.shape != predicted_cn.shape:
        print("Warning: The two control nets have different shapes.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    u_o, v_o, _ = original_cn.shape
    for i in range(u_o):
        ax.plot(original_cn[i, :, 0],
                original_cn[i, :, 1],
                original_cn[i, :, 2],
                color='blue', linestyle='--', marker='o', label='Original CN' if i == 0 else "")
    for j in range(v_o):
        ax.plot(original_cn[:, j, 0],
                original_cn[:, j, 1],
                original_cn[:, j, 2],
                color='blue', linestyle='--', marker='o')

    u_p, v_p, _ = predicted_cn.shape
    for i in range(u_p):
        ax.plot(predicted_cn[i, :, 0],
                predicted_cn[i, :, 1],
                predicted_cn[i, :, 2],
                color='red', linestyle='--', marker='^', label='Predicted CN' if i == 0 else "")
    for j in range(v_p):
        ax.plot(predicted_cn[:, j, 0],
                predicted_cn[:, j, 1],
                predicted_cn[:, j, 2],
                color='red', linestyle='--', marker='^')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Original vs. Predicted Control Nets")
    ax.legend()
    set_axes_equal(ax)
    plt.show()


import surface_approx_start

def compare_approximated_control_nets(
    original_entry,
    predicted_ctrl_net,
    degree_u=3,
    degree_v=3,
    ctrlpts_size_u=None,
    ctrlpts_size_v=None
):
    """
    Compare:
      • control net approximated directly from the noisy surface   (BLUE)
      • predicted control net (after cleaning)                     (RED)
      • original ground-truth control net                          (GREEN)
    """

    # ----------------------- tidy inputs ------------------------
    if predicted_ctrl_net.ndim == 4:          # strip batch dim
        predicted_ctrl_net = predicted_ctrl_net[0]

    orig_cn = original_entry["rotated_control_net"]
    if orig_cn.ndim == 4:
        orig_cn = orig_cn[0]

    # derive sizes if not given
    if ctrlpts_size_u is None or ctrlpts_size_v is None:
        ctrlpts_size_u, ctrlpts_size_v, _ = orig_cn.shape

    # ------------------ build helper surfaces ------------------
    surface_points = original_entry["rotated_noisy_points"]

    # 1. approximated FROM surface (baseline)
    approx_surf, approx_cn = approximate_control_net(
        surface_points, degree_u, degree_v,
        ctrlpts_size_u, ctrlpts_size_v
    )

    # 2. clean predicted
    predicted_cn_clean = remove_padding_and_shrink_control_net(
        predicted_ctrl_net, threshold=1e-1
    )

    flat_points = surface_points.reshape(-1, 3).tolist()
    flat_points_cts = predicted_cn_clean.reshape(-1, 3).tolist()




    # --------------------- diagnostics --------------------------
    print("Shapes:")
    print("  Approximated CN :", approx_cn.shape)
    print("  Predicted CN    :", predicted_cn_clean.shape)
    print("  Original CN     :", orig_cn.shape)

    # choose common sub-grid for metric calculation
    mu = min(orig_cn.shape[0], predicted_cn_clean.shape[0],
             approx_cn.shape[0])
    mv = min(orig_cn.shape[1], predicted_cn_clean.shape[1],
             approx_cn.shape[1])

    def crop(a): return a[:mu, :mv].reshape(-1, 3)

    orig_crop   = crop(orig_cn)
    pred_crop   = crop(predicted_cn_clean)
    approx_crop = crop(approx_cn)

    # MSE
    mse_pred   = np.mean((pred_crop   - orig_crop)**2)
    mse_approx = np.mean((approx_crop - orig_crop)**2)

    # Chamfer
    cd_pred   = chamfer_distance(pred_crop,   orig_crop)
    cd_approx = chamfer_distance(approx_crop, orig_crop)

    print(f"MSE  (Predicted  vs Original): {mse_pred:.6f}")
    print(f"MSE  (Approximated vs Original): {mse_approx:.6f}")
    print(f"Chamfer (Predicted  vs Original): {cd_pred:.6f}")
    print(f"Chamfer (Approximated vs Original): {cd_approx:.6f}")

    # ------------------------- plots ----------------------------
    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection='3d')

    def plot_net(net, colour, marker, label):
        u, v, _ = net.shape
        for i in range(u):
            ax.plot(net[i, :, 0], net[i, :, 1], net[i, :, 2],
                    color=colour, linestyle='--', marker=marker,
                    label=label if i == 0 else "")
        for j in range(v):
            ax.plot(net[:, j, 0], net[:, j, 1], net[:, j, 2],
                    color=colour, linestyle='--', marker=marker)

    plot_net(approx_cn,   'blue',   'o', 'Approximated CN')
    plot_net(predicted_cn_clean, 'red',    '^', 'Predicted CN')
    plot_net(orig_cn,     'green',  's', 'Original CN')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()




def compare_approximated_control_nets_new(
    original_entry,
    predicted_ctrl_net,
    degree_u=3,
    degree_v=3,
    ctrlpts_size_u=None,
    ctrlpts_size_v=None,
    num_samples=None,           # if None, will match the original surface grid
    show_control_nets=True,      # toggle wireframes on surface plots
    show_surface=False,
    centri=True
):
    """
    Compare:
      • control net approximated directly from the noisy surface   (BLUE)
      • predicted control net (after cleaning)                     (RED)
      • original ground-truth control net                          (GREEN)

    Also compares reconstructed surfaces:
      • Predicted surface vs Original surface
      • Approximated surface vs Original surface

    Returns
    -------
    dict with CN- and surface-level metrics.
    """

    # ----------------------- tidy inputs ------------------------
    if predicted_ctrl_net.ndim == 4:          # strip batch dim
        predicted_ctrl_net = predicted_ctrl_net[0]

    orig_cn = original_entry["rotated_control_net"]
    if orig_cn.ndim == 4:
        orig_cn = orig_cn[0]

    # derive sizes if not given
    if ctrlpts_size_u is None or ctrlpts_size_v is None:
        ctrlpts_size_u, ctrlpts_size_v, _ = orig_cn.shape

    # Original surface (used as reference)
    orig_surf = np.asarray(original_entry["rotated_noisy_points"])
    if num_samples is None:
        su, sv = orig_surf.shape[:2]
    else:
        su = sv = int(num_samples)

    # ------------------ build helper surfaces ------------------
    surface_points = original_entry["rotated_noisy_points"]

    # 1) Approximate FROM surface (baseline)
    approx_surf_obj, approx_cn = approximate_control_net(
        surface_points, degree_u, degree_v,
        ctrlpts_size_u, ctrlpts_size_v,
        centri=centri
    )


    # 2) Clean predicted
    predicted_cn_clean = remove_padding_and_shrink_control_net(
        predicted_ctrl_net, threshold=1e-1
    )

    if show_surface:
        compare_surfaces(original_entry, predicted_ctrl_net, num_samples=50, show_control_net=True)
        compare_surfaces(original_entry, approx_cn, num_samples=50, show_control_net=True)


    # --------------------- CN diagnostics ----------------------
    print("Shapes:")
    print("  Approximated CN :", approx_cn.shape)
    print("  Predicted CN    :", predicted_cn_clean.shape)
    print("  Original CN     :", orig_cn.shape)

    # choose common sub-grid for CN metric calculation
    mu = min(orig_cn.shape[0], predicted_cn_clean.shape[0], approx_cn.shape[0])
    mv = min(orig_cn.shape[1], predicted_cn_clean.shape[1], approx_cn.shape[1])

    def crop(a): return a[:mu, :mv].reshape(-1, 3)

    orig_crop   = crop(orig_cn)
    pred_crop   = crop(predicted_cn_clean)
    approx_crop = crop(approx_cn)

    # CN MSE
    mse_pred_cn   = float(np.mean((pred_crop   - orig_crop)**2))
    mse_approx_cn = float(np.mean((approx_crop - orig_crop)**2))

    # CN Chamfer
    cd_pred_cn   = float(chamfer_distance(pred_crop,   orig_crop))
    cd_approx_cn = float(chamfer_distance(approx_crop, orig_crop))

    print(f"MSE  (Predicted  CN vs Original): {mse_pred_cn:.6f}")
    print(f"MSE  (Approximated CN vs Original): {mse_approx_cn:.6f}")
    print(f"Chamfer (Predicted  CN vs Original): {cd_pred_cn:.6f}")
    print(f"Chamfer (Approximated CN vs Original): {cd_approx_cn:.6f}")

    # -------------------- Surface recon & metrics --------------------
    # Predicted surface from predicted CN
    pred_surf = reconstruct_surface_from_ctrlnet(
        predicted_cn_clean, su, sv, degree_u, degree_v
    )

    # Approximated surface from approx CN (reuse fitted knot vectors for alignment)
    approx_surf = reconstruct_surface_from_ctrlnet(
        approx_cn, su, sv, degree_u, degree_v,
        knotvector_u=getattr(approx_surf_obj, "knotvector_u", None),
        knotvector_v=getattr(approx_surf_obj, "knotvector_v", None),
    )

    # Ensure orig_surf matches sampling resolution
    if orig_surf.shape[:2] != pred_surf.shape[:2]:
        # resample predicted to original resolution (shouldn't happen when num_samples=None)
        pred_surf = reconstruct_surface_from_ctrlnet(
            predicted_cn_clean, orig_surf.shape[0], orig_surf.shape[1], degree_u, degree_v
        )
        approx_surf = reconstruct_surface_from_ctrlnet(
            approx_cn, orig_surf.shape[0], orig_surf.shape[1], degree_u, degree_v,
            knotvector_u=getattr(approx_surf_obj, "knotvector_u", None),
            knotvector_v=getattr(approx_surf_obj, "knotvector_v", None),
        )

    # Error maps (per-point L2)
    err_pred   = np.linalg.norm(orig_surf - pred_surf,   axis=2)
    err_approx = np.linalg.norm(orig_surf - approx_surf, axis=2)

    # Surface-level metrics
    mse_pred_surf   = float(np.mean((orig_surf - pred_surf)   ** 2))
    mse_approx_surf = float(np.mean((orig_surf - approx_surf) ** 2))

    pts_o = orig_surf.reshape(-1, 3)
    pts_p = pred_surf.reshape(-1, 3)
    pts_a = approx_surf.reshape(-1, 3)

    cd_pred_surf   = float(chamfer_distance(pts_p, pts_o))
    cd_approx_surf = float(chamfer_distance(pts_a, pts_o))

    print(f"MSE  (Predicted  Surface vs Original): {mse_pred_surf:.6f}")
    print(f"MSE  (Approximated Surface vs Original): {mse_approx_surf:.6f}")
    print(f"Chamfer (Predicted  Surface vs Original): {cd_pred_surf:.6f}")
    print(f"Chamfer (Approximated Surface vs Original): {cd_approx_surf:.6f}")

    # ------------------------- plots ----------------------------
    # (A) Control nets in one plot (as before)
    fig_cn = plt.figure(figsize=(12, 9))
    ax_cn  = fig_cn.add_subplot(111, projection='3d')

    def plot_net(net, colour, marker, label):
        u, v, _ = net.shape
        for i in range(u):
            ax_cn.plot(net[i, :, 0], net[i, :, 1], net[i, :, 2],
                       color=colour, linestyle='--', marker=marker,
                       label=label if i == 0 else "")
        for j in range(v):
            ax_cn.plot(net[:, j, 0], net[:, j, 1], net[:, j, 2],
                       color=colour, linestyle='--', marker=marker)

    plot_net(approx_cn,          'blue',  'o', 'Approximated CN')
    plot_net(predicted_cn_clean, 'red',   '^', 'Predicted CN')
    plot_net(orig_cn,            'green', 's', 'Original CN')

    ax_cn.set_xlabel('X'); ax_cn.set_ylabel('Y'); ax_cn.set_zlabel('Z')
    ax_cn.set_title("Original vs Predicted vs Approximated Control Nets")
    ax_cn.legend()
    set_axes_equal(ax_cn)
    plt.tight_layout()
    plt.show()

    # (B) Surfaces with error colormaps (two separate subplots)
    eps1 = max(err_pred.max(), 1e-12)
    eps2 = max(err_approx.max(), 1e-12)

    fig_sf = plt.figure(figsize=(18, 8))

    # Left: Predicted surface error
    ax1 = fig_sf.add_subplot(121, projection='3d')
    ax1.plot_surface(
        pred_surf[:, :, 0], pred_surf[:, :, 1], pred_surf[:, :, 2],
        facecolors=cm.inferno(err_pred / eps1),
        rcount=pred_surf.shape[0], ccount=pred_surf.shape[1],
        linewidth=0, antialiased=False
    )
    ax1.set_title(f"Predicted Surface\nMSE={mse_pred_surf:.4f}, Chamfer={cd_pred_surf:.4f}")
    if show_control_nets:
        u_p, v_p, _ = predicted_cn_clean.shape
        for i in range(u_p):
            ax1.plot(predicted_cn_clean[i, :, 0], predicted_cn_clean[i, :, 1], predicted_cn_clean[i, :, 2],
                     'r--o', markersize=3, alpha=0.9)
        for j in range(v_p):
            ax1.plot(predicted_cn_clean[:, j, 0], predicted_cn_clean[:, j, 1], predicted_cn_clean[:, j, 2],
                     'r--o', markersize=3, alpha=0.9)
    m1 = cm.ScalarMappable(cmap=cm.inferno); m1.set_array(err_pred)
    fig_sf.colorbar(m1, ax=ax1, shrink=0.6, label="L2 error to Original")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    set_axes_equal(ax1)

    # Right: Approximated surface error
    ax2 = fig_sf.add_subplot(122, projection='3d')
    ax2.plot_surface(
        approx_surf[:, :, 0], approx_surf[:, :, 1], approx_surf[:, :, 2],
        facecolors=cm.inferno(err_approx / eps2),
        rcount=approx_surf.shape[0], ccount=approx_surf.shape[1],
        linewidth=0, antialiased=False
    )
    ax2.set_title(f"Approximated Surface\nMSE={mse_approx_surf:.4f}, Chamfer={cd_approx_surf:.4f}")
    if show_control_nets:
        u_a, v_a, _ = approx_cn.shape
        for i in range(u_a):
            ax2.plot(approx_cn[i, :, 0], approx_cn[i, :, 1], approx_cn[i, :, 2],
                     'b--o', markersize=3, alpha=0.9)
        for j in range(v_a):
            ax2.plot(approx_cn[:, j, 0], approx_cn[:, j, 1], approx_cn[:, j, 2],
                     'b--o', markersize=3, alpha=0.9)
    m2 = cm.ScalarMappable(cmap=cm.inferno); m2.set_array(err_approx)
    fig_sf.colorbar(m2, ax=ax2, shrink=0.6, label="L2 error to Original")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    set_axes_equal(ax2)

    plt.tight_layout()
    plt.show()

    # ------------------------- return ---------------------------
    return {
        # Control-net metrics
        "mse_ctrl_net_predicted": mse_pred_cn,
        "mse_ctrl_net_approximated": mse_approx_cn,
        "chamfer_ctrl_net_predicted": cd_pred_cn,
        "chamfer_ctrl_net_approximated": cd_approx_cn,

        # Surface metrics
        "mse_surface_predicted": mse_pred_surf,
        "mse_surface_approximated": mse_approx_surf,
        "chamfer_surface_predicted": cd_pred_surf,
        "chamfer_surface_approximated": cd_approx_surf,
    }









def prepare_input_for_model(
    dataset_entry: dict,
    target_shape=(100, 100, 3),
    *,
    schema: Schema = "auto"
):
    """
    Build a (1, H, W, 3) input tensor from either:
      - rotated schema: entry['rotated_noisy_points']
      - coons   schema: entry['points']
    """
    if schema == "auto":
        schema = detect_schema(dataset_entry)

    pts = np.array(
        pick(dataset_entry, "surface_points", candidates=aliases_for("surface_points", schema)),
        dtype=np.float32
    )

    if pts.shape != target_shape:
        raise ValueError(f"Input data shape {pts.shape} does not match expected {target_shape}.")

    return np.expand_dims(pts, axis=0)


def evaluate_dataset_control_nets(
    entries,
    model,
    target_shape=(35, 35, 3),
    threshold=10,       # kept for signature compatibility; not used with negative-based cleaning
    padding_value=-10,  # kept for signature compatibility
    padding_tol=20,     # kept for signature compatibility
    degree_u=3,
    degree_v=3,
    centri=True,
    *,
    schema: Schema = "auto"  # <— NEW
):
    """
    Same behavior as before, but works with either schema:
      rotated: uses 'rotated_noisy_points' / 'rotated_control_net'
      coons  : uses 'points' / 'control_net'
    """
    import numpy as np
    from scipy.spatial import cKDTree

    mse_cn_pred_crop, cd_cn_pred_full = [], []
    mse_cn_app_crop,  cd_cn_app_full  = [], []

    mse_surf_pred_true, cd_surf_pred_true = [], []
    mse_surf_app_true,  cd_surf_app_true  = [], []

    shape_mismatch = 0

    for e in entries:
        # Decide schema per-entry (lets you mix entries in one call)
        s = detect_schema(e) if schema == "auto" else schema

        # --- originals ---
        orig_surf_obs = np.asarray(
            pick(e, "surface_points", candidates=aliases_for("surface_points", s)),
            dtype=float
        )
        print(orig_surf_obs.shape)
        n_u, n_v, _ = orig_surf_obs.shape

        orig_cn = np.asarray(
            pick(e, "control_net", candidates=aliases_for("control_net", s)),
            dtype=float
        )
        if orig_cn.ndim == 4:
            orig_cn = orig_cn[0]

        kv_u_true = e.get("knotvector_u", None)
        kv_v_true = e.get("knotvector_v", None)

        # TRUE surface from GT CN
        true_surf = reconstruct_surface_from_ctrlnet(
            orig_cn, n_u, n_v, degree_u, degree_v, kv_u_true, kv_v_true
        )
        pts_true = true_surf.reshape(-1, 3)

        # predicted control-net from model input (schema-aware)
        x = prepare_input_for_model(e, target_shape=target_shape, schema=s)
        pred = model.predict(x)
        pred = np.asarray(pred, dtype=float)
        if pred.ndim == 4 and pred.shape[0] == 1:
            pred = pred[0]
        elif pred.ndim != 3:
            raise ValueError(
                f"Model output shape {pred.shape} is not supported. Expected (u,v,3) or (1,u,v,3)."
            )

        pred_cn = remove_negative_and_shrink_control_net(pred)

        # surface from predicted CN
        can_build_pred_surface = (
            pred_cn.shape[0] >= degree_u + 1 and
            pred_cn.shape[1] >= degree_v + 1 and
            np.isfinite(pred_cn).all()
        )
        if can_build_pred_surface:
            #TODO using GT knotvector
            pred_surf = reconstruct_surface_from_ctrlnet(pred_cn, n_u, n_v, degree_u, degree_v, kv_u_true, kv_v_true)
            pts_pred = pred_surf.reshape(-1, 3)
            mse_surf_pred_true.append(float(np.mean((true_surf - pred_surf) ** 2)))
            da, _ = cKDTree(pts_true).query(pts_pred)
            db, _ = cKDTree(pts_pred).query(pts_true)
            cd_surf_pred_true.append(float(np.mean(da**2) + np.mean(db**2)))
        else:
            mse_surf_pred_true.append(float('nan'))
            cd_surf_pred_true.append(float('nan'))

        # approximation baseline (fit from observed surface)
        ctrl_u, ctrl_v, _ = orig_cn.shape
        appr_surf_obj, appr_cn = approximate_control_net(
            orig_surf_obs, degree_u, degree_v, ctrl_u, ctrl_v, centri
        )
        appr_surf = reconstruct_surface_from_ctrlnet(
            appr_cn, n_u, n_v, degree_u, degree_v,
            #appr_surf_obj.knotvector_u, appr_surf_obj.knotvector_v
            kv_u_true, kv_v_true
        )
        pts_app = appr_surf.reshape(-1, 3)
        mse_surf_app_true.append(float(np.mean((true_surf - appr_surf) ** 2)))
        da, _ = cKDTree(pts_true).query(pts_app)
        db, _ = cKDTree(pts_app).query(pts_true)
        cd_surf_app_true.append(float(np.mean(da**2) + np.mean(db**2)))

        # CN metrics (cropped MSE)
        mu_p = min(orig_cn.shape[0], pred_cn.shape[0])
        mv_p = min(orig_cn.shape[1], pred_cn.shape[1])
        if mu_p > 0 and mv_p > 0:
            o_crop = orig_cn[:mu_p, :mv_p].reshape(-1, 3)
            p_crop = pred_cn[:mu_p, :mv_p].reshape(-1, 3)
            mse_cn_pred_crop.append(float(np.mean((p_crop - o_crop) ** 2)))
        else:
            mse_cn_pred_crop.append(float('nan'))

        mu_a = min(orig_cn.shape[0], appr_cn.shape[0])
        mv_a = min(orig_cn.shape[1], appr_cn.shape[1])
        if mu_a > 0 and mv_a > 0:
            o_app = orig_cn[:mu_a, :mv_a].reshape(-1, 3)
            a_crop = appr_cn[:mu_a, :mv_a].reshape(-1, 3)
            mse_cn_app_crop.append(float(np.mean((a_crop - o_app) ** 2)))
        else:
            mse_cn_app_crop.append(float('nan'))

        # CN Chamfer on full sets
        def _ch(A, B):
            da, _ = cKDTree(A).query(B)
            db, _ = cKDTree(B).query(A)
            return float(np.mean(da**2) + np.mean(db**2))

        o_full = orig_cn.reshape(-1, 3)
        p_full = pred_cn.reshape(-1, 3)
        a_full = appr_cn.reshape(-1, 3)

        cd_cn_pred_full.append(_ch(p_full, o_full))
        cd_cn_app_full.append(_ch(a_full, o_full))

        if pred_cn.shape != orig_cn.shape:
            shape_mismatch += 1

    total = len(entries)
    def _nanmean(x): return float(np.nanmean(x)) if x else float('nan')
    def _nanmedian(x): return float(np.nanmedian(x)) if x else float('nan')
    mismatch_ratio = (shape_mismatch / total) if total > 0 else float('nan')

    return {
        "total_entries": total,
        "shape_mismatch_count": shape_mismatch,
        "mismatch_ratio": mismatch_ratio,

        "avg_mse_cn_pred_cropped": _nanmean(mse_cn_pred_crop),
        "median_mse_cn_pred_cropped": _nanmedian(mse_cn_pred_crop),
        "avg_mse_cn_app_cropped": _nanmean(mse_cn_app_crop),
        "median_mse_cn_app_cropped": _nanmedian(mse_cn_app_crop),

        "avg_ch_cn_pred_full": _nanmean(cd_cn_pred_full),
        "median_ch_cn_pred_full": _nanmedian(cd_cn_pred_full),
        "avg_ch_cn_app_full": _nanmean(cd_cn_app_full),
        "median_ch_cn_app_full": _nanmedian(cd_cn_app_full),

        "avg_mse_surf_pred_true": _nanmean(mse_surf_pred_true),
        "median_mse_surf_pred_true": _nanmedian(mse_surf_pred_true),
        "avg_ch_surf_pred_true": _nanmean(cd_surf_pred_true),
        "median_ch_surf_pred_true": _nanmedian(cd_surf_pred_true),

        "avg_mse_surf_app_true": _nanmean(mse_surf_app_true),
        "median_mse_surf_app_true": _nanmedian(mse_surf_app_true),
        "avg_ch_surf_app_true": _nanmean(cd_surf_app_true),
        "median_ch_surf_app_true": _nanmedian(cd_surf_app_true),
    }



import numpy as np
import warnings
from typing import Optional, Union, Callable, Tuple

ArrayLike = Union[np.ndarray, float, int]

def add_noise(
    pts: np.ndarray,
    kind: str = "gaussian",
    std: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    outlier_prob: float = 0.1,
    outlier_scale: float = 10.0,
    df: float = 3.0,
    sigma: Optional[Union[ArrayLike, Callable[[np.ndarray], ArrayLike]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add noise of a specified kind to pts, scaled to achieve approximately the same std.
    Supported kinds: gaussian, uniform, laplace, student_t, cauchy, impulsive, heteroscedastic.
    Falls back to gaussian if kind invalid or missing.
    """
    if rng is None:
        rng = np.random.default_rng()

    kind = (kind or "gaussian").lower()
    valid = {"gaussian", "uniform", "laplace", "student_t", "cauchy", "impulsive", "heteroscedastic"}
    if kind not in valid:
        warnings.warn(f"Unknown noise kind '{kind}', falling back to 'gaussian'.")
        kind = "gaussian"

    # --- Each branch rescales noise to roughly match desired std ---
    if kind == "gaussian":
        noise = rng.normal(0.0, std, size=pts.shape)

    elif kind == "uniform":
        # Uniform(-a, a) → variance = a^2/3  => a = sqrt(3)*std
        a = np.sqrt(3.0) * std
        noise = rng.uniform(-a, a, size=pts.shape)

    elif kind == "laplace":
        # Laplace(0, b) → variance = 2*b^2  => b = std / sqrt(2)
        b = std / np.sqrt(2.0)
        noise = rng.laplace(0.0, b, size=pts.shape)

    elif kind == "student_t":
        # Scale so var matches std^2 (if df>2)
        t = rng.standard_t(df, size=pts.shape)
        if df > 2:
            t_scale = std / np.sqrt(df / (df - 2.0))
        else:
            t_scale = std  # infinite variance, just scale down
        noise = t * t_scale

    elif kind == "cauchy":
        # Cauchy variance infinite, scale so median absolute deviation ≈ std
        scale = std / 1.4826  # since MAD for Cauchy ≈ 1.4826*scale
        noise = rng.standard_cauchy(size=pts.shape) * scale

    elif kind == "impulsive":
        # 90% normal(0,std), 10% normal(0, outlier_scale*std)
        base = rng.normal(0.0, std, size=pts.shape)
        outl = rng.normal(0.0, outlier_scale * std, size=pts.shape)
        mask = rng.random(size=pts.shape) < outlier_prob
        noise = np.where(mask, outl, base)

    elif kind == "heteroscedastic":
        if sigma is None:
            sigma = lambda x: std * (1 + 0.5 * np.abs(x))
        sig = sigma(pts) if callable(sigma) else np.broadcast_to(sigma, pts.shape)
        noise = rng.normal(0.0, sig, size=pts.shape)

    else:
        noise = rng.normal(0.0, std, size=pts.shape)

    return pts + noise, noise




def compare_across_noise_levels(
        entries_noiseless,
        model,
        noise_levels=None,
        target_shape=(35, 35, 3),
        threshold=0.25,
        padding_value=-0.1,
        padding_tol=0.11,
        degree_u=3,
        degree_v=3,
        noise_type="gaussian",
        centri=True,
        *,
        schema: Schema = "auto"   # <— NEW
):
    """
    Adds schema awareness; when injecting noise we perturb whichever
    surface key is active for the entry ('rotated_noisy_points' or 'points').
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02]

    records = []
    for std in noise_levels:
        noisy_entries = []
        for e in entries_noiseless:
            new_e = copy.deepcopy(e)
            s = detect_schema(new_e) if schema == "auto" else schema

            # read current observed surface
            pts = np.array(
                pick(new_e, "surface_points", candidates=aliases_for("surface_points", s)),
                dtype=float
            )
            #noise = np.random.normal(loc=0.0, scale=std, size=pts.shape)
            #noisy = pts + noise

            noisy, _ = add_noise(pts, noise_type, std)



            # write back into the correct key for this schema
            if s == "rotated":
                new_e["rotated_noisy_points"] = noisy
            else:
                new_e["points"] = noisy

            noisy_entries.append(new_e)

        stats = evaluate_dataset_control_nets(
            noisy_entries,
            model,
            target_shape=target_shape,
            threshold=threshold,
            padding_value=padding_value,
            padding_tol=padding_tol,
            degree_u=degree_u,
            degree_v=degree_v,
            schema=schema,  # pass through
            centri=centri
        )
        stats["noise_std"] = std
        records.append(stats)

    df = pd.DataFrame.from_records(records)
    df.set_index("noise_std", inplace=True)
    return df




# Example usage:
# import pickle, tensorflow as tf
# with open("dataset/noiseless_dataset.pkl","rb") as f:
#     entries_clean = pickle.load(f)["data"]
# model = tf.keras.models.load_model("models/your_model.keras")
# df_noise = compare_across_noise_levels(entries_clean, model)
# print(df_noise.to_markdown(floatfmt=".4f"))



# -----------------------------------------------------------------------------
# For testing or interactive use
if __name__ == '__main__':
    # Example usage:
    # Load or create a dataset entry (using your data generation library) and a predicted control net,
    # then call the functions below for comparison.
    pass
