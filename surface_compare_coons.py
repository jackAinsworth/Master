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
from typing import Union



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


# -----------------------------------------------------------------------------
# NURBS surface reconstruction functions

    # ------------------------------------------------------------------
    # helper: approximate a control-net from a noisy surface the *right* way
    # ------------------------------------------------------------------
def approximate_control_net(surface, degree_u, degree_v, ctrlpts_size_u, ctrlpts_size_v, centri=True):
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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# -----------------------------------------------------------------------------
# Utility helpers expected to be available elsewhere in the code‑base. We keep
# the imports here to make the function self‑contained but **do not** re‑define
# them – it is assumed that they are already available in the surrounding
# scope:
#   • reconstruct_surface_from_ctrlnet(ctrl_net, nu, nv, deg_u, deg_v)
#   • chamfer_distance(a_pts, b_pts)
#   • set_axes_equal(ax)
# -----------------------------------------------------------------------------

def compare_surfaces(
    original_entry,
    predicted_ctrl_net,
    num_samples: int = 35,
    show_control_net: bool = True,
    degree_u: int = 3,
    degree_v: int = 3,
):
    """Compare *original* vs *predicted* Coons/NURBS surfaces.

    The predicted surface is reconstructed from a control‑net tensor coming out
    of a neural network.  The function visualises both surfaces in 3‑D,
    colours the prediction by the point‑wise Euclidean error, and returns four
    key error metrics.

    Notes
    -----
    * **Padding removal** – Any padding has already been stripped *before* this
      function is called, so no further clean‑up is performed here.
    * Supports both the legacy field names (``rotated_noisy_points`` /
      ``rotated_control_net``) and the newer ones (``points`` / ``control_net``)
      that come out of *create_dataset_entry*.
    """

    # ------------------------------------------------------------------
    # 1) Reconstruct the predicted and fetch the original surface at the
    #    same resolution.
    # ------------------------------------------------------------------
    orig_surf = np.array(
        original_entry.get("rotated_noisy_points", original_entry["points"])
    )
    nu, nv, _ = orig_surf.shape

    # Rebuild predicted surface at matching resolution if necessary
    pred_surf = reconstruct_surface_from_ctrlnet(
        predicted_ctrl_net, nu, nv, degree_u, degree_v
    )

    # ------------------------------------------------------------------
    # 2) Per‑point Euclidean error & surface‑level metrics
    # ------------------------------------------------------------------
    error_vec = orig_surf - pred_surf
    error_map = np.linalg.norm(error_vec, axis=2)
    mse_surface = np.mean(error_vec ** 2)

    chamfer_surf = chamfer_distance(
        orig_surf.reshape(-1, 3), pred_surf.reshape(-1, 3)
    )

    # ------------------------------------------------------------------
    # 3) Control‑net metrics (no padding removal – already handled upstream)
    # ------------------------------------------------------------------
    orig_cn = np.array(
        original_entry.get("rotated_control_net", original_entry["control_net"])
    )

    # If the network kept a leading batch‑dim, drop it
    pred_cn = predicted_ctrl_net[0] if predicted_ctrl_net.ndim == 4 else predicted_ctrl_net

    # Compare only the overlapping area if the two nets differ in size
    min_u = min(orig_cn.shape[0], pred_cn.shape[0])
    min_v = min(orig_cn.shape[1], pred_cn.shape[1])

    mse_cn = np.mean(
        (orig_cn[:min_u, :min_v].reshape(-1, 3) - pred_cn[:min_u, :min_v].reshape(-1, 3))
        ** 2
    )

    chamfer_cn = chamfer_distance(
        orig_cn[:min_u, :min_v].reshape(-1, 3), pred_cn[:min_u, :min_v].reshape(-1, 3)
    )

    # ------------------------------------------------------------------
    # 4) Visualisation
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 6))

    # (A) Original surface – solid purple for context
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        orig_surf[:, :, 0], orig_surf[:, :, 1], orig_surf[:, :, 2],
        color="purple", edgecolor="none", alpha=0.9,
    )
    ax1.set_title("Original surface")

    # Optionally overlay the original control net
    if show_control_net:
        for i in range(orig_cn.shape[0]):
            ax1.plot(orig_cn[i, :, 0], orig_cn[i, :, 1], orig_cn[i, :, 2], "r--o", ms=3)
        for j in range(orig_cn.shape[1]):
            ax1.plot(orig_cn[:, j, 0], orig_cn[:, j, 1], orig_cn[:, j, 2], "r--o", ms=3)

    set_axes_equal(ax1)

    # (B) Predicted surface coloured by error magnitude
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf_plot = ax2.plot_surface(
        pred_surf[:, :, 0], pred_surf[:, :, 1], pred_surf[:, :, 2],
        facecolors=cm.inferno(error_map / error_map.max()),
        rcount=nu, ccount=nv,
        linewidth=0, antialiased=False,
    )
    ax2.set_title(
        (
            "Predicted surface\n"
            f"MSE = {mse_surface:.4f}   "
            f"Chamfer = {chamfer_surf:.4f}"
        )
    )

    # Colour‑bar for the error map
    m = cm.ScalarMappable(cmap=cm.inferno)
    m.set_array(error_map)
    fig.colorbar(m, ax=ax2, shrink=0.6, label="Euclidean error (×10⁻³)")

    # Optionally overlay the predicted control net
    if show_control_net:
        for i in range(pred_cn.shape[0]):
            ax2.plot(pred_cn[i, :, 0], pred_cn[i, :, 1], pred_cn[i, :, 2], "r--o", ms=3)
        for j in range(pred_cn.shape[1]):
            ax2.plot(pred_cn[:, j, 0], pred_cn[:, j, 1], pred_cn[:, j, 2], "r--o", ms=3)

    set_axes_equal(ax2)

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 5) Return metrics for downstream evaluation
    # ------------------------------------------------------------------
    return {
        "mse_surface": mse_surface,
        "chamfer_surface": chamfer_surf,
        "mse_ctrl_net": mse_cn,
        "chamfer_ctrl_net": chamfer_cn,
    }



def compare_control_nets_in_one_plot(
    original: Union[np.ndarray, dict],
    predicted_cn: np.ndarray,
):
    """Overlay original vs. predicted control nets in a single 3‑D figure.

    ``original`` can be either:
      * a *dataset entry* (dict) returned by KoonsDataGenerator, **or**
      * the raw original control net as an ``(u,v,3)`` array (optionally
        batched as ``(1,u,v,3)``).

    ``predicted_cn`` must be an ``(u,v,3)`` array or batched variant.
    """
    # --- Extract / un‑batch ----------------------------------------------
    if isinstance(original, dict):
        original_cn = original.get("control_net")
        if original_cn is None:
            original_cn = original["rotated_control_net"]
    else:
        original_cn = np.asarray(original)

    if original_cn.ndim == 4:
        original_cn = original_cn[0]

    if predicted_cn.ndim == 4:
        predicted_cn = predicted_cn[0]

    # --- Reconcile shape differences via cropping ------------------------
    if original_cn.shape != predicted_cn.shape:
        min_u = min(original_cn.shape[0], predicted_cn.shape[0])
        min_v = min(original_cn.shape[1], predicted_cn.shape[1])
        original_cn = original_cn[:min_u, :min_v]
        predicted_cn = predicted_cn[:min_u, :min_v]
        print("Warning: control nets had different shapes – cropped to common subset for plotting.")

    # --- Plot ------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Original (blue circles)
    u_o, v_o, _ = original_cn.shape
    for i in range(u_o):
        ax.plot(
            original_cn[i, :, 0],
            original_cn[i, :, 1],
            original_cn[i, :, 2],
            color="blue",
            linestyle="--",
            marker="o",
            markersize=4,
            label="Original CN" if i == 0 else None,
        )
    for j in range(v_o):
        ax.plot(
            original_cn[:, j, 0],
            original_cn[:, j, 1],
            original_cn[:, j, 2],
            color="blue",
            linestyle="--",
            marker="o",
            markersize=4,
        )

    # Predicted (red triangles)
    u_p, v_p, _ = predicted_cn.shape
    for i in range(u_p):
        ax.plot(
            predicted_cn[i, :, 0],
            predicted_cn[i, :, 1],
            predicted_cn[i, :, 2],
            color="red",
            linestyle="--",
            marker="^",
            markersize=4,
            label="Predicted CN" if i == 0 else None,
        )
    for j in range(v_p):
        ax.plot(
            predicted_cn[:, j, 0],
            predicted_cn[:, j, 1],
            predicted_cn[:, j, 2],
            color="red",
            linestyle="--",
            marker="^",
            markersize=4,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Original vs. Predicted Control Nets")
    ax.legend()
    set_axes_equal(ax)
    plt.show()

def compare_approximated_control_nets(
    original_entry: dict,
    predicted_ctrl_net: np.ndarray,
    degree_u: int = 3,
    degree_v: int = 3,
    ctrlpts_size_u: int = None,
    ctrlpts_size_v: int = None,
):
    """Three‑way comparison **(approximated vs. predicted vs. original)**.

    Steps
    -----
    1. Build a *reference* control net by fitting a NURBS surface to the **noisy
       surface points** contained in ``original_entry`` (this mimics the
       procedure used during dataset creation).
    2. Collect the *ground‑truth* control net from ``original_entry``.
    3. Compare both the *predicted* control net **and** the *approximated* one
       against the ground‑truth via MSE + symmetric Chamfer distance.
    4. Plot all three nets in a single 3‑D figure (blue = approximated,
       red = predicted, green = original).
    """
    # ------------------------------------------------------------------
    # Un‑batch predicted control net
    # ------------------------------------------------------------------
    if predicted_ctrl_net.ndim == 4:
        predicted_cn = predicted_ctrl_net[0]
    else:
        predicted_cn = predicted_ctrl_net

    # ------------------------------------------------------------------
    # Retrieve original control net (+ noisy surface)
    # ------------------------------------------------------------------
    orig_cn = original_entry.get("control_net")
    if orig_cn is None:
        orig_cn = original_entry["rotated_control_net"]
    if orig_cn.ndim == 4:
        orig_cn = orig_cn[0]

    surface_pts = original_entry.get("points")
    if surface_pts is None:
        surface_pts = original_entry["rotated_noisy_points"]
    surface_pts = np.asarray(surface_pts)

    # ------------------------------------------------------------------
    # Choose control‑net sizes for approximation if not supplied
    # ------------------------------------------------------------------
    if ctrlpts_size_u is None or ctrlpts_size_v is None:
        ctrlpts_size_u, ctrlpts_size_v, _ = orig_cn.shape

    # ------------------------------------------------------------------
    # Approximate control net from noisy surface
    # ------------------------------------------------------------------
    _, approx_cn = approximate_control_net(
        surface_pts, degree_u, degree_v, ctrlpts_size_u, ctrlpts_size_v
    )

    # ------------------------------------------------------------------
    # Metric calculation (crop to common overlap)
    # ------------------------------------------------------------------
    mu = min(orig_cn.shape[0], predicted_cn.shape[0], approx_cn.shape[0])
    mv = min(orig_cn.shape[1], predicted_cn.shape[1], approx_cn.shape[1])

    orig_crop   = orig_cn[:mu, :mv].reshape(-1, 3)
    pred_crop   = predicted_cn[:mu, :mv].reshape(-1, 3)
    approx_crop = approx_cn[:mu, :mv].reshape(-1, 3)

    mse_pred   = float(np.mean((pred_crop - orig_crop) ** 2))
    mse_approx = float(np.mean((approx_crop - orig_crop) ** 2))

    cd_pred   = float(chamfer_distance(pred_crop, orig_crop))
    cd_approx = float(chamfer_distance(approx_crop, orig_crop))

    # ------------------------------------------------------------------
    # Print summary to console
    # ------------------------------------------------------------------
    print(
        f"Control‑net shapes (u×v): orig={orig_cn.shape[:2]}, "
        f"pred={predicted_cn.shape[:2]}, approx={approx_cn.shape[:2]}"
    )
    print(f"MSE  – Predicted vs Original    : {mse_pred:.6f}")
    print(f"MSE  – Approximated vs Original : {mse_approx:.6f}")
    print(f"Chamfer – Predicted vs Original    : {cd_pred:.6f}")
    print(f"Chamfer – Approximated vs Original : {cd_approx:.6f}\n")

    # ------------------------------------------------------------------
    # Plot three control nets together
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Approximated – blue circles
    u_a, v_a, _ = approx_cn.shape
    for i in range(u_a):
        ax.plot(
            approx_cn[i, :, 0], approx_cn[i, :, 1], approx_cn[i, :, 2],
            color="blue", linestyle="--", marker="o",
            label="Approximated CN" if i == 0 else None,
        )
    for j in range(v_a):
        ax.plot(
            approx_cn[:, j, 0], approx_cn[:, j, 1], approx_cn[:, j, 2],
            color="blue", linestyle="--", marker="o",
        )

    # Predicted – red triangles
    u_p, v_p, _ = predicted_cn.shape
    for i in range(u_p):
        ax.plot(
            predicted_cn[i, :, 0], predicted_cn[i, :, 1], predicted_cn[i, :, 2],
            color="red", linestyle="--", marker="^",
            label="Predicted CN" if i == 0 else None,
        )
    for j in range(v_p):
        ax.plot(
            predicted_cn[:, j, 0], predicted_cn[:, j, 1], predicted_cn[:, j, 2],
            color="red", linestyle="--", marker="^",
        )

    # Original – green squares
    u_o, v_o, _ = orig_cn.shape
    for i in range(u_o):
        ax.plot(
            orig_cn[i, :, 0], orig_cn[i, :, 1], orig_cn[i, :, 2],
            color="green", linestyle="--", marker="s",
            label="Original CN" if i == 0 else None,
        )
    for j in range(v_o):
        ax.plot(
            orig_cn[:, j, 0], orig_cn[:, j, 1], orig_cn[:, j, 2],
            color="green", linestyle="--", marker="s",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Control Nets: Approximated vs Predicted vs Original")
    ax.legend()
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()

    return {
        "mse_predicted": mse_pred,
        "mse_approx": mse_approx,
        "chamfer_predicted": cd_pred,
        "chamfer_approx": cd_approx,
    }





def prepare_input_for_model(dataset_entry, target_shape=(100, 100, 3)):
    input_data = np.array(dataset_entry["rotated_noisy_points"], dtype=np.float32)
    if input_data.shape != target_shape:
        raise ValueError(f"Input data shape {input_data.shape} does not match the expected shape {target_shape}.")
    input_data = np.expand_dims(input_data, axis=0)
    return input_data



def evaluate_dataset_control_nets(
    entries,
    model,
    target_shape=(35, 35, 3),
    threshold=10,
    padding_value=-10,
    padding_tol=20,
    degree_u=3,
    degree_v=3,
):
    """
    Evaluate predicted and approximated control-nets (and the corresponding
    reconstructed surfaces) against the original ground-truth data.

    The surface-approximation step is now *identical* to the logic used in
    `compare_approximated_control_nets()`: the noisy surface is first
    down-sampled to the exact lattice expected by `geomdl.fitting.approximate_surface`
    before fitting, guaranteeing that both functions use the same procedure.

    Returns
    -------
    dict
        Aggregated MSE and Chamfer-distance statistics (control-net level,
        surface level, and mismatched-shape cases).
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from geomdl import fitting

    # ------------------------------------------------------------------
    # helper: approximate a control-net from a noisy surface the *right* way
    # ------------------------------------------------------------------
    def approximate_control_net(surface, degree_u, degree_v, ctrlpts_size_u, ctrlpts_size_v):
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
            centripetal=True
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

    # -------------------- accumulators --------------------------------
    mse_cn,        cd_cn        = [], []
    mse_app_cn,    cd_app_cn    = [], []
    mse_surf_pred_all, cd_surf_pred_all = [], []
    mse_surf_app_all,  cd_surf_app_all  = [], []
    mse_surf_pred, cd_surf_pred = [], []
    mse_surf_app,  cd_surf_app  = [], []
    shape_mismatch = 0

    # -------------------- main loop -----------------------------------
    for e in entries:
        # ---------- originals ----------
        orig_surf = np.asarray(e["rotated_noisy_points"])
        pts_o = orig_surf.reshape(-1, 3)

        orig_cn = e["rotated_control_net"]
        if orig_cn.ndim == 4:  # remove batch dim if present
            orig_cn = orig_cn[0]

        # ---------- predicted control-net ----------
        x = prepare_input_for_model(e, target_shape)
        pred = model.predict(x)
        pred_cn = remove_padding_and_shrink_control_net(
            pred, threshold, padding_value, padding_tol
        )
        if pred_cn.ndim == 4:
            pred_cn = pred_cn[0]

        # ---------- surface from predicted control-net ----------
        pred_surf = reconstruct_surface_from_ctrlnet(
            pred_cn, orig_surf.shape[0], orig_surf.shape[1], degree_u, degree_v
        )

        pts_p = pred_surf.reshape(-1, 3)
        mse_pred_all = np.mean((orig_surf - pred_surf) ** 2)
        da, _ = cKDTree(pts_o).query(pts_p)
        db, _ = cKDTree(pts_p).query(pts_o)
        cd_pred_all = np.mean(da ** 2) + np.mean(db ** 2)
        mse_surf_pred_all.append(mse_pred_all)
        cd_surf_pred_all.append(cd_pred_all)

        print("mse_pred_all", mse_pred_all, "cd_pred_all", cd_pred_all)

        # ---------- approximated control-net (same logic as compare_… ) ----------
        ctrlpts_size_u, ctrlpts_size_v, _ = orig_cn.shape

        appr_surf, appr_cn = approximate_control_net(
            orig_surf,3,3, ctrlpts_size_u, ctrlpts_size_v
        )


        # ---------- surface from approximated control-net ----------
        appr_surf = reconstruct_surface_from_ctrlnet(
            appr_cn, orig_surf.shape[0], orig_surf.shape[1], degree_u, degree_v, appr_surf.knotvector_u, appr_surf.knotvector_v
            #appr_cn, orig_surf.shape[0], orig_surf.shape[1], degree_u, degree_v
        )

        pts_a = appr_surf.reshape(-1, 3)
        mse_app_all = np.mean((orig_surf - appr_surf) ** 2)
        da, _ = cKDTree(pts_o).query(pts_a)
        db, _ = cKDTree(pts_a).query(pts_o)
        cd_app_all = np.mean(da ** 2) + np.mean(db ** 2)
        mse_surf_app_all.append(mse_app_all)
        cd_surf_app_all.append(cd_app_all)
        print("mse_app_all", mse_app_all, "cd_app_all", cd_app_all)


        # ---------- control-net-level metrics (only if shapes match) ----------
        if pred_cn.shape == orig_cn.shape:
            mu = min(orig_cn.shape[0], pred_cn.shape[0], appr_cn.shape[0])
            mv = min(orig_cn.shape[1], pred_cn.shape[1], appr_cn.shape[1])

            o = orig_cn[:mu, :mv].reshape(-1, 3)
            p = pred_cn[:mu, :mv].reshape(-1, 3)
            a = appr_cn[:mu, :mv].reshape(-1, 3)

            mse_cn.append(np.mean((p - o) ** 2))
            mse_app_cn.append(np.mean((a - o) ** 2))

            print('mse app cn', np.mean((a - o) ** 2))


            def _ch(A, B):
                da, _ = cKDTree(A).query(B)
                db, _ = cKDTree(B).query(A)
                return np.mean(da ** 2) + np.mean(db ** 2)

            cd_cn.append(_ch(p, o))
            cd_app_cn.append(_ch(a, o))
        else:
            # ---------- shapes do not match; keep surface-level stats ----------
            shape_mismatch += 1
            mse_surf_pred.append(mse_pred_all)
            cd_surf_pred.append(cd_pred_all)
            mse_surf_app.append(mse_app_all)
            cd_surf_app.append(cd_app_all)

    # -------------------- aggregate results ---------------------------
    total = len(entries)
    # compute medians in the same fashion
    median_mse_cn = float(np.median(mse_cn)) if mse_cn else np.nan
    median_ch_cn = float(np.median(cd_cn)) if cd_cn else np.nan
    median_mse_app_cn = float(np.median(mse_app_cn)) if mse_app_cn else np.nan
    median_ch_app_cn = float(np.median(cd_app_cn)) if cd_app_cn else np.nan

    median_mse_surf_pred_all = float(np.median(mse_surf_pred_all))
    median_ch_surf_pred_all = float(np.median(cd_surf_pred_all))
    median_mse_surf_app_all = float(np.median(mse_surf_app_all))
    median_ch_surf_app_all = float(np.median(cd_surf_app_all))

    median_mse_surf_pred = float(np.median(mse_surf_pred)) if mse_surf_pred else np.nan
    median_ch_surf_pred = float(np.median(cd_surf_pred)) if cd_surf_pred else np.nan
    median_mse_surf_app = float(np.median(mse_surf_app)) if mse_surf_app else np.nan
    median_ch_surf_app = float(np.median(cd_surf_app)) if cd_surf_app else np.nan

    return {
        "total_entries": total,
        "shape_mismatch_count": shape_mismatch,
        "mismatch_ratio": shape_mismatch / total,

        # control-net MEANs
        "avg_mse_predicted": float(np.mean(mse_cn)) if mse_cn else np.nan,
        "avg_ch_predicted": float(np.mean(cd_cn)) if cd_cn else np.nan,
        "avg_mse_approximated": float(np.mean(mse_app_cn)) if mse_app_cn else np.nan,
        "avg_ch_approximated": float(np.mean(cd_app_cn)) if cd_app_cn else np.nan,

        # control-net MEDIANs
        "median_mse_predicted": median_mse_cn,
        "median_ch_predicted": median_ch_cn,
        "median_mse_approximated": median_mse_app_cn,
        "median_ch_approximated": median_ch_app_cn,

        # surface-all MEANs
        "avg_mse_surf_pred_all": float(np.mean(mse_surf_pred_all)),
        "avg_ch_surf_pred_all": float(np.mean(cd_surf_pred_all)),
        "avg_mse_surf_app_all": float(np.mean(mse_surf_app_all)),
        "avg_ch_surf_app_all": float(np.mean(cd_surf_app_all)),

        # surface-all MEDIANs
        "median_mse_surf_pred_all": median_mse_surf_pred_all,
        "median_ch_surf_pred_all": median_ch_surf_pred_all,
        "median_mse_surf_app_all": median_mse_surf_app_all,
        "median_ch_surf_app_all": median_ch_surf_app_all,

        # mismatched-shape surface MEANs
        "avg_mse_surf_pred": float(np.mean(mse_surf_pred)) if mse_surf_pred else np.nan,
        "avg_ch_surf_pred": float(np.mean(cd_surf_pred)) if cd_surf_pred else np.nan,
        "avg_mse_surf_app": float(np.mean(mse_surf_app)) if mse_surf_app else np.nan,
        "avg_ch_surf_app": float(np.mean(cd_surf_app)) if cd_surf_app else np.nan,

        # mismatched-shape surface MEDIANs
        "median_mse_surf_pred": median_mse_surf_pred,
        "median_ch_surf_pred": median_ch_surf_pred,
        "median_mse_surf_app": median_mse_surf_app,
        "median_ch_surf_app": median_ch_surf_app,
    }


def compare_across_noise_levels(
        entries_noiseless,
        model,
        noise_levels=None,
        target_shape=(35, 35, 3),
        threshold=10,
        padding_value=-10,
        padding_tol=20,
        degree_u=3,
        degree_v=3
):
    """
    Evaluate model performance across different Gaussian noise levels.

    Args:
        entries_noiseless (list of dict): Dataset entries with noiseless 'rotated_noisy_points'.
        model (tf.keras.Model): Trained model to predict control nets.
        noise_levels (list of float): Noise standard deviations to test.
                                     Defaults to [0.1, 0.5, 1.0, 2.0, 4.0].
        Other args: passed through to evaluate_dataset_control_nets.

    Returns:
        pd.DataFrame: Rows indexed by noise level, columns are stats keys.
    """
    if noise_levels is None:
        noise_levels = [0.1, 0.5, 1.0, 2.0, 4.0]

    records = []
    for std in noise_levels:
        # Create noisy entries
        noisy_entries = []
        for e in entries_noiseless:
            new_e = copy.deepcopy(e)
            pts = np.array(e["rotated_noisy_points"], dtype=float)
            noise = np.random.normal(loc=0.0, scale=std, size=pts.shape)
            new_e["rotated_noisy_points"] = pts + noise
            noisy_entries.append(new_e)

        # Evaluate
        stats = evaluate_dataset_control_nets(
            noisy_entries,
            model,
            target_shape=target_shape,
            threshold=threshold,
            padding_value=padding_value,
            padding_tol=padding_tol,
            degree_u=degree_u,
            degree_v=degree_v
        )
        stats["noise_std"] = std
        records.append(stats)

    # Build DataFrame
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

    orig_cn = original_entry["control_net"]
    if orig_cn.ndim == 4:
        orig_cn = orig_cn[0]

    # derive sizes if not given
    if ctrlpts_size_u is None or ctrlpts_size_v is None:
        ctrlpts_size_u, ctrlpts_size_v, _ = orig_cn.shape

    # Original surface (used as reference)
    orig_surf = np.asarray(original_entry["points"])
    if num_samples is None:
        su, sv = orig_surf.shape[:2]
    else:
        su = sv = int(num_samples)

    # ------------------ build helper surfaces ------------------
    surface_points = original_entry["points"]

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



# -----------------------------------------------------------------------------
# For testing or interactive use
if __name__ == '__main__':
    # Example usage:
    # Load or create a dataset entry (using your data generation library) and a predicted control net,
    # then call the functions below for comparison.
    pass
