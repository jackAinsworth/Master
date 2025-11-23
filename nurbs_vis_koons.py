"""
nurbs_vis.py

A collection of visualization functions for the NURBS surface generation workflow.
This module provides functions to visualize:
  - 2D B-Spline curves with control points and vertical sections,
  - 3D curves (lifted from 2D),
  - The interpolated surface between two 3D curves,
  - The fitted NURBS surface with its control net,
  - And a complete dataset entry (rotated noisy surface points and control net)
    from a saved dataset.

Usage:
  Import this module and call the desired visualization function, e.g.:
      import nurbs_vis
      nurbs_vis.visualize_curve_2d(control_points, bspline, section_bounds)
      nurbs_vis.visualize_dataset_entry(dataset_entry)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Ensures 3D plotting
from matplotlib import cm


# Enable interactive mode (optional; you can also set your matplotlib backend externally)
plt.ion()


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


def get_camera_direction(ax):
    """
    Convert Matplotlib 3D ax.azim (azimuth) and ax.elev (elevation) into
    a 3D unit direction vector. This is approximate for the default Matplotlib camera.

    ax.azim: Angle in degrees, 0 = +x, 90 = +y
    ax.elev: Angle in degrees, 0 = xy-plane, + is above
    """
    # Convert angles to radians
    elev_r = np.radians(ax.elev)
    azim_r = np.radians(ax.azim)

    # Based on the standard spherical coords, but note that Matplotlib's orientation
    # might differ slightly. This is a common approximation:
    # x = r cos(elev) cos(azim)
    # y = r cos(elev) sin(azim)
    # z = r sin(elev)
    camera_dir = np.array([
        np.cos(elev_r) * np.cos(azim_r),
        np.cos(elev_r) * np.sin(azim_r),
        np.sin(elev_r)
    ])
    return camera_dir


def color_by_camera_view(ax, points):
    """
    Given a set of points (N x 3) and the current camera orientation (from ax),
    return a color array labeling points that are 'in front' (dot > 0) or 'behind' (dot < 0)
    relative to the camera direction.

    This approach:
    1) Finds the camera direction (a unit vector).
    2) Finds the bounding center of points.
    3) For each point, compute dot product with camera direction.
       If dot > 0 => 'in front', else => 'behind'.
    """
    camera_dir = get_camera_direction(ax)
    center = points.mean(axis=0)  # e.g. the centroid
    # Vector from center to each point
    vecs = points - center
    # Dot product with camera direction
    dots = np.einsum('ij,j->i', vecs, camera_dir)  # or (vecs @ camera_dir)
    # For demonstration: in_front if dot>0 => color='red', else color='blue'
    colors = np.where(dots > 0, 'red', 'red')
    return colors

def visualize_curve_2d(
    control_points,
    bspline,
    section_bounds,
    num_samples=161,
    colors=('blue', 'red'),
    section_color='green'
):
    """
    Visualize a 2D B-Spline curve along with its control points and vertical section boundaries.

    Args:
        control_points (list of tuple): Control point (x, y) coordinates.
        bspline (tuple): (spline_x, spline_y) BSpline objects.
        section_bounds (list of tuple): List of (x_min, x_max) for vertical sections.
        num_samples (int): Number of points to sample along the curve.
        colors (tuple): (curve_color, control_point_color). Defaults to ('blue', 'red').
        section_color (str): Color for section boundary lines. Defaults to 'green'.
    """
    spline_x, spline_y = bspline
    t_min = spline_x.t[spline_x.k]
    t_max = spline_x.t[-spline_x.k - 1]
    t_vals = np.linspace(t_min, t_max, num_samples)
    curve_x = spline_x(t_vals)
    curve_y = spline_y(t_vals)

    control_arr = np.array(control_points)
    curve_color, cp_color = colors

    plt.figure(figsize=(10, 6))
    # Plot the B-Spline curve
    plt.plot(curve_x, curve_y, label="B-Spline Curve", color=curve_color)
    # Plot control points
    plt.scatter(
        control_arr[:, 0], control_arr[:, 1],
        color=cp_color, marker='o', label="Control Points", zorder=5
    )
    # Draw lines connecting control points
    plt.plot(
        control_arr[:, 0], control_arr[:, 1],
        linestyle='--', color=cp_color, alpha=0.5
    )

    # Plot vertical section boundaries
    for idx, (x_min, x_max) in enumerate(section_bounds):
        label = "Section Boundaries" if idx == 0 else None
        plt.axvline(x=x_min, color=section_color, linestyle='--', label=label)
        plt.axvline(x=x_max, color=section_color, linestyle='--')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D B-Spline Curve with Control Points and Vertical Sections")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_curves_3d(
    edge_curves_3d,
    plane_sizes,
    num_samples=161
):
    """
    Visualize up to four 3D boundary curves (the four edges of a Coons patch).

    Parameters
    ----------
    edge_curves_3d : list of length 4
        [(cp, bspline3d, sec_bounds, tmin, tmax, z),  # bottom  (D0)
         (cp, bspline3d, sec_bounds, tmin, tmax, z),  # top     (D1)
         (cp, bspline3d, sec_bounds, tmin, tmax, z),  # left    (C0)
         (cp, bspline3d, sec_bounds, tmin, tmax, z)]  # right   (C1)

    plane_sizes : list of length 4
        [(width0, height0),  # size for bottom curve’s plane (z = z0)
         (width1, height1),  # size for top curve’s    plane (z = z1)
         (width2, height2),  # size for left curve’s   plane (x = x_const)
         (width3, height3)]  # size for right curve’s  plane (x = x_const)

    num_samples : int
        how many points to sample along each spline for plotting
    """
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')

    labels = ["Bottom (D0)", "Top (D1)", "Left (C0)", "Right (C1)"]
    colors = ["blue", "magenta", "cyan", "orange"]
    cp_colors = ["red", "orange", "green", "purple"]

    for idx, ((cp, bs, sec, t0, t1, z), (w, h)) \
            in enumerate(zip(edge_curves_3d, plane_sizes)):

        # sample the spline
        t_vals = np.linspace(t0, t1, num_samples)
        pts    = bs(t_vals)  # shape (num_samples, 3)

        # plot spline curve
        ax.plot(
            pts[:, 0], pts[:, 1], pts[:, 2],
            label=f"{labels[idx]} @ z={z:.2f}",
            color=colors[idx]
        )

        # plot control points
        cp_arr = np.array(cp)
        ax.scatter(
            cp_arr[:, 0], cp_arr[:, 1], cp_arr[:, 2],
            color=cp_colors[idx], marker='o', s=50,
            label=f"CP {labels[idx]}"
        )

        # vertical or horizontal section lines:
        #  - for bottom/top (idx 0 or 1), sec lists (x_min, x_max)
        #  - for left/right (idx 2 or 3), sec lists (y_min, y_max)
        if idx in (0, 1):
            # bottom/top: each (x_min, x_max) spans along X,
            #  y→[0,h], z→constant
            for (x_min, x_max) in sec or []:
                ax.plot(
                    [x_min, x_min], [0, h], [z, z],
                    linestyle='--', color='gray', alpha=0.5
                )
                ax.plot(
                    [x_max, x_max], [0, h], [z, z],
                    linestyle='--', color='gray', alpha=0.5
                )
        else:
            # left/right: each (y_min, y_max) spans along Y,
            #  x→constant, z→[0,z_gap]
            x_const = cp[0][0]  # every cp for C0/C1 shares same x
            z_vals  = np.linspace(0, z, 2)
            for (y_min, y_max) in sec or []:
                # draw a line at x_const, y=y_min from z=0→z
                ax.plot(
                    [x_const, x_const], [y_min, y_min], [0, z],
                    linestyle='--', color='gray', alpha=0.5
                )
                # and similarly for y_max
                ax.plot(
                    [x_const, x_const], [y_max, y_max], [0, z],
                    linestyle='--', color='gray', alpha=0.5
                )

    # set limits so that all four fit in view
    # find global bounds from control points
    all_pts = np.vstack([np.array(cp_net).reshape(-1, 3)
                         for cp_net, *_ in edge_curves_3d])
    x_min, y_min, z_min = all_pts.min(axis=0)
    x_max, y_max, z_max = all_pts.max(axis=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("All Four Boundary Curves of the Coons Patch")
    ax.legend()
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()


def visualize_interpolated_surface(surface_points,
                                   edge_curves_3d,
                                   show_control_points=False):
    """
    Visualize the Coons patch given its interpolated surface grid and the four boundary curves.

    Parameters
    ----------
    surface_points : ndarray, shape (num_v, num_u, 3)
        The 3D grid of points on the surface.

    edge_curves_3d : list of length 4
        The four boundary curves in the order [D0, D1, C0, C1], each a tuple:
            (control_pts_3d, bspline_3d, section_bounds, t0, t1, z_tag)

    show_control_points : bool
        If True, scatter-plot the control points for each boundary curve.
    """
    num_v, num_u, _ = surface_points.shape
    X = surface_points[:, :, 0]
    Y = surface_points[:, :, 1]
    Z = surface_points[:, :, 2]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(X, Y, Z,
                    color='magenta',
                    alpha=0.7,
                    edgecolor='none')

    # Unpack four boundary curves
    D0, D1, C0, C1 = edge_curves_3d

    # Helper to plot an edge
    def plot_edge(curve, label, color_curve, color_cp):
        cp, bspline_3d, sec_bounds, t0, t1, z_tag = curve
        t_vals = np.linspace(t0, t1, num_u)
        pts = bspline_3d(t_vals)

        # Build label: include z_tag only if it's not None
        if z_tag is not None:
            full_label = f"{label} (z={z_tag:.2f})"
        else:
            full_label = label

        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                label=full_label,
                color=color_curve, linewidth=2)
        if show_control_points:
            cp_arr = np.array(cp)
            ax.scatter(cp_arr[:, 0], cp_arr[:, 1], cp_arr[:, 2],
                       color=color_cp, marker='o', s=50,
                       label=f"{label} CPs")

    # Plot bottom & top edges
    plot_edge(D0, "D0 (bottom)", "blue", "red")
    plot_edge(D1, "D1 (top)",    "magenta", "orange")

    # Plot left & right edges
    plot_edge(C0, "C0 (left)",   "cyan", "green")
    plot_edge(C1, "C1 (right)",  "orange", "purple")

    # Draw section boundaries for D0/D1 (horizontal edges):
    for curve in (D0, D1):
        _, _, sec_bounds, _, _, z_tag = curve
        if z_tag is None:
            continue
        # Without a stored plane‐height we skip drawing those vertical lines.
        # (You can later add if you know “h” for each horizontal plane.)

    # Draw section boundaries for C0/C1 (vertical edges):
    # Each (y_min, y_max) should be drawn at constant x, from z=0 → z_gap.
    for curve in (C0, C1):
        cp_edge, _, sec_bounds, _, _, _ = curve
        # Final control point’s z gives the top of this edge:
        z_val = cp_edge[-1][2]
        x_const = cp_edge[0][0]  # all CPs for C0/C1 share the same x
        for (y_min, y_max) in sec_bounds or []:
            ax.plot([x_const, x_const], [y_min, y_min], [0, z_val],
                    linestyle='--', color='gray', alpha=0.5)
            ax.plot([x_const, x_const], [y_max, y_max], [0, z_val],
                    linestyle='--', color='gray', alpha=0.5)

    # Set limits to encompass all four curves
    all_cp = np.vstack([np.array(curve[0]).reshape(-1, 3)
                        for curve in edge_curves_3d])
    x_min, y_min, z_min = all_cp.min(axis=0)
    x_max, y_max, z_max = all_cp.max(axis=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Interpolated Coons Patch with Boundary Curves")
    ax.legend()
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()






def visualize_dataset_entry_camera_view(entry, num_samples=50):
    """
    Example function that colors control net points based on whether they are "in front" or
    "behind" the camera viewpoint (approx. by dot product with the camera direction).
    """
    rotated_pts = entry["points"]
    rotated_cn = entry["control_net"]
    rot_angle = entry["rotation_angle"]
    z_gap = entry["z_gap"]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the noisy surface points (without color-by-camera for demonstration)
    pts_flat = rotated_pts.reshape(-1, 3)
    ax.scatter(pts_flat[:, 0], pts_flat[:, 1], pts_flat[:, 2],
               color='magenta', marker='.', s=20, label="Noisy Surface Points", zorder=5)

    # We'll do a first pass to set up the axis so that ax.azim, ax.elev reflect user preference
    # (In practice, you might set ax.view_init(elev=..., azim=...) or let the user rotate, then proceed.)
    #ax.view_init(elev=20, azim=30)  # example camera angles
    plt.draw()  # force Matplotlib to update the figure so ax.azim and ax.elev become valid

    # Now we can color the control net points based on the camera direction
    try:
        size_u, size_v, _ = rotated_cn.shape
    except Exception:
        size_u, size_v = 0, 0
    flat_cn = rotated_cn.reshape(-1, 3)

    # This is the key step: color each point by whether it's "in front" or "behind"
    colors_cn = color_by_camera_view(ax, flat_cn)

    # Plot the control net points
    ax.scatter(flat_cn[:, 0], flat_cn[:, 1], flat_cn[:, 2],
               c=colors_cn, marker='o', s=50, label="Control Net Points", zorder=10)

    # Then, draw connecting lines in the original grid order (not depth-sorted)
    for i in range(size_u):
        ax.plot(rotated_cn[i, :, 0], rotated_cn[i, :, 1], rotated_cn[i, :, 2],
                color='black', linestyle='--', linewidth=2, label="Control Net" if i == 0 else "", zorder=11)
    for j in range(size_v):
        ax.plot(rotated_cn[:, j, 0], rotated_cn[:, j, 1], rotated_cn[:, j, 2],
                color='black', linestyle='--', linewidth=2, zorder=11)

    ax.set_title(f"Dataset Entry Visualization (Camera-based Color)\n"
                 f"Rotation Angle: {rot_angle:.2f} rad, Z-Gap: {z_gap:.2f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    set_axes_equal(ax)
    plt.show()

def visualize_nurbs_surface_with_control_net(surf, num_samples_u=50, num_samples_v=50):
    """
    Evaluate and visualize a geomdl NURBS surface with its control net (control point polygon).
    Control net points that are behind the surface (i.e. have a lower z-value than the median of the surface)
    are shown in a different color.
    """
    # Compute parameter ranges from the knot vectors
    u_start = surf.knotvector_u[surf.degree_u]
    u_end = surf.knotvector_u[-(surf.degree_u + 1)]
    v_start = surf.knotvector_v[surf.degree_v]
    v_end = surf.knotvector_v[-(surf.degree_v + 1)]

    # Evenly sample the parameter domain
    u_vals = np.linspace(u_start, u_end, num_samples_u)
    v_vals = np.linspace(v_start, v_end, num_samples_v)

    pts = np.empty((num_samples_u, num_samples_v, 3))
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            pts[i, j, :] = np.array(surf.evaluate_single([u, v]))

    X = pts[:, :, 0]
    Y = pts[:, :, 1]
    Z = pts[:, :, 2]

    # Compute the median z-value from the surface samples as a heuristic threshold.
    median_z = np.median(Z)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    #surf_plot = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7, edgecolor='none')
    #fig.colorbar(surf_plot, shrink=0.5, aspect=10, label='Surface Height')
    surf_plot = ax.plot_surface(X, Y, Z, color='magenta', alpha=0.7, edgecolor='none')

    # Extract and reshape control net.
    try:
        size_u, size_v = surf.ctrlpts_size
    except TypeError:
        size_u = surf.ctrlpts_size_u
        size_v = surf.ctrlpts_size_v
    ctrlpts_arr = np.array(surf.ctrlpts).reshape((size_u, size_v, 3))

    # Flatten the control net for scatter plotting.
    flat_cn = ctrlpts_arr.reshape(-1, 3)
    # Assign colors: if a point's z is below the median, consider it "behind" (e.g. blue), else "in front" (e.g. red)
    colors = np.where(flat_cn[:, 2] < median_z, 'red', 'red')

    ax.scatter(flat_cn[:, 0], flat_cn[:, 1], flat_cn[:, 2],
               color=colors, marker='o', s=70, label='Control Net Points', zorder=10)

    # Draw connecting lines in the original grid order
    for i in range(size_u):
        ax.plot(ctrlpts_arr[i, :, 0], ctrlpts_arr[i, :, 1], ctrlpts_arr[i, :, 2],
                color='black', linestyle='--', linewidth=2, zorder=11)
    for j in range(size_v):
        ax.plot(ctrlpts_arr[:, j, 0], ctrlpts_arr[:, j, 1], ctrlpts_arr[:, j, 2],
                color='black', linestyle='--', linewidth=2, zorder=11)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("NURBS Surface with Control Net\n(Median z = {:.2f})".format(median_z))
    ax.legend()
    set_axes_equal(ax)
    plt.show()


def visualize_dataset_entry(entry, num_samples=50):
    """
    Visualize a single dataset entry (as generated by create_dataset_entry in the new Coons-patch pipeline).
    Colors control-net points based on depth relative to the current camera view:
      • “in front” (dot > 0) → red
      • “behind”  (dot < 0) → blue

    Args:
        entry (dict): A dataset entry containing:
                      - "points"       : np.ndarray of surface points (V×U×3), already rotated/noisy/normalized
                      - "control_net"  : np.ndarray of control-net points (m×n×3), rotated/normalized
                      - "rotation_angle": tuple (rx, ry, rz) of applied rotation angles (radians)
                      - "z_gap"        : float
        num_samples (int): Unused here (only kept for signature compatibility).
    """
    # Fetch from the new entry keys
    surface_pts = entry["points"]         # shape (V, U, 3)
    control_net = entry["control_net"]    # shape (m, n, 3)
    rot_angle   = entry["rotation_angle"]
    z_gap       = entry["z_gap"]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1) Plot the (flattened) noisy surface points
    pts_flat = surface_pts.reshape(-1, 3)
    ax.scatter(
        pts_flat[:, 0], pts_flat[:, 1], pts_flat[:, 2],
        color='magenta', marker='.', s=20,
        label="Noisy Surface Points", zorder=5
    )

    # 2) Prepare control-net for coloring by camera depth
    try:
        size_u, size_v, _ = control_net.shape
    except Exception:
        size_u = size_v = 0
    flat_cn = control_net.reshape(-1, 3)

    # 3) Compute camera direction from current view
    cam_dir = get_camera_direction(ax)
    # 4) Compute centroid of control-net
    center_cn = flat_cn.mean(axis=0)
    # 5) For each control-point, compute dot((pt - center), cam_dir)
    vecs = flat_cn - center_cn
    dots = np.dot(vecs, cam_dir)
    colors_cn = np.where(dots > 0, 'red', 'blue')

    # 6) Plot control-net points with depth-based coloring
    ax.scatter(
        flat_cn[:, 0], flat_cn[:, 1], flat_cn[:, 2],
        c=colors_cn, marker='o', s=50,
        label="Control Net Points", zorder=10
    )

    # 7) Draw connecting lines in the control-net grid order
    for i in range(size_u):
        ax.plot(
            control_net[i, :, 0],
            control_net[i, :, 1],
            control_net[i, :, 2],
            color='black', linestyle='--', linewidth=2,
            label="Control Net" if i == 0 else "", zorder=11
        )
    for j in range(size_v):
        ax.plot(
            control_net[:, j, 0],
            control_net[:, j, 1],
            control_net[:, j, 2],
            color='black', linestyle='--', linewidth=2,
            zorder=11
        )

    # 8) Title with rotation angles and z_gap
    if isinstance(rot_angle, tuple):
        rot_str = ", ".join(f"{a:.2f}" for a in rot_angle)
    else:
        rot_str = f"{rot_angle:.2f}"
    ax.set_title(
        f"Dataset Entry (Camera-Based Coloring)\n"
        f"Rotation Angles: {rot_str} rad, Z-Gap: {z_gap:.2f}"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    set_axes_equal(ax)
    plt.show()



def visualize_dataset_entry_multiview(entry, num_samples=50, output_prefix="multiview_dataset_entry"):
    """
    Create a 2x2 subplot showing four different viewpoints of a dataset entry.
    The four views correspond to different azimuth angles (e.g., front, right, back, left)
    with a fixed elevation.

    The figure is saved as both SVG and PNG in separate folders, with a unique 5-digit ID appended
    to the file names.

    Args:
        entry (dict): A dataset entry containing:
                      - "points": np.ndarray of surface points (V×U×3).
                      - "control_net": np.ndarray of control net points (m×n×3).
                      - "rotation_angle": The applied rotation angle (radians).
                      - "z_gap": The z-gap between the curves.
        num_samples (int): Number of samples for surface evaluation (unused here but kept for signature).
        output_prefix (str): Base name for the output files.
    """
    # Define four views: (elevation, azimuth)
    views = [(20, 0), (20, 90), (20, 180), (20, 270)]

    fig = plt.figure(figsize=(18, 12))

    for idx, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

        # Plot noisy surface points
        pts_flat = entry["points"].reshape(-1, 3)
        ax.scatter(
            pts_flat[:, 0], pts_flat[:, 1], pts_flat[:, 2],
            color='magenta', marker='.', s=20,
            label="Noisy Surface Points", zorder=5
        )

        # Plot control net points
        try:
            size_u, size_v, _ = entry["control_net"].shape
        except Exception:
            size_u = size_v = 0
        cn = entry["control_net"]
        ax.scatter(
            cn[:, :, 0], cn[:, :, 1], cn[:, :, 2],
            color='red', marker='o', s=50,
            label="Control Net Points", zorder=10
        )

        # Draw connecting lines (maintaining original grid order)
        for i in range(size_u):
            ax.plot(
                cn[i, :, 0], cn[i, :, 1], cn[i, :, 2],
                color='black', linestyle='--', linewidth=2, zorder=11
            )
        for j in range(size_v):
            ax.plot(
                cn[:, j, 0], cn[:, j, 1], cn[:, j, 2],
                color='black', linestyle='--', linewidth=2, zorder=11
            )

        # Set the view for this subplot
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"View: Elevation {elev}°, Azimuth {azim}°")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_equal(ax)
        ax.legend()

    rot_angle = entry["rotation_angle"]
    if isinstance(rot_angle, tuple):
        rot_angle_str = ", ".join(f"{angle:.2f}" for angle in rot_angle)
    else:
        rot_angle_str = f"{rot_angle:.2f}"
    fig.suptitle(
        f"Dataset Entry Multi-View Visualization\n"
        f"Rotation Angles: {rot_angle_str} rad, Z-Gap: {entry['z_gap']:.2f}",
        fontsize=16
    )

    # Generate a unique 5-digit ID
    unique_id = np.random.randint(10000, 100000)

    # Create output directories for SVG and PNG if they do not exist.
    svg_dir = "svgs"
    png_dir = "pngs"
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    svg_filename = os.path.join(svg_dir, f"{output_prefix}_{unique_id}.svg")
    png_filename = os.path.join(png_dir, f"{output_prefix}_{unique_id}.png")

    fig.savefig(svg_filename, format="svg")
    fig.savefig(png_filename, format="png")

    plt.show()


# For testing or interactive use, you can run this module directly.
if __name__ == '__main__':
    # Example: Use a saved dataset entry or generated one.
    pass
