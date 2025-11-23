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
        curve1_3d,
        curve2_3d,
        plane1_size,
        plane2_size,
        num_samples=161
):
    """
    Visualize two 3D curves with their control points and vertical section boundaries.

    Parameters
    ----------
    curve1_3d : tuple
        (cp1, bspline1_3d, sec_bounds1, tmin1, tmax1, z1)
    curve2_3d : tuple
        (cp2, bspline2_3d, sec_bounds2, tmin2, tmax2, z2)
    plane1_size : tuple
        (plane1_width, plane1_height)
    plane2_size : tuple
        (plane2_width, plane2_height)
    num_samples : int
        Number of points to sample along each spline.
    """
    # Unpack
    cp1, bspline1, sec_bounds1, tmin1, tmax1, z1 = curve1_3d
    cp2, bspline2, sec_bounds2, tmin2, tmax2, z2 = curve2_3d
    w1, h1 = plane1_size
    w2, h2 = plane2_size

    # Sample curves
    t1 = np.linspace(tmin1, tmax1, num_samples)
    t2 = np.linspace(tmin2, tmax2, num_samples)
    pts1 = bspline1(t1)
    pts2 = bspline2(t2)

    # Prepare plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Curve 1
    ax.plot(pts1[:, 0], pts1[:, 1], pts1[:, 2],
            label=f"Curve 1 (z={z1:.2f})",
            color='blue')
    ax.scatter(*np.array(cp1).T,
               color='red',
               marker='o',
               s=50,
               label="CP 1")
    for x_min, x_max in sec_bounds1:
        ax.plot([x_min, x_min], [0, h1], [z1, z1],
                linestyle='--', color='green')
        ax.plot([x_max, x_max], [0, h1], [z1, z1],
                linestyle='--', color='green')

    # Curve 2
    ax.plot(pts2[:, 0], pts2[:, 1], pts2[:, 2],
            label=f"Curve 2 (z={z2:.2f})",
            color='magenta')
    ax.scatter(*np.array(cp2).T,
               color='orange',
               marker='o',
               s=50,
               label="CP 2")
    for x_min, x_max in sec_bounds2:
        ax.plot([x_min, x_min], [0, h2], [z2, z2],
                linestyle='--', color='green')
        ax.plot([x_max, x_max], [0, h2], [z2, z2],
                linestyle='--', color='green')

    # Global axis limits to cover both planes
    ax.set_xlim(0, max(w1, w2))
    ax.set_ylim(0, max(h1, h2))
    ax.set_zlim(min(z1, z2) - 0.1, max(z1, z2) + 0.1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Curves with Control Points and Vertical Sections")
    ax.legend()
    set_axes_equal(ax)

    plt.tight_layout()
    plt.show()


def visualize_interpolated_surface(surface_points, curve1_3d=None, curve2_3d=None, show_control_points=False):
    """
    Visualize the interpolated surface between two 3D curves.
    """
    num_v, num_u, _ = surface_points.shape
    X = surface_points[:, :, 0]
    Y = surface_points[:, :, 1]
    Z = surface_points[:, :, 2]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    #surf_plot = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7, edgecolor='none')
    #fig.colorbar(surf_plot, shrink=0.5, aspect=10, label='Surface Height')
    surf_plot = ax.plot_surface(X, Y, Z, color='magenta', alpha=0.7, edgecolor='none')


    if curve1_3d is not None and curve2_3d is not None:
        cp1, bspline1_3d, _, tmin1, tmax1, z1 = curve1_3d
        cp2, bspline2_3d, _, tmin2, tmax2, z2 = curve2_3d

        t_vals = np.linspace(tmin1, tmax1, num_u)
        curve1_pts = bspline1_3d(t_vals)
        curve2_pts = bspline2_3d(t_vals)

        ax.plot(curve1_pts[:, 0], curve1_pts[:, 1], curve1_pts[:, 2],
                label=f"Curve 1 (z = {z1})", color='blue', linewidth=2)
        ax.plot(curve2_pts[:, 0], curve2_pts[:, 1], curve2_pts[:, 2],
                label=f"Curve 2 (z = {z2})", color='magenta', linewidth=2)

        if show_control_points:
            ax.scatter(*np.array(cp1).T, color='red', marker='o', s=50, label="Curve 1 Control Points")
            ax.scatter(*np.array(cp2).T, color='orange', marker='o', s=50, label="Curve 2 Control Points")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Interpolated Surface Between 3D Curves")
    ax.legend()
    set_axes_equal(ax)
    plt.show()



def visualize_dataset_entry_camera_view(entry, num_samples=50):
    """
    Example function that colors control net points based on whether they are "in front" or
    "behind" the camera viewpoint (approx. by dot product with the camera direction).
    """
    rotated_pts = entry["rotated_noisy_points"]
    rotated_cn = entry["rotated_control_net"]
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
    Visualize a single dataset entry (as generated by create_dataset_entry and stored in a dataset).
    This version colors control net points based on their depth relative to the current camera view.

    Points that are "in front" (according to the dot product with the camera direction) are shown in red,
    while those "behind" are shown in blue.

    Args:
        entry (dict): A dataset entry containing:
                      - "rotated_noisy_points": np.ndarray of surface points.
                      - "rotated_control_net": np.ndarray of control net points.
                      - "rotation_angle": The applied rotation angle (radians).
                      - "z_gap": The z-gap between the two curves.
        num_samples (int): Number of samples for any surface evaluation (for labeling, etc.)
    """
    rotated_pts = entry["rotated_noisy_points"]
    rotated_cn = entry["rotated_control_net"]
    rot_angle = entry["rotation_angle"]
    z_gap = entry["z_gap"]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the noisy surface points
    pts_flat = rotated_pts.reshape(-1, 3)
    ax.scatter(pts_flat[:, 0], pts_flat[:, 1], pts_flat[:, 2],
               color='magenta', marker='.', s=20, label="Noisy Surface Points", zorder=5)

    # Set an initial view (you can adjust these angles or let the user set them interactively)
    #ax.view_init(elev=20, azim=30)
    #plt.draw()  # Update the view so that ax.azim and ax.elev reflect the set values

    # Get control net shape and flatten
    try:
        size_u, size_v, _ = rotated_cn.shape
    except Exception:
        size_u, size_v = 0, 0
    flat_cn = rotated_cn.reshape(-1, 3)

    # Compute the camera direction based on the current view
    cam_dir = get_camera_direction(ax)
    # Compute the centroid of control net points
    center_cn = flat_cn.mean(axis=0)
    # Compute vectors from the centroid to each control net point
    vecs = flat_cn - center_cn
    # Dot product with camera direction: if positive => in front, else behind
    dots = np.dot(vecs, cam_dir)
    # Assign colors: red for in front, blue for behind
    colors_cn = np.where(dots > 0, 'red', 'red')

    ax.scatter(flat_cn[:, 0], flat_cn[:, 1], flat_cn[:, 2],
               c=colors_cn, marker='o', s=50, label="Control Net Points", zorder=10)

    # Draw connecting lines in the original grid order to preserve net structure
    for i in range(size_u):
        ax.plot(rotated_cn[i, :, 0], rotated_cn[i, :, 1], rotated_cn[i, :, 2],
                color='black', linestyle='--', linewidth=2, label="Control Net" if i == 0 else "", zorder=11)
    for j in range(size_v):
        ax.plot(rotated_cn[:, j, 0], rotated_cn[:, j, 1], rotated_cn[:, j, 2],
                color='black', linestyle='--', linewidth=2, zorder=11)

    if isinstance(rot_angle, tuple):
        rot_angle_str = ", ".join(f"{angle:.2f}" for angle in rot_angle)
    else:
        rot_angle_str = f"{rot_angle:.2f}"
    ax.set_title(f"Dataset Entry Visualization (Camera-Based Coloring)\n"
                 f"Rotation Angles: {rot_angle_str} rad, Z-Gap: {z_gap:.2f}")

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
                      - "rotated_noisy_points": np.ndarray of surface points.
                      - "rotated_control_net": np.ndarray of control net points.
                      - "rotation_angle": The applied rotation angle (radians).
                      - "z_gap": The z-gap between the curves.
        num_samples (int): Number of samples for surface evaluation.
        output_prefix (str): Base name for the output files.
    """
    # Define four views: (elevation, azimuth)
    views = [(20, 0), (20, 90), (20, 180), (20, 270)]

    fig = plt.figure(figsize=(18, 12))

    for idx, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

        # Plot noisy surface points
        pts_flat = entry["rotated_noisy_points"].reshape(-1, 3)
        ax.scatter(pts_flat[:, 0], pts_flat[:, 1], pts_flat[:, 2],
                   color='magenta', marker='.', s=20, label="Noisy Surface Points", zorder=5)

        # Plot control net points
        try:
            size_u, size_v, _ = entry["rotated_control_net"].shape
        except Exception:
            size_u, size_v = 0, 0
        cn = entry["rotated_control_net"]
        ax.scatter(cn[:, :, 0], cn[:, :, 1], cn[:, :, 2],
                   color='red', marker='o', s=50, label="Control Net Points", zorder=10)

        # Draw connecting lines (maintaining original grid order)
        for i in range(size_u):
            ax.plot(cn[i, :, 0], cn[i, :, 1], cn[i, :, 2],
                    color='black', linestyle='--', linewidth=2, zorder=11)
        for j in range(size_v):
            ax.plot(cn[:, j, 0], cn[:, j, 1], cn[:, j, 2],
                    color='black', linestyle='--', linewidth=2, zorder=11)

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
        f"Dataset Entry Multi-View Visualization\nRotation Angles: {rot_angle_str} rad, Z-Gap: {entry['z_gap']:.2f}",
        fontsize=16)

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
