"""
nurbs_lib.py

A highly customizable Python library for:
  - Generating 2D B-Spline curves (via vertical sections),
  - Lifting them to 3D,
  - Interpolating a surface between two 3D curves,
  - Approximating a NURBS surface from the interpolated data using geomdl,
  - Uniformly sampling the NURBS surface (and extracting its control net),
  - Adding Gaussian noise to the sampled points,
  - Randomly rotating both the noisy sampled points and the control net in 3D.

This version also adds an optional visualization flag (show_visu) to display intermediate steps
via the visualization module (nurbs_vis).
"""

import os
import pickle
import numpy as np
from scipy.interpolate import BSpline
from geomdl import NURBS, utilities, fitting
from datetime import datetime


# Import the visualization library (ensure nurbs_vis is in your PYTHONPATH)
import nurbs_vis


def generate_control_points_in_vertical_sections(num_points, plane_width, plane_height):
    section_width = plane_width / num_points
    control_points = []
    section_bounds = []
    for i in range(num_points):
        x_min = i * section_width
        x_max = (i + 1) * section_width
        section_bounds.append((x_min, x_max))
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(0, plane_height)
        control_points.append((x, y))
    return control_points, section_bounds


def create_bspline_curve(control_points, degree=None, extrapolate=False):
    n_points = len(control_points)

    # Ensure we can have at least a degree-2 spline
    if n_points < 3:
        raise ValueError("At least 3 control points are required for degree ≥ 2 splines.")

    # Choose a random valid degree between 2 and n_points - 1
    if degree is None:
        max_degree = n_points - 1
        if max_degree < 2:
            raise ValueError("Too few control points for degree ≥ 2 B-spline.")
        elif max_degree == 2:
            degree = 2
        else:
            degree = np.random.randint(2, max_degree)

    x = np.array([pt[0] for pt in control_points])
    y = np.array([pt[1] for pt in control_points])

    # Generate an open uniform knot vector
    knots = np.concatenate((
        np.zeros(degree),
        np.linspace(0, 1, n_points - degree + 1),
        np.ones(degree)
    ))

    spline_x = BSpline(knots, x, degree, extrapolate=extrapolate)
    spline_y = BSpline(knots, y, degree, extrapolate=extrapolate)

    return (spline_x, spline_y)


def create_test_data_vertical_sections(
    plane_width_range,
    plane_height_range,
    num_points_range=(6, 6)
):
    """
    Generate test data consisting of exactly two 2D B-spline curves,
    each on its own randomly‐sized plane drawn from the given ranges.
    """
    data = []
    for i in range(2):
        # sample a fresh width & height for this curve
        w = np.random.uniform(*plane_width_range)
        h = np.random.uniform(*plane_height_range)

        chosen_num_points = np.random.randint(
            num_points_range[0], num_points_range[1] + 1
        )
        cp, section_bounds = generate_control_points_in_vertical_sections(
            chosen_num_points, w, h
        )
        bspline = create_bspline_curve(cp, degree=None, extrapolate=False)
        data.append(( cp, bspline, section_bounds, w, h))

    return data


def wrap_bspline_in_3d(bspline, z):
    spline_x, spline_y = bspline
    t_min = spline_x.t[spline_x.k]
    t_max = spline_x.t[-spline_x.k - 1]

    def bspline_3d(t):
        x = spline_x(t)
        y = spline_y(t)
        z_arr = np.full(np.shape(x), z)
        return np.column_stack((x, y, z_arr))

    return bspline_3d, t_min, t_max


def place_curves_in_z(curve1, curve2, z_gap):
    """
    Place the two 2D curves in 3D by lifting one of them along the z-axis using the given z_gap.
    """
    control_points1, bspline1, section_bounds1, _, _ = curve1
    control_points2, bspline2, section_bounds2,  _, _ = curve2

    cp1_3d = [(x, y, 0) for (x, y) in control_points1]
    cp2_3d = [(x, y, z_gap) for (x, y) in control_points2]

    bspline1_3d, tmin1, tmax1 = wrap_bspline_in_3d(bspline1, 0)
    bspline2_3d, tmin2, tmax2 = wrap_bspline_in_3d(bspline2, z_gap)

    new_curve1 = (cp1_3d, bspline1_3d, section_bounds1, tmin1, tmax1, 0)
    new_curve2 = (cp2_3d, bspline2_3d, section_bounds2, tmin2, tmax2, z_gap)
    return new_curve1, new_curve2, z_gap


def interpolate_surface_between_curves(curve1_3d, curve2_3d, num_u=161, num_v=20):
    _, bspline1_3d, _, tmin1, tmax1, _ = curve1_3d
    _, bspline2_3d, _, tmin2, tmax2, _ = curve2_3d
    s = np.linspace(0.0, 1.0, num_u)
    t1 = (1 - s) * tmin1 + s * tmax1
    t2 = (1 - s) * tmin2 + s * tmax2

    points1 = bspline1_3d(t1)
    points2 = bspline2_3d(t2)

    v_vals = np.linspace(0, 1, num_v)
    surface_points = np.empty((num_v, num_u, 3))
    for i, v in enumerate(v_vals):
        surface_points[i, :, :] = (1 - v) * points1 + v * points2
    return surface_points



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




def get_surface_control_net(surface_points, degree_u=3, degree_v=3, ctrlpts_size_u_range=(6, 6),
                            ctrlpts_size_v_range=(6, 6)):
    """
    Downsample the high-resolution surface grid to a grid of shape
    (ctrlpts_size_u, ctrlpts_size_v) and flatten it according to geomdl's expectations.
    The control net sizes are randomly selected within the provided ranges.

    Note: If you supply ctrlpts_size_u_range=(6, 6), the actual chosen size will be 6;
    then +1 is added to obtain the final control net dimension (i.e. 7), ensuring that
    approximate_surface() receives a flat list with the proper length.
    """
    chosen_ctrlpts_size_u = np.random.randint(ctrlpts_size_u_range[0], ctrlpts_size_u_range[1] + 1)
    chosen_ctrlpts_size_v = np.random.randint(ctrlpts_size_v_range[0], ctrlpts_size_v_range[1] + 1)


    #print("Chosen ctrlpts_size_u=", chosen_ctrlpts_size_u, " target_v=", chosen_ctrlpts_size_v)
    approx_surf, approx_cn = approximate_control_net(surface_points,degree_u,degree_v, chosen_ctrlpts_size_u, chosen_ctrlpts_size_v)

    return approx_surf


def rebuild_nurbs_surface_from_control_net(old_surf):
    try:
        size_u, size_v = old_surf.ctrlpts_size
    except TypeError:
        size_u = old_surf.ctrlpts_size_u
        size_v = old_surf.ctrlpts_size_v
    ctrlpts = old_surf.ctrlpts
    new_surf = NURBS.Surface()
    new_surf.degree_u = old_surf.degree_u
    new_surf.degree_v = old_surf.degree_v
    new_surf.ctrlpts_size_u = size_u
    new_surf.ctrlpts_size_v = size_v
    new_surf.ctrlpts = ctrlpts
    new_surf.knotvector_u = utilities.generate_knot_vector(new_surf.degree_u, size_u)
    new_surf.knotvector_v = utilities.generate_knot_vector(new_surf.degree_v, size_v)
    return new_surf


def sample_nurbs_surface_with_control_net(surf, num_samples_u=50, num_samples_v=50):
    u_start = surf.knotvector_u[surf.degree_u]
    u_end = surf.knotvector_u[-(surf.degree_u + 1)]
    v_start = surf.knotvector_v[surf.degree_v]
    v_end = surf.knotvector_v[-(surf.degree_v + 1)]
    u_vals = np.linspace(u_start, u_end, num_samples_u)
    v_vals = np.linspace(v_start, v_end, num_samples_v)
    pts = np.empty((num_samples_u, num_samples_v, 3))
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


def add_gaussian_noise(points, noise_std=0.1, noise_mean=0.0):
    noise = np.random.normal(loc=noise_mean, scale=noise_std, size=points.shape)
    return points + noise


def rotate_surface_randomly(pts, control_net, random_rotation=True):
    """
    If random_rotation is True, rotates the points and control net with random angles about all three axes.
    Rotation is performed about the centroid of the control net.

    Returns:
        rotated_pts, rotated_control_net, rotation_angles (tuple: (angle_x, angle_y, angle_z))
    If random_rotation is False, returns the original arrays and (0.0, 0.0, 0.0).
    """
    if not random_rotation:
        return pts, control_net, (0.0, 0.0, 0.0)

    angle_x, angle_y, angle_z = np.random.uniform(0, 2 * np.pi, size=3)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx
    centroid = np.mean(control_net.reshape(-1, 3), axis=0)
    original_shape_pts = pts.shape
    pts_flat = pts.reshape(-1, 3)
    rotated_pts_flat = (pts_flat - centroid) @ R.T + centroid
    rotated_pts = rotated_pts_flat.reshape(original_shape_pts)
    original_shape_cn = control_net.shape
    cn_flat = control_net.reshape(-1, 3)
    rotated_cn_flat = (cn_flat - centroid) @ R.T + centroid
    rotated_control_net = rotated_cn_flat.reshape(original_shape_cn)
    return rotated_pts, rotated_control_net, (angle_x, angle_y, angle_z)




def perturb_and_resample_entry(
    entry,
    u_index,
    v_index,
    new_position,
    num_samples_u=None,
    num_samples_v=None,
    degree_u=3,
    degree_v=3
):
    """
    Take an existing dataset entry, move one control point to a new position,
    rebuild a NURBS surface from that mutated net, then resample it.
    """
    # 1. Copy & mutate the control net
    cn = entry["rotated_control_net"]
    if cn.ndim == 4: cn = cn[0]
    mutated_cn = cn.copy()
    mutated_cn[u_index, v_index] = np.array(new_position, dtype=float)

    # 2. Build a new NURBS surface from the mutated control net
    surf = NURBS.Surface()
    surf.degree_u = degree_u
    surf.degree_v = degree_v
    u_count, v_count, _ = mutated_cn.shape
    surf.ctrlpts_size_u = u_count
    surf.ctrlpts_size_v = v_count
    surf.ctrlpts = mutated_cn.reshape(-1, 3).tolist()
    surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, u_count)
    surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, v_count)

    # 3. Determine sampling resolution
    orig_pts = entry["rotated_noisy_points"]
    nu, nv, _ = orig_pts.shape
    if num_samples_u is None: num_samples_u = nu
    if num_samples_v is None: num_samples_v = nv

    # 4. Uniformly sample the new surface
    pts, sampled_cn = sample_nurbs_surface_with_control_net(
        surf,
        num_samples_u=num_samples_u,
        num_samples_v=num_samples_v
    )

    # 5. Return a new entry dict
    return {
        "rotated_noisy_points": pts,
        "rotated_control_net": sampled_cn,
        "nurbs_surface": surf,
        "rotation_angle": (0.0, 0.0, 0.0),
        "z_gap": entry.get("z_gap", None)
    }



def normalize_surface_and_net(points: np.ndarray,
                              control_net: np.ndarray):
    """
    Shift + isotropically scale so that:
        • min corner → (0,0,0)
        • max extent along *any* axis → 1.0
    Returns
    -------
    norm_points      : np.ndarray  same shape as `points`
    norm_control_net : np.ndarray  same shape as `control_net`
    tfm              : dict        {'scale': float, 'translation': np.ndarray(3,)}
    """

    # 1)  find axis-aligned bounding box of the *union*
    all_pts = np.concatenate([points.reshape(-1, 3),
                              control_net.reshape(-1, 3)])
    p_min = all_pts.min(axis=0)
    p_max = all_pts.max(axis=0)

    # 2)  translation: bring min corner to origin
    translation = -p_min

    # 3)  uniform scale: longest side → 1
    extents = p_max - p_min
    scale = 1.0 / extents.max()

    def apply(arr):
        return (arr + translation) * scale

    norm_points      = apply(points)
    norm_control_net = apply(control_net)

    tfm = {'scale': scale, 'translation': translation}

    return norm_points, norm_control_net, tfm


def prepare_input_for_model(dataset_entry, target_shape=(100, 100, 3)):
    input_data = np.array(dataset_entry["rotated_noisy_points"], dtype=np.float32)
    if input_data.shape != target_shape:
        raise ValueError(f"Input data shape {input_data.shape} does not match the expected shape {target_shape}.")
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def create_dataset_entry(num_points_range=(6, 6),
                         plane_width_range=(100, 100),
                         plane_height_range=(120, 120),
                         num_samples=161,
                         num_v=30,
                         noise_std=0.05,
                         noise_mean=0.0,
                         degree_u=3,
                         degree_v=3,
                         ctrlpts_size_u_range=(6, 6),
                         ctrlpts_size_v_range=(6, 6),
                         z_gap_range=(40, 100),
                         bspline_degree=3,
                         extrapolate=False,
                         random_rotation=True,
                         normalise=True,
                         show_visu=False):
    """
    Full workflow function that creates a dataset entry consisting of:
      - A NURBS surface (rebuilt with uniform knot vectors),
      - Uniformly sampled surface points with added Gaussian noise,
      - The corresponding control net,
      - And both are rotated in 3D if random_rotation is True.

    The plane dimensions, the Z gap, the number of control points for curves, and the control net sizes
    are randomly sampled from their respective ranges.

    Returns:
        dict: {
            "rotated_noisy_points": np.ndarray,
            "rotated_control_net": np.ndarray,
            "nurbs_surface": Rebuilt NURBS surface,
            "rotation_angle": Rotation angles (tuple of three values in radians),
            "z_gap": z gap between curves
        }
    """
    if num_points_range is None:
        raise ValueError("num_points_range must be provided.")
    chosen_num_points = np.random.randint(num_points_range[0], num_points_range[1] + 1)
    #plane_width = np.random.uniform(*plane_width_range)
    #plane_height = np.random.uniform(*plane_height_range)
    test_data = create_test_data_vertical_sections(plane_width_range, plane_height_range,
                                                   num_points_range=num_points_range)
    curve2d_1 = test_data[0]
    curve2d_2 = test_data[1]

    plane1_size = (curve2d_1[3], curve2d_1[4])
    plane2_size = (curve2d_2[3], curve2d_2[4])

    if show_visu:
        nurbs_vis.visualize_curve_2d(curve2d_1[0], curve2d_1[1], curve2d_1[2], colors=('blue', 'red'))
        nurbs_vis.visualize_curve_2d(curve2d_2[0], curve2d_2[1], curve2d_2[2], colors=('magenta', 'orange'))
    chosen_z_gap = np.random.uniform(*z_gap_range)
    curve3d_1, curve3d_2, gap = place_curves_in_z(curve2d_1, curve2d_2, z_gap=chosen_z_gap)

    if show_visu:
        nurbs_vis.visualize_curves_3d(curve3d_1, curve3d_2, plane1_size, plane2_size, num_samples=num_samples)
    surface_points = interpolate_surface_between_curves(curve3d_1, curve3d_2, num_u=num_samples, num_v=num_v)

    if show_visu:
        nurbs_vis.visualize_interpolated_surface(surface_points, curve1_3d=curve3d_1, curve2_3d=curve3d_2,
                                                 show_control_points=True)
    nurbs_surf = get_surface_control_net(surface_points,
                                         degree_u=degree_u,
                                         degree_v=degree_v,
                                         ctrlpts_size_u_range=ctrlpts_size_u_range,
                                         ctrlpts_size_v_range=ctrlpts_size_v_range)
    nurbs_surf = rebuild_nurbs_surface_from_control_net(nurbs_surf)

    if show_visu:
        nurbs_vis.visualize_nurbs_surface_with_control_net(nurbs_surf, num_samples_u=50, num_samples_v=50)
    pts, control_net = sample_nurbs_surface_with_control_net(nurbs_surf,
                                                             num_samples_u=num_samples,
                                                             num_samples_v=num_samples)
    noisy_pts = add_gaussian_noise(pts, noise_std=noise_std, noise_mean=noise_mean)
    rotated_pts, rotated_control_net, used_angles = rotate_surface_randomly(
        noisy_pts, control_net, random_rotation=random_rotation
    )

    if normalise:
        rotated_pts, rotated_control_net, tfm = normalize_surface_and_net(rotated_pts, rotated_control_net)
    else:
        tfm = None


    dataset_entry = {
        "rotated_noisy_points": rotated_pts,
        "rotated_control_net": rotated_control_net,
        "nurbs_surface": nurbs_surf,
        "rotation_angle": used_angles,
        "z_gap": gap,
        "tfm": tfm
    }
    return dataset_entry

def create_and_save_dataset(num_entries=10, save_dir="dataset", filename_prefix="dataset", **dataset_params):
    os.makedirs(save_dir, exist_ok=True)
    dataset = []
    # Generate each dataset entry without using a seed.
    for i in range(num_entries):
        entry = create_dataset_entry(**dataset_params)
        dataset.append(entry)

    dataset_dict = {
        "data": dataset,
        "configuration_options": dataset_params.copy()
    }
    # Use the current date and time (ddmm_hhmm) in the filename.
    date_str = datetime.now().strftime("%d%m_%H%M")
    noise_std = dataset_params.get("noise_std", 0.05)
    filename = f"{filename_prefix}_{num_entries}_surfaces_noise{noise_std}_{date_str}.pkl"
    file_path = os.path.join(save_dir, filename)
    with open(file_path, "wb") as f:
        pickle.dump(dataset_dict, f)
    return file_path


if __name__ == '__main__':
    # Example usage with visualization enabled.
    entry = create_dataset_entry(
        show_visu=True,
        plane_width_range=(100, 150),
        plane_height_range=(120, 180),
        z_gap_range=(40, 100),
        random_rotation=True,
        num_points_range=(6, 6),
        ctrlpts_size_u_range=(6, 6),
        ctrlpts_size_v_range=(6, 6)
    )
    print("Dataset entry created:")
    print(f"Rotation angles (radians): {entry['rotation_angle']}")
    print(f"Z-gap: {entry['z_gap']}")
    print("Rotated noisy points shape:", entry["rotated_noisy_points"].shape)
    print("Rotated control net shape:", entry["rotated_control_net"].shape)

    dataset_file = create_and_save_dataset(
        num_entries=5,
        num_points_range=(6, 6),
        num_samples=161,
        num_v=20,
        noise_std=0.05,
        degree_u=3,
        degree_v=3,
        plane_width_range=(100, 150),
        plane_height_range=(120, 180),
        z_gap_range=(40, 100),
        random_rotation=True,
        ctrlpts_size_u_range=(6, 6),
        ctrlpts_size_v_range=(6, 6)
    )
    print(f"Dataset saved to: {dataset_file}")
