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
import nurbs_vis_koons
import nurbsVisKoons


import numpy as np

def generate_control_points_in_sections(num_points,
                                        plane_width,
                                        plane_height,
                                        plane_depth,
                                        fixed_coord=None):
    """
    Create `num_points` random control points inside a 3D volume divided into sections
    along x (width) and y (depth), with a free z (height) coordinate.

    Parameters
    ----------
    num_points : int
        Number of control points to generate.
    plane_width : float
        The total extent along the x‐axis. This dimension is divided into `num_points`
        equal slices (sections), and each control point’s x is sampled from its section.
    plane_height : float
        The total extent along the z‐axis (height). Each control point’s z is sampled
        uniformly from [0, plane_height], unless `fixed_coord` is not None (then z is clamped).
    plane_depth : float
        The total extent along the y‐axis (depth). This dimension is also divided into
        `num_points` equal slices. However, each control point’s section index along y
        must be either the same as, or ±1 from, the previous point’s y‐section index.
    fixed_coord : float | None
        If not None, all z coordinates are clamped to this value (i.e., z = fixed_coord).

    Returns
    -------
    control_points : list[tuple[float, float, float]]
        A list of length `num_points`, where each element is an (x, y, z) tuple.
    x_section_bounds : list[tuple[float, float]]
        A list of length `num_points`, where each entry is the (low, high) bounds of the
        x‐section for the corresponding index i: (i * (plane_width/num_points), (i+1) * (plane_width/num_points)).
    y_section_bounds : list[tuple[float, float]]
        A list of length `num_points`, where each entry is the (low, high) bounds of the
        y‐sections. Note that although there are `num_points` possible y‐sections, the
        actual control point at index i may snap to a y‐section index that differs from i,
        subject to the “same or ±1” adjacency rule.
    """
    # Precompute section size along x and y
    x_section_size = plane_width / num_points
    y_section_size = plane_depth / num_points

    # Build the list of all possible section bounds along x and y
    x_section_bounds = [(i * x_section_size, (i + 1) * x_section_size) for i in range(num_points)]
    y_section_bounds = [(j * y_section_size, (j + 1) * y_section_size) for j in range(num_points)]

    control_points = []

    # We'll track the previous y‐section index to enforce the adjacency rule.
    prev_y_idx = None

    for i in range(num_points):
        # 1) Choose x uniformly within the i‐th x‐section
        x_lo, x_hi = x_section_bounds[i]
        x = np.random.uniform(x_lo, x_hi)

        # 2) Choose a valid y‐section index for this point
        if prev_y_idx is None:
            # For the very first point, pick any y‐section uniformly
            y_idx = np.random.randint(0, num_points)
        else:
            # Subsequent points: allowed indices are prev_y_idx - 1, prev_y_idx, prev_y_idx + 1
            candidates = [prev_y_idx]
            if prev_y_idx - 1 >= 0:
                candidates.append(prev_y_idx - 1)
            if prev_y_idx + 1 < num_points:
                candidates.append(prev_y_idx + 1)
            y_idx = np.random.choice(candidates)

        # Sample y uniformly within the chosen y‐section
        y_lo, y_hi = y_section_bounds[y_idx]
        y = np.random.uniform(y_lo, y_hi)

        # Update prev_y_idx for the next iteration
        prev_y_idx = y_idx

        # 3) Choose z uniformly from [0, plane_height], or clamp if fixed_coord is given
        if fixed_coord is None:
            z = np.random.uniform(0, plane_height)
        else:
            z = fixed_coord

        control_points.append((x, y, z))

    return control_points, x_section_bounds, y_section_bounds







def create_boundary_curves_for_coons(
        plane_width_range,
        plane_height_range,
        plane_depth_range,
        num_points_range=(6, 6)
):
    """
    Generate test data consisting of exactly four 3D B-spline curves, each on its own
    randomly-sized 3D “plane” (rectangular prism) drawn from the given ranges.

    Parameters
    ----------
    plane_width_range : tuple(float, float)
        Range from which to sample each curve’s X-extent.
    plane_height_range : tuple(float, float)
        Range from which to sample each curve’s Y-extent.
    plane_depth_range : tuple(float, float)
        Range from which to sample each curve’s Z-extent.
    num_points_range : tuple(int, int)
        Range from which to sample the number of control points for each edge curve.
        For example, (6, 6) means “always 6 points per edge.”

    Returns
    -------
    data : list of length 4
        Each element is a tuple:
          ( ctrlpts_3d, bspline_3d_callable, section_bounds, w, h, d )
        in the order [Curve1, Curve2, Curve3, Curve4].
        - ctrlpts_3d : list of (x, y, z) control points
        - bspline_3d_callable : (knots & coef data hidden inside a 3D BSpline)
        - section_bounds : list of ((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))
        - w, h, d      : the sampled plane_width, plane_height, plane_depth for that curve
    """
    from scipy.interpolate import BSpline

    def _make_3d_bspline(ctrlpts3d, degree=None, extrapolate=False):
        """
        Given a list of 3D control points, build a BSpline in each dimension.
        Returns (bspline_callable, t_min, t_max).
        """
        n_ctrl = len(ctrlpts3d)
        if n_ctrl < 2:
            raise ValueError("Need ≥2 control points for a valid BSpline.")

        pts = np.array(ctrlpts3d)  # shape = (n_ctrl, 3)
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        # Choose degree = min(requested, n_ctrl−1), default cubic if possible
        if degree is None:
            chosen_degree = min(3, n_ctrl - 1)
        else:
            chosen_degree = min(degree, n_ctrl - 1)

        # Build an open-uniform knot vector of length (n_ctrl + degree + 1)
        # (same knots for each coordinate; we’ll evaluate separately)
        knots = np.concatenate((
            np.zeros(chosen_degree),
            np.linspace(0.0, 1.0, n_ctrl - chosen_degree + 1),
            np.ones(chosen_degree)
        ))

        spline_x = BSpline(knots, x, chosen_degree, extrapolate=extrapolate)
        spline_y = BSpline(knots, y, chosen_degree, extrapolate=extrapolate)
        spline_z = BSpline(knots, z, chosen_degree, extrapolate=extrapolate)

        t_min = float(knots[chosen_degree])
        t_max = float(knots[-(chosen_degree + 1)])

        def bspline_3d(t_vals):
            """
            If t_vals is scalar, return a (1×3) array.  If array, returns (m×3).
            """
            tv = np.asarray(t_vals).ravel()
            x_out = spline_x(tv)
            y_out = spline_y(tv)
            z_out = spline_z(tv)
            return np.vstack((x_out, y_out, z_out)).T

        return bspline_3d, t_min, t_max

    data = []
    for i in range(4):
        # 1) Sample random width, height, depth for this curve
        w = np.random.uniform(*plane_width_range)
        h = np.random.uniform(*plane_height_range)
        d = np.random.uniform(*plane_depth_range)

        # 2) How many control points?
        chosen_num_points = np.random.randint(
            num_points_range[0], num_points_range[1] + 1
        )

        # 3) Generate 3D control points (and per-point section_bounds)
        cp3d, section_bounds, _ = generate_control_points_in_sections(
            chosen_num_points,
            w,    # plane_width
            h,    # plane_height
            d,    # plane_depth
            fixed_coord=None
        )

        # 4) Build a 3D BSpline from those points
        bspline3d, t0, t1 = _make_3d_bspline(cp3d, degree=None, extrapolate=False)

        # 5) Append to list: we keep (w,h,d) around for debugging or legend‐info
        data.append((cp3d, bspline3d, section_bounds, w, h, d))

    return data



from geomdl import NURBS, utilities

def _build_nurbs_curve_from_ctrlpts_3d(ctrlpts3d, degree=None):
    """
    Build a 3D geomdl.NURBS.Curve from a list of 3D control points.
    Returns (curve_obj, t_min, t_max).

    In modern geomdl, `curve.ctrlpts_size` is read-only: we just assign `curve.ctrlpts`.
    """
    n_ctrl = len(ctrlpts3d)
    if n_ctrl < 2:
        raise ValueError("Need at least 2 control points to build a BSpline curve.")

    # Choose a sensible degree (e.g. cubic or smaller if too few points)
    if degree is None:
        degree = min(3, n_ctrl - 1)
    else:
        degree = min(degree, n_ctrl - 1)

    curve = NURBS.Curve()
    curve.degree = degree

    # Assign the control points; geomdl infers ctrlpts_size from len(ctrlpts)
    curve.ctrlpts = [list(pt) for pt in ctrlpts3d]

    # Build a uniform open knot vector of length (n_ctrl + degree + 1)
    curve.knotvector = utilities.generate_knot_vector(curve.degree, n_ctrl)

    # The valid parameter range is [t_min, t_max]:
    t_min = curve.knotvector[curve.degree]
    t_max = curve.knotvector[-(curve.degree + 1)]
    return curve, t_min, t_max



# ──────────────────────────────────────────────────────────────────────────────
#  Helpers (unchanged or slightly adapted)
# ──────────────────────────────────────────────────────────────────────────────

def project_to_xz(ctrlpts3d):
    """
    Project a list of 2D points (x, y) into 3D as (x, 0, y).
    """
    return [(x, y, z) for (x, y, z) in ctrlpts3d]


def build_nurbs_and_wrapper(ctrlpts3d, degree):
    """
    Given a list of 3D control points, build a geomdl.NURBS.Curve and return:
      (ctrlpts3d, bspline_callable, t0, t1).

    - `ctrlpts3d` is simply echoed back for convenience.
    - `bspline_callable(t_vals)` returns an (N×3) numpy array when t_vals is an array,
      or a (1×3) array if t_vals is scalar.
    """
    n_ctrl = len(ctrlpts3d)
    if n_ctrl < 2:
        raise ValueError("Need at least 2 control points to build a NURBS curve.")

    curve = NURBS.Curve()
    curve.degree = min(degree, n_ctrl - 1)
    curve.ctrlpts = [list(pt) for pt in ctrlpts3d]
    curve.knotvector = utilities.generate_knot_vector(curve.degree, n_ctrl)

    t0 = float(curve.knotvector[curve.degree])
    t1 = float(curve.knotvector[-(curve.degree + 1)])

    def bspline3d(t_vals):
        if np.isscalar(t_vals):
            # evaluate_single returns a Python list of length 3
            return np.array(curve.evaluate_single(float(t_vals)))[None, :]
        else:
            arr = np.asarray(t_vals).ravel()
            pts = [curve.evaluate_single(float(u)) for u in arr]
            return np.array(pts)

    return ctrlpts3d, bspline3d, t0, t1


def axis_angle_alignment_matrix(v_from, v_to):
    """
    Return a 3×3 rotation matrix that rotates unit‐vector v_from → unit‐vector v_to
    by the minimal‐angle axis–angle transform.  Handles edge cases automatically.
    """
    norm_from = np.linalg.norm(v_from)
    norm_to   = np.linalg.norm(v_to)
    if norm_from < 1e-8 or norm_to < 1e-8:
        return np.eye(3)

    u = v_from / norm_from
    w = v_to   / norm_to
    dot = float(np.clip(np.dot(u, w), -1.0, 1.0))

    # Already aligned or opposite?
    if abs(dot - 1.0) < 1e-8:
        return np.eye(3)
    if abs(dot + 1.0) < 1e-8:
        # 180° rotation about any axis perpendicular to u
        ortho = np.array([1.0, 0.0, 0.0])
        if abs(u[0]) > 0.999:
            ortho = np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, ortho)
        axis /= np.linalg.norm(axis)
        K = np.array([[    0,    -axis[2],  axis[1]],
                      [ axis[2],     0,   -axis[0]],
                      [-axis[1], axis[0],     0   ]])
        # R = I + 2·K²  when θ = π
        return np.eye(3) + 2 * (K @ K)

    axis = np.cross(u, w)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(dot)

    K = np.array([[    0,    -axis[2],  axis[1]],
                  [ axis[2],     0,   -axis[0]],
                  [-axis[1], axis[0],     0   ]])
    R = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)
    return R


# ──────────────────────────────────────────────────────────────────────────────
# 1) Place Curve 1 (“bottom” edge)
# ──────────────────────────────────────────────────────────────────────────────

def place_curve1(cp2d, degree):
    """
    Curve 1 (bottom) lives on the X–Z plane (y in 2D → z in 3D):
      • Project (x,y) → (x, 0, z=y)
      • Build a NURBS curve without any rotation or translation.
    """
    ctrl1_3d = project_to_xz(cp2d)
    return build_nurbs_and_wrapper(ctrl1_3d, degree)


# ──────────────────────────────────────────────────────────────────────────────
# 2) Place Curve 2 (“top” edge) → rotate 90° about Z, then random ±yaw about Z
# ──────────────────────────────────────────────────────────────────────────────

def place_curve2(cp2d, reference_start3d, degree, angle_ranges_deg):
    """
    Curve 2 (“top”):
      • Project into X–Z.
      • Rotate exactly +90° about the Z-axis, so that what used to point along +X now points along +Y.
      • Then rotate by a random “±Δ” about Z, where Δ ∈ angle_ranges_deg['y'] (in degrees).
      • Finally translate so its “first control‐point” lands on reference_start3d.
    """
    # 2.1) Project the 2D prototype into X–Z
    proto2_3d = project_to_xz(cp2d)
    p_start   = np.array(proto2_3d[0])

    # 2.2) Build R_fixed = “+90° about Z”
    #      Rz( 90°) = [[ 0, -1, 0],
    #                  [ 1,  0, 0],
    #                  [ 0,  0, 1]]
    Rz_90 = np.array([[ 0.0, -1.0, 0.0],
                      [ 1.0,  0.0, 0.0],
                      [ 0.0,  0.0, 1.0]])

    # 2.3) Build a random “±Δ” yaw about Z, where Δ ∈ angle_ranges_deg['y']
    low_deg, high_deg = angle_ranges_deg['y']
    yaw_offset_deg = np.random.uniform(low_deg, high_deg)  # e.g. ±45°
    yaw_offset     = np.radians(yaw_offset_deg)
    cos_z, sin_z   = np.cos(yaw_offset), np.sin(yaw_offset)
    Rz_rand = np.array([[ cos_z, -sin_z, 0.0],
                        [ sin_z,  cos_z, 0.0],
                        [   0.0,    0.0,  1.0]])

    # 2.4) Combined rotation for Curve 2:
    R2 = Rz_rand @ Rz_90

    # 2.5) Translate so that (R2·p_start) → reference_start3d
    rot2_start = R2.dot(p_start)
    trans2     = np.array(reference_start3d) - rot2_start

    ctrl2_3d = []
    for p in proto2_3d:
        pr = R2.dot(np.array(p)) + trans2
        ctrl2_3d.append(tuple(pr.tolist()))

    return build_nurbs_and_wrapper(ctrl2_3d, degree)


# ──────────────────────────────────────────────────────────────────────────────
# 3) Place Curve 3 (“left” edge) → only random ±yaw about Z
# ──────────────────────────────────────────────────────────────────────────────

def place_curve3(cp2d, reference_start3d, degree, angle_ranges_deg):
    """
    Curve 3 (“left”):
      • Project into X–Z.
      • **Do NOT** do any fixed 90° rotation—assume that “left” already points roughly toward +Y.
      • Only rotate by a random “±Δ” about Z, where Δ ∈ angle_ranges_deg['y'].
      • Then translate so its “first control‐point” lands exactly at reference_start3d.
    """
    proto3_3d = project_to_xz(cp2d)
    p_start   = np.array(proto3_3d[0])

    # 3.2) Random yaw about Z
    low_deg, high_deg = angle_ranges_deg['y']
    yaw_offset_deg = np.random.uniform(low_deg, high_deg)
    yaw_offset     = np.radians(yaw_offset_deg)
    cos_z, sin_z   = np.cos(yaw_offset), np.sin(yaw_offset)
    Rz_rand = np.array([[ cos_z, -sin_z, 0.0],
                        [ sin_z,  cos_z, 0.0],
                        [   0.0,    0.0,  1.0]])

    # 3.3) Translate so that (Rz_rand·p_start) → reference_start3d
    rot3_start = Rz_rand.dot(p_start)
    trans3     = np.array(reference_start3d) - rot3_start

    ctrl3_3d = []
    for p in proto3_3d:
        pr = Rz_rand.dot(np.array(p)) + trans3
        ctrl3_3d.append(tuple(pr.tolist()))

    return build_nurbs_and_wrapper(ctrl3_3d, degree)


# ──────────────────────────────────────────────────────────────────────────────
# 4) Place Curve 4 (“right” edge) → scale + axis‐angle so its ends match exactly
# ──────────────────────────────────────────────────────────────────────────────

def place_curve4(cp2d, reference_start3d, reference_end3d, degree, angle_ranges_deg):
    """
    Curve 4 (“right”):
      • Project cp2d into X–Z.
      • Let v_proto  = (prototype_end − prototype_start).
      • Let v_target = (reference_end3d − reference_start3d).
      • Compute scale_factor = |v_target| / |v_proto|.
      • Compute R_align = axis_angle_alignment_matrix(v_proto, v_target).
      • For each prototype‐point p:
          p_scaled = prototype_start + scale_factor · (p − prototype_start)
          pr4      = R_align · p_scaled + trans4,
        where trans4 = (reference_start3d − R_align·prototype_start).
      • (No extra yaw here—so the final p_scaled_end will land exactly on reference_end3d.)
    """
    proto4_3d = project_to_xz(cp2d)
    p_start   = np.array(proto4_3d[0])
    p_end     = np.array(proto4_3d[-1])
    v_proto   = p_end - p_start

    p3_end    = np.array(reference_start3d)  # Curve 3’s end
    p1_end    = np.array(reference_end3d)    # Curve 1’s end
    v_target  = p1_end - p3_end

    norm_proto = np.linalg.norm(v_proto)
    norm_targ  = np.linalg.norm(v_target)
    if norm_proto < 1e-8 or norm_targ < 1e-8:
        scale_factor = 1.0
    else:
        scale_factor = norm_targ / norm_proto

    R_align = axis_angle_alignment_matrix(v_proto, v_target)

    rot4_start = R_align.dot(p_start)
    trans4     = p3_end - rot4_start

    ctrl4_3d = []
    for p in proto4_3d:
        p_arr    = np.array(p)
        p_scaled = p_start + scale_factor * (p_arr - p_start)
        pr4      = R_align.dot(p_scaled) + trans4
        ctrl4_3d.append(tuple(pr4.tolist()))

    return build_nurbs_and_wrapper(ctrl4_3d, degree)


# ──────────────────────────────────────────────────────────────────────────────
# ── Top‐level assembly function ────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def assemble_patch_edges_3d_from_four_prototypes(curves_2d,
                                                 angle_ranges_deg,
                                                 degree=3):
    """
    curves_2d = [
      (cp1_2d, bs1_2d, sec1, w1, h1),   # bottom  prototype
      (cp2_2d, bs2_2d, sec2, w2, h2),   # top     prototype
      (cp3_2d, bs3_2d, sec3, w3, h3),   # left    prototype
      (cp4_2d, bs4_2d, sec4, w4, h4)    # right   prototype
    ]

    angle_ranges_deg = {
      'x': (low_x_deg, high_x_deg),  # UNUSED here
      'y': (low_y_deg, high_y_deg),  # used for random ±rotation about Z on Curves 2 & 3
      'z': (low_z_deg, high_z_deg)   # UNUSED here
    }

    Returns a list of 4 tuples, in the order [bottom, top, left, right]:
      (ctrlpts3d, bspline_callable, section_bounds, t0, t1, None)
    """
    edges_3d = []

    # 1) Curve 1 (bottom)
    cp1_2d, _, sec1, _, _, _ = curves_2d[0]
    ctrl1_3d, bs1, t0_1, t1_1 = place_curve1(cp1_2d, degree)
    edges_3d.append((ctrl1_3d, bs1, sec1, t0_1, t1_1, None))

    # 2) Curve 2 (top)
    cp2_2d, _, sec2, _, _, _ = curves_2d[1]
    ctrl2_3d, bs2, t0_2, t1_2 = place_curve2(
        cp2_2d,
        reference_start3d=ctrl1_3d[0],
        degree=degree,
        angle_ranges_deg=angle_ranges_deg
    )
    edges_3d.append((ctrl2_3d, bs2, sec2, t0_2, t1_2, None))

    # 3) Curve 3 (left)
    cp3_2d, _, sec3, _, _, _ = curves_2d[2]
    ctrl3_3d, bs3, t0_3, t1_3 = place_curve3(
        cp3_2d,
        reference_start3d=ctrl2_3d[-1],
        degree=degree,
        angle_ranges_deg=angle_ranges_deg
    )
    edges_3d.append((ctrl3_3d, bs3, sec3, t0_3, t1_3, None))

    # 4) Curve 4 (right)
    cp4_2d, _, sec4, _, _, _ = curves_2d[3]
    ctrl4_3d, bs4, t0_4, t1_4 = place_curve4(
        cp4_2d,
        reference_start3d=ctrl3_3d[-1],
        reference_end3d=ctrl1_3d[-1],
        degree=degree,
        angle_ranges_deg=angle_ranges_deg
    )
    edges_3d.append((ctrl4_3d, bs4, sec4, t0_4, t1_4, None))

    return edges_3d

def make_bspline_3d(curve_obj):
    """
    Return a function f(t_vals) that evaluates `curve_obj` at every t in t_vals.
    If t_vals is a scalar, return a 1-D np.array of length 3.
    If t_vals is a 1-D array of length m, return an (m,3) array.
    """
    def f(t_vals):
        if np.isscalar(t_vals):
            # evaluate_single(t) returns a list[3], so convert to 1-D array
            return np.array(curve_obj.evaluate_single(float(t_vals)))
        else:
            arr = np.asarray(t_vals).ravel()               # flatten to 1-D
            pts = [curve_obj.evaluate_single(float(u))
                   for u in arr]
            return np.array(pts)  # shape = (len(arr), 3)
    return f





# ------------------------------------------------------------------
#  Coons-patch interpolation (four edge curves)
# ------------------------------------------------------------------
import numpy as np

def interpolate_coons_patch(edges_3d, num_u: int = 50, num_v: int = 50) -> np.ndarray:
    """
    Given four 3D boundary curves, build a Coons patch that interpolates between them.

    Parameters
    ----------
    edges_3d : list of length 4
        Each element is a tuple:
            (ctrlpts_3d, bspline_3d_callable, section_bounds, t0, t1, z_tag)
        in the order [D0 (bottom), D1 (top), C0 (left), C1 (right)].
        - 'ctrlpts_3d' is a list of 3D control‐points (unused here, but kept for consistency).
        - 'bspline_3d_callable' is a function f(t_scalar) → np.ndarray shape (3,) that evaluates the
          NURBS curve at parameter t (in [t0, t1]).
        - 'section_bounds' is a list of (low, high) intervals in the divided dimension (unused here).
        - 't0', 't1' are the valid start/end parameter values for that curve.
        - 'z_tag' is an optional float tag (unused here).
    num_u : int
        Number of sample points along the “u” (horizontal) direction—i.e. how finely to sample between
        D0(u) and D1(u).
    num_v : int
        Number of sample points along the “v” (vertical) direction—i.e. how finely to sample between
        C0(v) and C1(v).

    Returns
    -------
    surface_pts : np.ndarray, shape (num_v, num_u, 3)
        A grid of points (u varying fastest) on the Coons patch that smoothly interpolates all four edges.
    """



    # --------------------------
    # 1) Unpack the four edges
    # --------------------------
    # Edge 0 = D0 (bottom), Edge 1 = D1 (top), Edge 2 = C0 (left), Edge 3 = C1 (right)
    _, bs_D0, _, t0_D0, t1_D0, _ = edges_3d[0]
    _, bs_D1, _, t0_D1, t1_D1, _ = edges_3d[1]
    _, bs_C0, _, t0_C0, t1_C0, _ = edges_3d[2]
    _, bs_C1, _, t0_C1, t1_C1, _ = edges_3d[3]


    # --------------------------
    # 2) Define boundary functions D0(u), D1(u), C0(v), C1(v)
    # --------------------------
    # Each one maps a scalar u or v in [0,1] into a 3‐vector on that edge.

    def D0(u_scalar: float) -> np.ndarray:
        t = (1.0 - u_scalar) * t0_D0 + u_scalar * t1_D0
        return bs_D0(float(t))   # expect shape (3,)

    def D1(u_scalar: float) -> np.ndarray:
        t = (1.0 - u_scalar) * t0_D1 + u_scalar * t1_D1
        return bs_D1(float(t))   # expect shape (3,)

    def C0(v_scalar: float) -> np.ndarray:
        t = (1.0 - v_scalar) * t0_C0 + v_scalar * t1_C0
        return bs_C0(float(t))   # expect shape (3,)

    def C1(v_scalar: float) -> np.ndarray:
        t = (1.0 - v_scalar) * t0_C1 + v_scalar * t1_C1
        return bs_C1(float(t))   # expect shape (3,)

    # Corner points from D0/D1:
    p00 = bs_D0(t0_D0)  # bottom‐left
    p10 = bs_D0(t1_D0)  # bottom‐right
    p01 = bs_D1(t0_D1)  # top‐left
    p11 = bs_D1(t1_D1)  # top‐right

    #print("p00 (BL) =", p00)
    #print("p10 (BR) =", p10)
    #print("p01 (TL) =", p01)
    #print("p11 (TR) =", p11)

    # Now check that C0(v=0) == p00,  C0(v=1) == p01,
    # and              C1(v=0) == p10,  C1(v=1) == p11:
    #print("C0(0) =", bs_C0(t0_C0), "Should match p00")
    #print("C0(1) =", bs_C0(t1_C0), "Should match p01")
    #print("C1(0) =", bs_C1(t0_C1), "Should match p10")
    #print("C1(1) =", bs_C1(t1_C1), "Should match p11")

    # --------------------------
    # 3) Pre‐compute the four corner points
    #    p00 = D0(0,0),   p10 = D0(1,0)
    #    p01 = D1(0,1),   p11 = D1(1,1)
    # --------------------------
    p00 = D0(0.0)   # bottom-left corner
    p10 = D0(1.0)   # bottom-right corner
    p01 = D1(0.0)   # top-left corner
    p11 = D1(1.0)   # top-right corner

    # --------------------------
    # 4) Allocate the output array
    #    shape = (num_v, num_u, 3)
    # --------------------------
    surface_pts = np.zeros((num_v, num_u, 3), dtype=float)

    # Pre-build evenly spaced parameter vectors in [0,1]
    u_vals = np.linspace(0.0, 1.0, num_u)
    v_vals = np.linspace(0.0, 1.0, num_v)

    # --------------------------
    # 5) Fill the grid with the Coons‐patch formula
    #    For each (u,v) in [0,1]×[0,1]:
    #
    #      S(u,v) = (1−u)*C0(v) + u*C1(v)    [“vertical blending”]
    #             + (1−v)*D0(u) + v*D1(u)    [“horizontal blending”]
    #             − [bilinear corners]
    #
    #    where the bilinear term is:
    #      B(u,v) = (1−u)(1−v)*p00 + u(1−v)*p10 + (1−u)v*p01 + u v *p11
    # --------------------------
    for iv, v in enumerate(v_vals):
        Cv_left  = C0(v)   # shape (3,)
        Cv_right = C1(v)   # shape (3,)
        one_minus_v = 1.0 - v

        for iu, u in enumerate(u_vals):
            one_minus_u = 1.0 - u

            # Vertical‐edge blend
            blend_C = (one_minus_u * Cv_left) + (u * Cv_right)

            # Horizontal‐edge blend
            D0_at_u = D0(u)   # shape (3,)
            D1_at_u = D1(u)   # shape (3,)
            blend_D = (one_minus_v * D0_at_u) + (v * D1_at_u)

            # Bilinear corner correction
            bilinear = (
                (one_minus_u * one_minus_v) * p00
                + (u * one_minus_v)        * p10
                + (one_minus_u * v)        * p01
                + (u * v)                  * p11
            )

            # Final Coons patch point
            surface_pts[iv, iu, :] = blend_C + blend_D - bilinear

    return surface_pts






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
    approx_surf, approx_cn = approximate_control_net(surface_points,3,3, chosen_ctrlpts_size_u, chosen_ctrlpts_size_v)

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


# ----------------------------------------------------------------------
#  Normalisation utilities
# ----------------------------------------------------------------------
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


def resample_uniform_grid(surface_points: np.ndarray,
                          target_shape=(100, 100, 3)):
    """
    Bilinear resampling of a *normalized* surface grid to a fixed resolution
    so every sample fits the NN input tensor.

    surface_points : (V, U, 3)
    target_shape   : (V_target, U_target, 3)

    Returns
    -------
    resampled : np.ndarray   shape `target_shape`
    """
    V_src, U_src, _ = surface_points.shape
    V_tgt, U_tgt, _ = target_shape

    # create parameter lattices in 0…1
    u_src = np.linspace(0.0, 1.0, U_src)
    v_src = np.linspace(0.0, 1.0, V_src)
    u_tgt = np.linspace(0.0, 1.0, U_tgt)
    v_tgt = np.linspace(0.0, 1.0, V_tgt)

    # meshgrids for vectorised bilinear interpolation
    uu, vv = np.meshgrid(u_tgt, v_tgt)
    uu_src = uu * (U_src - 1)
    vv_src = vv * (V_src - 1)

    u0 = np.floor(uu_src).astype(int)
    v0 = np.floor(vv_src).astype(int)
    u1 = np.clip(u0 + 1, 0, U_src - 1)
    v1 = np.clip(v0 + 1, 0, V_src - 1)

    su = uu_src - u0     # fractional part
    sv = vv_src - v0

    # gather four neighbours and bilinearly blend
    p00 = surface_points[v0, u0]
    p10 = surface_points[v0, u1]
    p01 = surface_points[v1, u0]
    p11 = surface_points[v1, u1]

    resampled = ( (1 - su)[:, :, None] * (1 - sv)[:, :, None] * p00 +
                  su[:, :, None]       * (1 - sv)[:, :, None] * p10 +
                  (1 - su)[:, :, None] * sv[:, :, None]       * p01 +
                  su[:, :, None]       * sv[:, :, None]       * p11 )

    return resampled.astype(np.float32)


def prepare_input_for_model(dataset_entry, target_shape=(100, 100, 3)):
    input_data = np.array(dataset_entry["points"], dtype=np.float32)
    if input_data.shape != target_shape:
        raise ValueError(f"Input data shape {input_data.shape} does not match the expected shape {target_shape}.")
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def create_dataset_entry(
        # ---------- geometry / randomness ----------------------------
        num_points_range=(6, 6),
        plane_width_range=(100, 150),
        plane_height_range=(120, 180),
        plane_depth_range=(10, 30),
        num_samples=161,          # samples along u (≈ X)
        num_v=30,                 # samples along v (≈ Z)
        # ---------- NURBS fitting -------------------------------------
        degree_u=3, degree_v=3,
        ctrlpts_size_u_range=(6, 6),
        ctrlpts_size_v_range=(6, 6),
        # ---------- data augmentation ---------------------------------
        noise_std=0.05, noise_mean=0.0,
        random_rotation=True,
        # ---------- rotation ranges (NEW PARAM!) -----------------------
        angle_ranges_deg=None,
        # ---------- normalisation / NN input --------------------------
        normalize=True,
        resample=True,
        target_shape=(100, 100, 3),
        # ---------- misc ----------------------------------------------
        show_visu=False):
    """
    New end‐to‐end generator for one Coons‐patch dataset sample, using
    assemble_patch_edges_3d_from_four_prototypes(...) in place of the old
    place_boundary_curves_in_z(...) logic.

    Returns a dict with keys:
      'points'          : (V,U,3)   final (optionally resampled) surface grid
      'control_net'     : (m,n,3)   matching normalized control net
      'nurbs_surface'   : geomdl.NURBS.Surface (rebuilt, un‐normalized)
      'tfm'             : {'scale', 'translation'} for denormalizing
      'rotation_angle'  : (rx,ry,rz) in radians
      'z_gap'           : float
    """
    # ------------------------------------------------
    # 1) Build four 2D prototypes (X–Y) via your old function
    # ------------------------------------------------
    edges_3d = create_boundary_curves_for_coons(
        plane_width_range, plane_height_range, plane_depth_range, num_points_range
    )
    if show_visu:
        nurbsVisKoons.plot_boundary_curves(edges_3d, num_plot_points=300)

    # Each element of edges_2d is (cp_list, bspline2d, section_bounds, w, h)

    # ------------------------------------------------
    # 2) “Lift” into 3D using our new assemble_patch_edges_3d_from_four_prototypes
    # ------------------------------------------------
    # We pass degree=degree_u or degree_v (either works for a single‐dim curve)
    if angle_ranges_deg is None:
        angle_ranges_deg = {
            'x': (-45, 45),
            'y': (-45, 45),
            'z': (-10, 10)
        }

    edges_3d = assemble_patch_edges_3d_from_four_prototypes(
        edges_3d,
        angle_ranges_deg=angle_ranges_deg,
        degree=3
    )

    if show_visu:
        nurbsVisKoons.plot_curve_projections(edges_3d[0])
        nurbsVisKoons.plot_edges3d(edges_3d, num_plot_points=200)

    # Now edges_3d is a list of four tuples:
    # (ctrlpts3d, bspline3d_callable, section_bounds, t0, t1, z_tag)

    # ------------------------------------------------
    # 3) Form the Coons patch on those four 3D edges
    # ------------------------------------------------

    # Suppose “edges_3d” was built exactly in the code you showed:
    bottom_tuple = edges_3d[0]  # your “bottom” curve
    left_tuple = edges_3d[1]  # your “left” curve
    top_tuple = edges_3d[2]  # your “top” curve
    right_tuple = edges_3d[3]  # your “right” curve

    (ctrlpts4_3d, bspline4_3d, sec4, t0_orig, t1_orig, tag4) = right_tuple

    # Build a new “right” edge‐tuple with its parameters flipped:
    flipped_right = (ctrlpts4_3d, bspline4_3d, sec4, t1_orig, t0_orig, tag4)

    # Now permute into Coons’ expected ordering: [D0, D1, C0, C1]
    corrected_edges = [
        bottom_tuple,  # D0
        top_tuple,  # D1
        left_tuple,  # C0
        flipped_right  # C1
    ]

    surf_pts = interpolate_coons_patch(corrected_edges,
                                       num_u=num_samples,
                                       num_v=num_v)





    if show_visu:
        #nurbs_vis_koons.visualize_interpolated_surface(
        #    surf_pts,
         #   edge_curves_3d=edges_3d,
           # show_control_points=True
        #)
        nurbsVisKoons.plot_coons_surface_and_edges(surf_pts, edges_3d)

    # ------------------------------------------------
    # 4) NURBS‐fit the Coons grid
    # ------------------------------------------------
    nurbs_surf = get_surface_control_net(
        surf_pts,
        degree_u, degree_v,
        ctrlpts_size_u_range, ctrlpts_size_v_range
    )
    nurbs_surf = rebuild_nurbs_surface_from_control_net(nurbs_surf)

    pts, control_net = sample_nurbs_surface_with_control_net(
        nurbs_surf,
        num_samples_u=num_samples,
        num_samples_v=num_samples
    )

    # ------------------------------------------------
    # 5) Add noise + random rotate (as before)
    # ------------------------------------------------
    noisy = add_gaussian_noise(pts, noise_std, noise_mean)
    noisy, control_net, rot_angles = rotate_surface_randomly(
        noisy, control_net, random_rotation=random_rotation
    )

    # ------------------------------------------------
    # 6) Normalize (if requested)
    # ------------------------------------------------
    if normalize:
        noisy, control_net, tfm = normalize_surface_and_net(noisy, control_net)
    else:
        tfm = {'scale': 1.0, 'translation': np.zeros(3)}

    # ------------------------------------------------
    # 7) (Optional) Resample to fixed target_shape
    # ------------------------------------------------
    if resample and noisy.shape != target_shape:
        noisy = resample_uniform_grid(noisy, target_shape)

    if show_visu:
        temp_entry = {
            "points":         noisy,
            "control_net":    control_net,
            "rotation_angle": rot_angles,
            "z_gap":          None  # z_gap no longer explicitly used here
        }
        nurbsVisKoons.visualize_dataset_entry(temp_entry)

    return {
        "points":          noisy.astype(np.float32),
        "control_net":     control_net.astype(np.float32),
        "nurbs_surface":   nurbs_surf,
        "tfm":             tfm,
        "rotation_angle":  rot_angles,
        "z_gap":           None
    }



def create_and_save_dataset(
        num_entries       = 10,
        save_dir          = "dataset",
        filename_prefix   = "dataset",
        # everything else is forwarded verbatim to create_dataset_entry
        **entry_params):

    os.makedirs(save_dir, exist_ok=True)

    dataset = [create_dataset_entry(**entry_params)
               for _ in range(num_entries)]

    dataset_dict = {
        "data"                 : dataset,
        "configuration_options": entry_params.copy()
    }

    date_str   = datetime.now().strftime("%d%m_%H%M")
    noise_std  = entry_params.get("noise_std", 0.0)
    norm_flag  = "norm" if entry_params.get("normalize", True) else "raw"
    filename   = (f"{filename_prefix}_{num_entries}_coons_"
                  f"{norm_flag}_noise{noise_std}_{date_str}.pkl")
    file_path  = os.path.join(save_dir, filename)

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
        # geometry
        plane_width_range=(100, 150),
        plane_height_range=(120, 180),
        num_points_range=(6, 8),
        # surface grid
        num_samples=161,
        num_v=30,
        # control-net sizing
        ctrlpts_size_u_range=(6, 6),
        ctrlpts_size_v_range=(6, 6),
        # augmentation
        noise_std=0.05,
        random_rotation=True,
        # NN input
        normalize=True,
        resample=True,
        target_shape=(100, 100, 3),
        # visuals
        show_visu=False
    )
    print("saved to", dataset_file)

