import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from geomdl import fitting

# Default data directory - override as needed
data_dir_env = os.getenv('SUBDIV_DATA_DIR')
DATA_DIR = data_dir_env if data_dir_env else '/home/ainsworth/master/dataset_1000/'

# --- PLY Loader ---
def load_ply(idx, subdiv=False, data_dir=None):
    """
    Load a PLY file by index. If subdiv=True, loads the subdivided mesh.
    Returns (verts: np.ndarray, faces: List[List[int]]).
    """
    base = data_dir or DATA_DIR
    name = f"sample{idx:06d}" + ("_subdiv" if subdiv else "")
    path = os.path.join(base, name + '.ply')
    verts, faces = [], []
    with open(path, 'r') as f:
        assert f.readline().strip() == 'ply'
        assert 'ascii' in f.readline()
        nv = nf = 0
        while True:
            line = f.readline().strip()
            if line.startswith('element vertex'):
                nv = int(line.split()[-1])
            elif line.startswith('element face'):
                nf = int(line.split()[-1])
            elif line == 'end_header':
                break
        for _ in range(nv):
            verts.append(list(map(float, f.readline().split())))
        for _ in range(nf):
            vals = list(map(int, f.readline().split()))
            faces.append(vals[1:])
    return np.array(verts), faces

# --- Mesh Segmentation ---
def build_adj(faces):
    """Build face adjacency from list of faces."""
    edge2faces = {}
    for i, f in enumerate(faces):
        for k in range(len(f)):
            e = tuple(sorted((f[k], f[(k+1) % len(f)])))
            edge2faces.setdefault(e, []).append(i)
    adj = {i: set() for i in range(len(faces))}
    for flist in edge2faces.values():
        if len(flist) > 1:
            for a in flist:
                adj[a].update(flist)
    return adj

def segment_patches(faces, size=8):
    """Segment mesh into patches of up to `size` faces."""
    adj = build_adj(faces)
    unassigned = set(range(len(faces)))
    patches = []
    while unassigned:
        seed = unassigned.pop()
        comp = {seed}
        frontier = [seed]
        while frontier and len(comp) < size:
            u = frontier.pop(0)
            for nb in adj[u]:
                if nb in unassigned:
                    unassigned.remove(nb)
                    comp.add(nb)
                    frontier.append(nb)
                    if len(comp) >= size:
                        break
        patches.append(list(comp))
    return patches

# --- NURBS Fitting Helpers ---
def choose_ctrl_size(n_pts, degree=3):
    """Map number of samples to control-net size (min = degree+2)."""
    base = degree + 2
    if n_pts <= 150:
        return base
    if n_pts <= 300:
        return base + 1
    if n_pts <= 500:
        return base + 2
    if n_pts <= 800:
        return base + 3
    if n_pts <= 1200:
        return base + 4
    return base + 5


def get_surface_control_net(surface_points, du, dv, cu, cv):
    """
    Approximate a NURBS surface from a UV-sampled point grid.
    surface_points: (nu x nv x 3) array
    du, dv: degrees
    cu, cv: control-net sizes
    """
    sp = np.transpose(surface_points, (1, 0, 2))
    iu = np.linspace(0, sp.shape[0] - 1, cu, dtype=int)
    iv = np.linspace(0, sp.shape[1] - 1, cv, dtype=int)
    down = sp[np.ix_(iu, iv)]
    flat = [down[i, j].tolist() for i in range(cu) for j in range(cv)]
    return fitting.approximate_surface(
        flat,
        size_u=cu, size_v=cv,
        degree_u=du, degree_v=dv
    )


def sample_nurbs(surf, nu=100, nv=100):
    """Evaluate NURBS and extract a 100×100 control-net sampling by default."""
    u0, u1 = surf.knotvector_u[surf.degree_u], surf.knotvector_u[-(surf.degree_u + 1)]
    v0, v1 = surf.knotvector_v[surf.degree_v], surf.knotvector_v[-(surf.degree_v + 1)]
    us = np.linspace(u0, u1, nu)
    vs = np.linspace(v0, v1, nv)
    pts = [surf.evaluate_single((u, v)) for u in us for v in vs]
    pts = np.array(pts).reshape((nu, nv, 3))
    size = getattr(surf, 'ctrlpts_size', None)
    if isinstance(size, (list, tuple)):
        cu, cv = size
    else:
        cu, cv = surf.ctrlpts_size_u, surf.ctrlpts_size_v
    ctrl = np.array(surf.ctrlpts).reshape((cu, cv, 3))
    return pts, ctrl

# --- Conversion Routines ---
def convert_patch(v_ctrl, f_ctrl, v_sub, patch_faces, viz=False):
    """
    Convert a single control-patch to its NURBS fit.
    Returns (fit_grid, control_net) or (None,None) on failure.
    fit_grid is a (100,100,3) array.
    """
    v_idxs = sorted({vi for fi in patch_faces for vi in f_ctrl[fi]})
    vid = {old: i for i, old in enumerate(v_idxs)}
    v_patch = v_ctrl[v_idxs]
    f_patch = [[vid[vi] for vi in f_ctrl[fi]] for fi in patch_faces]

    mn, mx = v_patch.min(axis=0), v_patch.max(axis=0)
    sub_pts = v_sub[np.all((v_sub >= mn) & (v_sub <= mx), axis=1)]
    n_pts = len(sub_pts)
    if n_pts < 4:
        return None, None

    # PCA->UV parametrization
    X = sub_pts - sub_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    uv = X.dot(Vt[:2].T)
    uv = (uv - uv.min(axis=0)) / uv.ptp(axis=0)
    U, V = uv[:, 0], uv[:, 1]

    # UV grid sampling (12×12 anchor points)
    G = 12
    us = np.linspace(0, 1, G)
    vs = np.linspace(0, 1, G)
    Q = np.empty((G, G, 3))
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            idx_min = np.argmin((U - u)**2 + (V - v)**2)
            Q[i, j] = sub_pts[idx_min]

    # NURBS fit
    cu = choose_ctrl_size(n_pts)
    try:
        surf = get_surface_control_net(Q, 3, 3, cu, cu)
        fit_grid, ctrl_net = sample_nurbs(surf)  # now 100×100 grid
    except Exception:
        return None, None

    if viz:
        fig, ax = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'projection':'3d'})
        # (1) control patch
        ax[0].add_collection3d(Poly3DCollection([v_patch[f] for f in f_patch], alpha=0.3, edgecolor='k'))
        ax[0].scatter(*v_patch.T, c='r'); ax[0].set_title(f"Patch ({n_pts})")
        # (2) samples
        ax[1].scatter(*sub_pts.T, s=2, c='b'); ax[1].set_title("Samples")
        # (3) fit
        pts_flat = fit_grid.reshape((-1,3))
        ax[2].scatter(pts_flat[:,0], pts_flat[:,1], pts_flat[:,2], s=2, c='g')
        ax[2].set_title(f"NURBS Fit (100×100)")
        # (4) control net
        for i in range(ctrl_net.shape[0]):
            ax[3].plot(*ctrl_net[i,:,:].T, 'k-')
        for j in range(ctrl_net.shape[1]):
            ax[3].plot(*ctrl_net[:,j,:].T, 'k-')
        ax[3].scatter(*ctrl_net.reshape(-1,3).T, c='r'); ax[3].set_title("Control Net")
        plt.tight_layout(); plt.show()

    return fit_grid, ctrl_net


def convert_shape(idx, viz_patches=False, viz_shape=False):
    """
    Convert all patches of a shape. Returns lists of fit_grids (100×100×3) + nets.
    """
    v_ctrl, f_ctrl = load_ply(idx, subdiv=False)
    v_sub, _       = load_ply(idx, subdiv=True)
    patches        = segment_patches(f_ctrl, size=8)

    all_grids, all_nets = [], []
    for pf in patches:
        grid, net = convert_patch(v_ctrl, f_ctrl, v_sub, pf, viz=viz_patches)
        if grid is not None:
            all_grids.append(grid)
            all_nets.append(net)

    if viz_shape:
        fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection':'3d'})
        ax[0].add_collection3d(Poly3DCollection([v_ctrl[f] for f in f_ctrl], alpha=0.1, edgecolor='k'))
        ax[0].scatter(*v_ctrl.T, c='r', s=1)
        ax[0].scatter(*v_sub.T, c='b', s=1)
        ax[0].set_title("Full Shape")
        # overlay all 100×100 fits
        for grid in all_grids:
            pts_flat = grid.reshape((-1,3))
            ax[1].scatter(pts_flat[:,0], pts_flat[:,1], pts_flat[:,2], s=1, alpha=0.5)
        ax[1].set_title("All Fitted Surfaces (100×100)")
        # overlay nets
        for net in all_nets:
            for i in range(net.shape[0]):
                ax[2].plot(*net[i,:,:].T, 'k-', alpha=0.3)
            for j in range(net.shape[1]):
                ax[2].plot(*net[:,j,:].T, 'k-', alpha=0.3)
        ax[2].set_title("All Control Nets")
        plt.tight_layout(); plt.show()

    return all_grids, all_nets


def convert_all_shapes(shape_ids, output_dir, **config):
    """
    Batch convert shapes. Saves each as a pickle with:
      shape_id, configuration_options, fitted_grids, control_nets.
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx in shape_ids:
        grids, nets = convert_shape(idx, viz_patches=False, viz_shape=False)
        data = {
            'shape_id': idx,
            'configuration_options': config,
            'fitted_grids': grids,
            'control_nets': nets
        }
        out_path = os.path.join(output_dir, f'shape_{idx:06d}.pkl')
        with open(out_path, 'wb') as fp:
            pickle.dump(data, fp)

# Example entry-point:
if __name__ == '__main__':
    ids = list(range(1000))
    convert_all_shapes(ids, '/home/ainsworth/master/dataset_subdivision', patch_size=8,
                       degree_u=3, degree_v=3, uv_grid_size=12,
                       eval_grid=(100,100))
