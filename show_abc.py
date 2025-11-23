import trimesh
import matplotlib
matplotlib.use("Agg")  # ensures it runs on servers with no display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- CONFIG ---
NUM = 5
#mesh_path = "/home/ainsworth/master/abc/00000010/00000010_b4b99d35e04b4277931f9a9c_trimesh_000.obj"
mesh_path = "/home/ainsworth/master/abc/00000002/00000002_1ffb81a71e5b402e966b9341_trimesh_001.obj"


save_path = "/home/ainsworth/master/abc/model_full.png"

# --- LOAD MESH ---
mesh = trimesh.load(mesh_path, process=False)
V, F = mesh.vertices, mesh.faces

print(f"Loaded mesh: {len(V)} vertices, {len(F)} faces")

# --- PLOT ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")

# Build list of triangle vertex triplets
tris = [V[f] for f in F]
collection = Poly3DCollection(tris, alpha=0.9, facecolor="#cccccc", edgecolor="k", linewidths=0.1)
ax.add_collection3d(collection)

# Auto-scale the axes to the mesh size
mins, maxs = V.min(axis=0), V.max(axis=0)
center = (mins + maxs) / 2
max_range = (maxs - mins).max() / 2
ax.set_xlim(center[0] - max_range, center[0] + max_range)
ax.set_ylim(center[1] - max_range, center[1] + max_range)
ax.set_zlim(center[2] - max_range, center[2] + max_range)

# Tidy view
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=30, azim=45)
ax.set_title("Full Mesh Visualization")

plt.tight_layout()
plt.savefig(save_path, dpi=200)
plt.close(fig)

print(f"Saved full model visualization to {save_path}")
