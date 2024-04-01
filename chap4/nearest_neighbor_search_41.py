import open3d as o3d
import numpy as np

def dist(p, X):
    dists = np.linalg.norm(p-X, axis=1)
    return min(dists), np.argmin(dists)


# produce points by sin curve
X_x = np.arange(-np.pi, np.pi, 0.1)
X_y = np.sin(X_x)
X_z = np.zeros(X_x.shape)
X = np.vstack([X_x, X_y, X_z]).T

# define point p
p = np.array([1., 0., 0.])

# calcurate min distance and the index
min_dist, min_idx = dist(p, X)
print(f"distance:{min_dist:.2f}, idx:{min_idx}")
print(f"nearest neighbor:{X[min_idx]}")

# ready X for Open3d point cloud
pcd_X = o3d.geometry.PointCloud()
pcd_X.points = o3d.utility.Vector3dVector(X)
pcd_X.paint_uniform_color([0.5, 0.5, 0.5])
np.asarray(pcd_X.colors)[min_idx] = (0., 1., 0.)

# ready p for Open3d point cloud
pcd_p = o3d.geometry.PointCloud()
pcd_p.points = o3d.utility.Vector3dVector([p])
pcd_p.paint_uniform_color([0., 0., 1.])

# ready axis
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()


# visualization
o3d.visualization.draw_geometries([mesh, pcd_X, pcd_p])