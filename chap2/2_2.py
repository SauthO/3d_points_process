import sys
import numpy as np
import open3d as o3d

filename = sys.argv[1]

print("Loading a point cloud from ", filename)
pcd = o3d.io.read_point_cloud(filename)
#pcd = o3d.io.read_triangle_mesh(filename)

print(pcd)
points = np.asarray(pcd.points)
print(points)
print(points.shape)

o3d.visualization.draw_geometries([pcd])