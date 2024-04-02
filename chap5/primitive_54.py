import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("../data/tabletop_scene.ply")
plane_model, inliers = pcd.segment_plane(distance_threshold=0.005, ransac_n=3, num_iterations=500)

[a, b, c, d] = plane_model
print(f"Plane equation : {a:.2f}x + {b:.2f}y + {c:.2f} + {d:.2f} = 0")

plane_cloud = pcd.select_by_index(inliers)
plane_cloud.paint_uniform_color([1., 0., 0.])
outlier_cloud = pcd.select_by_index(inliers, invert=True)

#o3d.visualization.draw_geometries([plane_cloud, outlier_cloud])
o3d.visualization.draw_geometries([outlier_cloud])