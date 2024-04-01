import open3d as o3d
import numpy as np

# read point cloud and color gray
pcd = o3d.io.read_point_cloud("../data/bun000.pcd")
pcd.paint_uniform_color([0.5, 0.5, 0.5])

# make kd-tree
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

## knn -----------------------------------------
query = 10000
#np.asarray(pcd.colors)[query] = [1., 0., 0.]
pcd.colors[query] = [1., 0., 0.]
# extract neighbor k(=200) points 
[k, idx, d] = pcd_tree.search_knn_vector_3d(pcd.points[query], 200)
np.asarray(pcd.colors)[idx[1:], :] = [0., 0., 1.]

## radius search -------------------------------
query = 20000
pcd.colors[query] = [1., 0., 0.]
# extract points within r(=0.01)
[k, idx, d] = pcd_tree.search_radius_vector_3d(pcd.points[query], 0.01)
np.asarray(pcd.colors)[idx[1:], :] = [0., 1., 0.]


## hybrid search ------------------------------
query = 30000
pcd.colors[query] = [1., 0., 0.]
# extract neighbor max_nn(=200) points within radius(=0.01)
[k, idx, d] = pcd_tree.search_hybrid_vector_3d(pcd.points[query], radius=0.01, max_nn=200)
np.asarray(pcd.colors)[idx[1:], :] = [0, 1., 1.]

# visualization
o3d.visualization.draw_geometries([pcd])