import open3d as o3d
import numpy as np
import copy

pcd1 = o3d.io.read_point_cloud("../data/bun000.pcd")
pcd2 = o3d.io.read_point_cloud("../data/bun045.pcd")

# source point cloud
pcd_s = pcd1.voxel_down_sample(voxel_size=0.005)

#target point cloud
pcd_t = pcd2.voxel_down_sample(voxel_size=0.005)

pcd_s.paint_uniform_color([0., 1., 0.])
pcd_t.paint_uniform_color([0., 0., 1.])

threshold = 0.05
trans_init = np.identity(4)

# use Point-to-Point for objective func
#obj_func = o3d.pipelines.registration.TransformationEstimationPointToPoint()

#use Point-to-Plane for objective func
obj_func = o3d.pipelines.registration.TransformationEstimationPointToPlane()
result = o3d.pipelines.registration.registration_icp( pcd_s, pcd_t, threshold, trans_init, obj_func)

trans_reg = result.transformation
print(trans_reg)

pcd_reg = copy.deepcopy(pcd_s).transform(trans_reg)
pcd_reg.paint_uniform_color([1., 0., 0.])

#visualization
o3d.visualization.draw_geometries([pcd_reg, pcd_t])
