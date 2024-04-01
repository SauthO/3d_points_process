import numpy as np
import open3d as o3d
import copy

pcd1 = o3d.io.read_point_cloud("../data/bun000.pcd")
pcd2 = o3d.io.read_point_cloud("../data/bun045.pcd")

pcd_s = pcd1.voxel_down_sample(voxel_size=0.003)
pcd_t = pcd2.voxel_down_sample(voxel_size=0.003)

pcd_s.paint_uniform_color([0., 1., 0.])
pcd_t.paint_uniform_color([0., 0., 1.])


## ----------------- Step1 : Matching source and target point clouds ------------------------

# generate kd-tree for target poin cloud
pcd_tree = o3d.geometry.KDTreeFlann(pcd_t)

idx_list = []

for i in range(len(pcd_s.points)):
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd_s.points[i], 1)
    idx_list.append(idx[0])

np_pcd_s = np.asarray(pcd_s.points)
np_pcd_t = np.asarray(pcd_t.points)
# get corresponding target points with source points 
np_pcd_y = np_pcd_t[idx_list].copy()

# get normals
np_normal_t = np.asarray(pcd_t.normals)
np_normal_y = np_normal_t[idx_list].copy()

## ----------------- Step2 : Estimate rigid body transformation -------------------------

# compute matrix A, vector b
A = np.zeros((6,6))
b = np.zeros((6,1))

for i in range(len(np_pcd_s)):
    xn = np.cross( np_pcd_s[i], np_normal_y[i] ) # p X n_x
    xn_n = np.hstack( (xn, np_normal_y[i]) ).reshape(-1, 1) #[ [(p X n_x)],  
                                                            #  [(n_x)] ]
    nT = np_normal_y[i].reshape(1, -1)
    p_x = ( np_pcd_y[i] - np_pcd_s[i] ).reshape(-1, 1)

    A += np.dot(xn_n, xn_n.T)
    b += xn_n * np.dot(nT, p_x)
print("A : \n",A)
print("b : \n",b)

# compute rotation axis w and rotation angle theta by cross A inverse and b
u_opt = np.dot( np.linalg.inv(A), b ) # solution for minimize objective func
                                    # u_opt = (a  t)
# a = theta * w
theta = np.linalg.norm(u_opt[:3])
w = (u_opt[:3]/theta).reshape(-1)
#print(u_opt.shape)
#print(w.shape)
print("w : ", w)
print("theta : ", theta)

# calcurate rotation matrix
def axis_angle_to_matrix( axis, theta ):
    # skew symmetric matrix
    W = np.array( [
                    [  0., -axis[2], axis[1] ],
                    [ axis[2], 0., -axis[0] ],
                    [ -axis[1], axis[0], 0. ] ] )
    rot = np.identity(3) + theta * W

    return rot
rot = axis_angle_to_matrix(w, theta)

transform = np.identity(4)
transform[0:3, 0:3] = rot.copy()
transform[0:3, 3] = u_opt[3:6].reshape(-1).copy()

## ----------------- Step3 : Update object' posture -------------------------
pcd_s2 = copy.deepcopy(pcd_s)
pcd_s2.transform(transform)
pcd_s2.paint_uniform_color([1., 0., 0.])

# visualization
o3d.visualization.draw_geometries([pcd_s, pcd_t, pcd_s2])