import open3d as o3d
import numpy as np
import copy
import numpy.linalg as LA

def quaternion2rotation(q):
    rot = np.array([
                    [ q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 
                        2. * ( q[1]*q[2] - q[0]*q[3] ),
                        2. * ( q[1]*q[3] + q[0]*q[2] )],
                    [ 2. * ( q[1]*q[2] + q[0]*q[3] ),
                        q[0]**2 + q[2]**2 - q[1]**2 - q[3]**2,
                        2. * ( q[2]*q[3] - q[0]*q[1] )],
                    [ 2. * ( q[1]*q[3] - q[0]*q[2] ),
                        2. * ( q[2]*q[3] + q[0]*q[1] ),
                        q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2 ]
                    ])
    return rot

def GetCorrespondenceLines( pcd_s, pcd_t, idx_list ):

    # generate corresponding point pairs
    np_pcd_s = np.asarray(pcd_s.points)
    np_pcd_t = np.asarray(pcd_t.points)
    np_pcd_pair = np.concatenate((np_pcd_s, np_pcd_t))

    # generate id list for start and final point
    lines = list()
    n_points = len(pcd_s.points)
    for i in range(n_points):
        lines.append([i, n_points+idx_list[i]])
    
    # generate line set
    line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(np_pcd_pair),
        lines = o3d.utility.Vector2iVector(lines)
    )

    return line_set

q = np.array([1., 0., 0., 0., 0., 0., 0.])
rot = quaternion2rotation(q)
#print(rot)

pcd1 = o3d.io.read_point_cloud("../data/bun000.pcd")
pcd2 = o3d.io.read_point_cloud("../data/bun045.pcd")

pcd_s = pcd1.voxel_down_sample(voxel_size=0.005)
pcd_t = pcd2.voxel_down_sample(voxel_size=0.005)

# source
pcd_s.paint_uniform_color([0., 1., 0.])
# target
pcd_t.paint_uniform_color([0., 0., 1.])

## ----------------- Step1 : Matching source and target point clouds ------------------------

pcd_tree = o3d.geometry.KDTreeFlann(pcd_t)
idx_list = []

# search nearest neighbor points for each source points
for i in range(len(pcd_s.points)):
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd_s.points[i], 1)
    idx_list.append(idx[0])

# convert pcd.points to ndarray
np_pcd_s = np.asarray(pcd_s.points)
np_pcd_t = np.asarray(pcd_t.points)

# get the points in the target corresponding to the source 
np_pcd_y = np_pcd_t[idx_list].copy()

## ----------------- Step2 : Estimate rigid body transformation -------------------------

# get center of mass
mu_s = np_pcd_s.mean(axis=0)
mu_y = np_pcd_y.mean(axis=0)

# compute covariance matrix S_py
covar = np.zeros( (3, 3) )
n_points = np_pcd_s.shape[0]

for i in range(n_points):
    covar += np.dot( np_pcd_s[i].reshape(-1, 1), np_pcd_y[i].reshape(1, -1) )

covar /= n_points
covar -= np.dot( mu_s.reshape(-1, 1), mu_y.reshape(1, -1) )
print("covariance matrix S_py")
print(covar)
print()

# compute symmetry matrix N_py
A = covar - covar.T
delta = np.array([A[1,2], A[2,0], A[0,1]])
tr_covar = np.trace(covar)
i3d = np.identity(3)

N_py = np.zeros( (4,4) )
N_py[0,0] = tr_covar
N_py[0, 1:4] = delta.T
N_py[1:4, 0] = delta
N_py[1:4, 1:4] = covar + covar.T - tr_covar * i3d
print("symmetry matrix N_py")
print(N_py)
print()

# calcurate eigen value for N_py
w, v = LA.eig(N_py)
rot = quaternion2rotation(v[:, np.argmax(w)])
print("eigen value:\n", w)
print()
print("eigen vector:\n", v)
print()
print("eigen vector for maximum eigen value:\n", v[:, np.argmax(w)])
print()
print("rotation matrix:\n", rot)

# calcurate translation
trans = mu_y - np.dot(rot, mu_s)

# transform (4,4) matrix
transform = np.identity(4)
transform[0:3, 0:3] = rot.copy()
transform[0:3, 3] = trans.copy()
print("rigid body transformation matrix\n", transform)
print()

## ----------------- Step3 : Update object' posture -------------------------

# transform source point cloud by transformation matrix
pcd_s2 = copy.deepcopy(pcd_s)
pcd_s2.transform(transform)
pcd_s2.paint_uniform_color([1., 0., 0.])

# generate line set
line_set = GetCorrespondenceLines( pcd_s, pcd_t, idx_list )

# visualization
#o3d.visualization.draw_geometries([pcd_t, pcd_s, line_set])
o3d.visualization.draw_geometries([pcd_t, pcd_s, pcd_s2])