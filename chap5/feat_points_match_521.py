import open3d as o3d
import numpy as np
import copy

path = "../3rdparty/Open3D/examples/test_data/ICP/"
source = o3d.io.read_point_cloud(path+"cloud_bin_0.pcd")
target = o3d.io.read_point_cloud(path+"cloud_bin_1.pcd")

source.paint_uniform_color([0.5, 0.5, 1.])
target.paint_uniform_color([1., 0.5, 0.5])

initial_trans = np.identity(4)
initial_trans[0, 3] = -3.0

# sourceをtransformationで変換してから表示
def draw_registration_result( source, target, transformation ):
    pcds = list()
    for s in source:
        temp = copy.deepcopy(s)
        pcds.append(temp.transform(transformation))
    pcds += target
    o3d.visualization.draw_geometries(pcds, 
                                        zoom = 0.3199,
                                        front = [0.024, -0.225, -0.973],
                                        lookat = [0.488, 1.722, 1.556],
                                        up = [0.047, -0.972, 0.226])

def keypoint_and_feature_extraction( pcd, voxel_size ):
    keypoints = pcd.voxel_down_sample(voxel_size)
    viewpoint = np.array([0., 0., 0.], dtype="float64")
    radius_normal = 2.0*voxel_size

    # calcurate normals
    keypoints.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid( radius=radius_normal, max_nn=30))
    keypoints.orient_normals_towards_camera_location( viewpoint )

    # compute feature for keypoints
    radius_feature = 5. * voxel_size
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        keypoints, 
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100) )
    
    return keypoints, feature

voxel_size = 0.1
s_kp, s_feature = keypoint_and_feature_extraction( source, voxel_size )
t_kp, t_feature = keypoint_and_feature_extraction( target, voxel_size)

np_s_feature = s_feature.data.T
np_t_feature = t_feature.data.T

corrs = o3d.utility.Vector2iVector() # Convert int32 numpy array of shape (n, 2) to Open3D format.
threshold = 0.9

for i, feat in enumerate(np_s_feature):
    distance = np.linalg.norm( np_t_feature - feat, axis=1)
    nearest_idx = np.argmin(distance)
    dist_order = np.argsort(distance)
    ratio = distance[dist_order[0]] / distance[dist_order[1]]
    if ratio < threshold:
        corr = np.array( [[i], [nearest_idx]], np.int32 )
        corrs.append( corr )


def create_lineset_from_correspondences( corrs_set, pcd1, pcd2, transformation=np.identity(4) ):
    pcd1_tmp = copy.deepcopy(pcd1)
    pcd1_tmp.transform(transformation)
    
    corrs = np.asarray(corrs_set)
    
    np_points1 = np.array(pcd1_tmp.points)
    np_points2 = np.array(pcd2.points)

    points = list()
    lines = list()

    for i in range(corrs.shape[0]):
        # 対応点ペアを生成
        points.append( np_points1[corrs[i, 0]] )
        points.append( np_points2[corrs[i, 1]] )
        # 始点と終点のidリストを生成
        lines.append([2*i, 2*i+1])
    colors = [np.random.rand(3) for i in range(len(lines))]
    # liseを生成
    line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(points),
        lines = o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

s_kp.paint_uniform_color([0., 1., 0.])
t_kp.paint_uniform_color([0., 1., 0.])
#line_set = create_lineset_from_correspondences( corrs, s_kp, t_kp, initial_trans )
#draw_registration_result([source,s_kp], [target, t_kp, line_set],initial_trans)

# 全ての点を使って姿勢計算
#trans_ptp = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
#trans_all = trans_ptp.compute_transformation( s_kp, t_kp, corrs )
#draw_registration_result([source], [target], trans_all)

# RANSAC
distance_threshold = voxel_size*1.5
result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
    s_kp, t_kp, corrs,
    distance_threshold,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n = 3,
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ],
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
)
line_set = create_lineset_from_correspondences( result.correspondence_set, s_kp, t_kp, initial_trans )
#draw_registration_result([source, s_kp],[target,t_kp,line_set], initial_trans)

print("fitness: ", result.fitness)
#draw_registration_result([source],[target], result.transformation)

