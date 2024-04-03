import numpy as np
import open3d as o3d

def ComputeSphereCoefficient(p0, p1, p2, p3):
    A = np.array([  p0 - p3,
                    p1 - p3,
                    p2 - p3 ])
    p0_sq = np.dot(p0, p0)
    p1_sq = np.dot(p1, p1)
    p2_sq = np.dot(p2, p2)
    p3_sq = np.dot(p3, p3)
    b = np.array( [(p0_sq - p3_sq)/2.,
                    (p1_sq - p3_sq)/2.,
                    (p2_sq - p3_sq)/2.] )
    coeff = np.zeros(3)

    try:
        ans = np.linalg.solve(A,b)
    except:
        print("!!Error!! Matrix rank is ", np.linalg.matrix_rank(A))
        print(" Return", coeff)
    else:
        tmp = p0 - ans
        r = np.sqrt(np.dot(tmp,tmp))
        coeff = np.append(ans, r)
    return coeff

def EvaluateSphereCoefficient( pcd, coeff, distance_th = 0.01 ):
    # coeff = [a, b, c, r]
    # sphere (x-a)**2 + (y-b)**2 + (z-c)**2 = r**2
    fitness = 0 # モデルのあてはめの良さ
    inlier_dist = 0 # インライアの平均誤差
    inliers = None # インライア点群のインデックスリスト

    dist = np.abs( np.linalg.norm( pcd - coeff[:3], axis=1) - coeff[3] )
    print("dist", dist.shape)
    n_inlier = np.sum( dist<distance_th )

    if n_inlier != 0 :
        fitness = n_inlier / pcd.shape[0]
        inlier_dist = np.sum( (dist<distance_th) * dist ) / n_inlier
        inliers = np.where(dist<distance_th)[0]
    
    return fitness, inlier_dist, inliers


pcd = o3d.io.read_point_cloud("../data/tabletop_scene.ply")
plane_model, inliers = pcd.segment_plane(distance_threshold=0.005, ransac_n=3, num_iterations=500)

[a, b, c, d] = plane_model
print(f"Plane equation : {a:.2f}x + {b:.2f}y + {c:.2f} + {d:.2f} = 0")

plane_cloud = pcd.select_by_index(inliers)
plane_cloud.paint_uniform_color([1., 0., 0.])
outlier_cloud = pcd.select_by_index(inliers, invert=True)

pcd = outlier_cloud
np_pcd = np.asarray(pcd.points)
ransac_n = 4
num_iterations = 1000
distance_th = 0.005
max_radius = 0.05 # 検出する球の半径の最大値

# initialization
best_fitness = 0 
best_inlier_dist = 10000. # インライア点の平均距離
best_inliers = None # 元の点群におけるインライアのインデックス
best_coeff = np.zeros(4) # model par

for n in range(num_iterations):
    # サンプリング
    c_id = np.random.choice( np_pcd.shape[0], 4, replace=False)
    # モデルを作成
    coeff = ComputeSphereCoefficient( np_pcd[c_id[0]], np_pcd[c_id[1]] , np_pcd[c_id[2]], np_pcd[c_id[3]] )
    
    # 外れ値を除去（枝刈り処理）
    if max_radius < coeff[3]:
        continue
    
    # 評価
    fitness, inlier_dist, inliers = EvaluateSphereCoefficient( np_pcd, coeff, distance_th )
    if ( best_fitness < fitness ) or \
        ( best_fitness == fitness ) and ( inlier_dist < best_inlier_dist ):
        best_fitness = fitness
        best_inlier_dist = inlier_dist
        best_inliers = inliers
        print(inliers)
        best_coeff = coeff
        print(f"Update : \n fitness = {best_fitness:.4f}\n Inlier_dist = {inlier_dist:.4f}")

if best_coeff.any() != False:
    print("Sphere detected!")
    print(f"(x-{best_coeff[0]:.2f})^2 + (y-{best_coeff[1]:.2f})^2 + (z-{best_coeff[2]:.2f})^2 = {best_coeff[3]:.2f}^2")
else:
    print("No sphere detected.")

sphere_cloud = pcd.select_by_index(best_inliers)
sphere_cloud.paint_uniform_color([0., 0., 1.])
outlier_cloud = pcd.select_by_index(best_inliers, invert=True)
#o3d.visualization.draw_geometries([sphere_cloud, outlier_cloud])

mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=best_coeff[3])
mesh_sphere.compute_vertex_normals()
mesh_sphere.paint_uniform_color([0.3, 0.3, 0.7])
mesh_sphere.translate(best_coeff[:3])
#o3d.visualization.draw_geometries([mesh_sphere]+[sphere_cloud + plane_cloud + outlier_cloud])