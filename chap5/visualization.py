import open3d as o3d

pcd_apple = o3d.io.read_point_cloud("rgbd-dataset/apple/apple_1/apple_1_1_100.pcd") 
pcd_banana = o3d.io.read_point_cloud("rgbd-dataset/banana/banana_1/banana_1_1_100.pcd")
pcd_camera = o3d.io.read_point_cloud("rgbd-dataset/camera/camera_1/camera_1_1_100.pcd")

o3d.visualization.draw_geometries([pcd_apple])
o3d.visualization.draw_geometries([pcd_banana])
o3d.visualization.draw_geometries([pcd_camera])