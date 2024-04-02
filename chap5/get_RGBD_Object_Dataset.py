import os

dirname = "rgbd-dataset"
classes = ["apple", "banana", "camera"]

url = "https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_pcd_ascii/"

for i in range(len(classes)):
    if not os.path.exists(dirname + "/" + classes[i]):
        os.system("wget " + url + classes[i] + "_1.tar")
        os.system("tar xvf " + classes[i] + "_1.tar")