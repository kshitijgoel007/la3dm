import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from gpoctomap_py import GPOctoMap
from sogmm_py.utils import np_to_o3d, o3d_to_np

gpom = GPOctoMap()
gpom.set_resolution(0.02)

# data = np.load('copier.npz')
# pcld = np.array(data["arr_0"])
pcld = o3d_to_np(o3d.io.read_point_cloud('test.pcd', format='pcd'))
n_samples = np.shape(pcld)[0]

ds_res = 0.1

gpom.insert_color_pointcloud(pcld, ds_res)

recon = gpom.get_pointcloud()

o3d.visualization.draw_geometries([np_to_o3d(recon)])