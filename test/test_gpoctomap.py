import open3d as o3d
import numpy as np

from gpoctomap_py import GPOctoMap
from sogmm_py.utils import np_to_o3d

gpom = GPOctoMap()
gpom.set_resolution(0.02)

data = np.load('copier.npz')
pcld = np.array(data["arr_0"])
n_samples = np.shape(pcld)[0]

gpom.insert_color_pointcloud(pcld)