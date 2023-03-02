import argparse
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from gpoctomap_py import GPOctoMap
from sogmm_py.utils import np_to_o3d, read_log_trajectory, dir_path
from sogmm_py.vis_open3d import VisOpen3D
from sogmm_py import ImageUtils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--decimate', type=int)
    args = parser.parse_args()

    d = args.decimate

    traj = read_log_trajectory('copyroom-traj.log')
    gt_pose = traj[9].pose

    K = np.eye(3)
    K[0, 0] = 525.0/d
    K[1, 1] = 525.0/d
    K[0, 2] = 319.5/d
    K[1, 2] = 239.5/d

    iu = ImageUtils(K)
    W = (int)(640/d)
    H = (int)(480/d)

    O3D_K = np.array([[935.30743609,   0.,         959.5],
                      [0.,         935.30743609, 539.5],
                      [0.,           0.,           1.]])

    vis = VisOpen3D(visible=True)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0., 0., 0.])
    vis.add_geometry(coord_frame)

    pcld_gt, im = iu.generate_pcld_wf(gt_pose, rgb_path='./color/000009.png',
                                    depth_path='./depth/000009.png',
                                    size=(W, H))

    gpom = GPOctoMap(0.02, 4, 1.0, 1.0, 0.01, 100, 0.001, 1000, 0.001, 0.3, 0.7)

    ds_res = 0.02
    gpom.insert_color_pointcloud(pcld_gt, gt_pose[:3, 3], ds_res)
    recon = gpom.get_intensity_recon(pcld_gt)

    vis.add_geometry(np_to_o3d(recon))
    # vis.add_geometry(np_to_o3d(pcld_gt))

    vis.update_view_point(O3D_K, np.linalg.inv(gt_pose))
    vis.update_renderer()
    # vis.capture_screen_image(str(fr) + '_gt.png')
    vis.run()

    del vis
