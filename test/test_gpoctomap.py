import argparse
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from termcolor import cprint

from gpoctomap_py import GPOctoMap
from sogmm_py.utils import np_to_o3d, read_log_trajectory, calculate_color_metrics, calculate_depth_metrics
from sogmm_py.vis_open3d import VisOpen3D
from sogmm_py import ImageUtils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--decimate', type=int)
    parser.add_argument('--frame', type=str)
    parser.add_argument('--case', type=str)
    args = parser.parse_args()

    d = args.decimate

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

    vis = VisOpen3D(visible=False)

    fr = args.frame
    case = args.case
    traj = read_log_trajectory(case + '-traj.log')
    gt_pose = traj[int(fr)].pose


    pcld_gt, im = iu.generate_pcld_wf(gt_pose, rgb_path='./color/' + fr + '.png',
                                    depth_path='./depth/' + fr + '.png',
                                    size=(W, H))

    gpom = GPOctoMap(0.02, 4, 1.0, 1.0, 0.01, 100, 0.001, 1000, 0.001, 0.3, 0.7)

    ds_res = 0.02
    gpom.insert_color_pointcloud(pcld_gt, gt_pose[:3, 3], ds_res)

    threed_recon = gpom.get_3d_recon()
    intensity_recon = gpom.get_intensity_recon(pcld_gt)

    _, gt_g = iu.pcld_wf_to_imgs(gt_pose, pcld_gt)
    _, pr_g = iu.pcld_wf_to_imgs(gt_pose, intensity_recon)

    psnr, ssim = calculate_color_metrics(gt_g, pr_g)
    
    f, p, re, rmean, rstd = calculate_depth_metrics(
        np_to_o3d(pcld_gt[:, :3]), np_to_o3d(threed_recon))

    cprint('case %s frame %s' % (case, fr), 'grey')
    cprint('psnr %f rmean %f' % (psnr, rmean), 'green')


    vis.add_geometry(np_to_o3d(intensity_recon))
    vis.update_view_point(O3D_K, np.linalg.inv(gt_pose))
    vis.update_renderer()
    vis.capture_screen_image(case + '_gpoctomap.png')