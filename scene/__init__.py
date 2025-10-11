#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
import os.path as osp
import torch
import pickle
from utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import fetchPly


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args,
        gaussians: GaussianModel,
        sample: torch.Tensor,
        load_iteration=None,
        init_from="pcd",
        ply_path="pcd",
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            if init_from[:6] == "random":
                n_point = int(init_from[7:])
                assert n_point > 0, "Specify valid number of random points"
                bbox = torch.tensor(args.bbox)
                pcd = bbox[0] + (bbox[1] - bbox[0]) * torch.rand(
                    [n_point, 3]
                )  # initial points position
                density = torch.rand([n_point, 1]) * 0.1  # initial points density
                point_cloud = torch.concat([pcd, density], dim=-1)
                init_from = "random"
            elif init_from[:6] == "measur":  # Init from measurements
                # Intensity as density
                # Binary mask by thresholding, to provide randow initial positions
                n_point = int(init_from[7:])
                assert n_point > 0, "Specify valid number of random points"
                bbox = torch.tensor(args.bbox)

                threshold = args.init_threshold
                mask = sample > threshold
                valid_points = torch.nonzero(mask, as_tuple=False)
                # random selection from binary mask
                assert valid_points.size(0) >= n_point, "Not enough valid points in mask"
                selected_indices = torch.randperm(valid_points.size(0))[:n_point]
                selected_points = valid_points[selected_indices]
                # 将选定的点转换为实际坐标
                pcd = bbox[0] + (bbox[1] - bbox[0]) * (
                    selected_points.float() / torch.tensor(sample.shape).float()
                )
                # ** random density now
                density = torch.rand([n_point, 1]) * 0.1
                # select density with sample value
                point_cloud = torch.concat([pcd, density], dim=-1)
                init_from = "measur"
            elif init_from[:6] == "unifrm":
                n_point = int(init_from[7:])
                assert n_point > 0, "Specify valid number of random points"
                bbox = torch.tensor(args.bbox)
                d, h, w = args.object_voxel_size
                length_per_point = np.cbrt(d * h * w / n_point)
                print(
                    f"{int(d/length_per_point)*int(h/length_per_point)*int(w/length_per_point)} Points for unifrom initialization."
                )
                x = torch.linspace(bbox[0, 0], bbox[1, 0], int(d / length_per_point))
                y = torch.linspace(bbox[0, 1], bbox[1, 1], int(h / length_per_point))
                z = torch.linspace(bbox[0, 2], bbox[1, 2], int(w / length_per_point))
                n_point = (
                    int(d / length_per_point)
                    * int(h / length_per_point)
                    * int(w / length_per_point)
                )
                # generate uniform point clouds with meshgrid
                xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
                pcd = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
                # random density
                density = torch.rand([n_point, 1]) * 0.1
                point_cloud = torch.cat([pcd, density], dim=-1)
                init_from = "unifrm"
            elif init_from == "pcd":
                point_cloud = fetchPly(ply_path)
                print(f"Initialize gaussians with pcd {ply_path}")
            elif init_from == "pickle":
                with open(ply_path[:-3] + "pickle", "rb") as handle:
                    point_cloud = pickle.load(handle)
            else:
                point_cloud = None

            self.gaussians.create_from_pcd(point_cloud, 1.0, init_from)

    def save(self, path, iteration, queryfunc, tv_vol_center, tv_vol_nVoxel, tv_vol_sVoxel, pipe):
        point_cloud_path = osp.join(path, "point_cloud/{}-it{}".format(pipe.output_file, iteration))
        self.gaussians.save_ply(osp.join(point_cloud_path, "point_cloud.ply"))

        query_pkg = queryfunc(
            self.gaussians,
            tv_vol_center,
            tv_vol_nVoxel,
            tv_vol_sVoxel,
            pipe,
        )
        vol_pred = query_pkg["vol"].clip(0.0, 1.0)
        np.save(osp.join(point_cloud_path, "vol_pred.npy"), vol_pred.detach().cpu().numpy())


def calculateImageDiff(image: torch.Tensor) -> torch.Tensor:
    """
    Calculate the diff of image along its all axis
    """
    grads = torch.gradient(image)
    grad = torch.stack(grads, dim=-1)
    grad = torch.norm(grad, dim=-1, p=2)
    grad = grad / grad.max()

    return grad
