import numpy as np
from plyfile import PlyData, PlyElement

from scene.gaussian_model import BasicPointCloud


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    density = np.vstack([vertices["density"]]).T
    rotations = np.vstack(
        [vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]]
    ).T
    scale = np.vstack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]]).T
    return BasicPointCloud(points=positions, rots=rotations, scales=scale, density=density)
