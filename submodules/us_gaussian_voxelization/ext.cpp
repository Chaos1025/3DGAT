#include <torch/extension.h>
#include "voxelize_points.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voxelize_gaussians", &VoxelizeGaussiansCUDA);
  m.def("voxelize_gaussians_backward", &VoxelizeGaussiansBackwardCUDA);
}