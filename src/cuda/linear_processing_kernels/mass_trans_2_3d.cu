/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/linear_processing_kernel_3d.h"
#include "cuda/linear_processing_kernel_3d.hpp"

namespace mgard_cuda {

#define KERNELS(T, D)                                                          \
  template void lpk_reo_2_3d<T, D>(                                            \
      mgard_cuda_handle<T, D> & handle, int nr, int nc, int nf_c, int nc_c,    \
      T *ddist_c, T *dratio_c, T *dv1, int lddv11, int lddv12, T *dv2,         \
      int lddv21, int lddv22, T *dw, int lddw1, int lddw2, int queue_idx,      \
      int config);

KERNELS(double, 1)
KERNELS(float, 1)
KERNELS(double, 2)
KERNELS(float, 2)
KERNELS(double, 3)
KERNELS(float, 3)
KERNELS(double, 4)
KERNELS(float, 4)
KERNELS(double, 5)
KERNELS(float, 5)

#undef KERNELS

} // namespace mgard_cuda