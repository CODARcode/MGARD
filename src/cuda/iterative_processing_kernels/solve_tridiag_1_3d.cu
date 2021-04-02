/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/iterative_processing_kernel_3d.h"
#include "cuda/iterative_processing_kernel_3d.hpp"

namespace mgard_cuda {

#define KERNELS(T, D)                                                          \
  template void ipk_1_3d<T, D>(mgard_cuda_handle<T, D> & handle, int nr,       \
                               int nc, int nf_c, T *am, T *bm, T *ddist_f,     \
                               T *dv, int lddv1, int lddv2, int queue_idx,     \
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