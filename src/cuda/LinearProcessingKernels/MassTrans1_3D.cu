/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/LinearProcessingKernel3D.h"
#include "cuda/LinearProcessingKernel3D.hpp"

namespace mgard_cuda {

#define KERNELS(D, T)                                                          \
  template void lpk_reo_1_3d<D, T>(                                            \
      Handle<D, T> & handle, int nr, int nc, int nf, int nf_c, int zero_r,     \
      int zero_c, int zero_f, T *ddist_f, T *dratio_f, T *dv1, int lddv11,     \
      int lddv12, T *dv2, int lddv21, int lddv22, T *dw, int lddw1, int lddw2, \
      int queue_idx, int config);

KERNELS(1, double)
KERNELS(1, float)
KERNELS(2, double)
KERNELS(2, float)
KERNELS(3, double)
KERNELS(3, float)
KERNELS(4, double)
KERNELS(4, float)
KERNELS(5, double)
KERNELS(5, float)

#undef KERNELS

} // namespace mgard_cuda