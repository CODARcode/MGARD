/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#include "cuda/CommonInternal.h"
 
#include "cuda/LinearProcessingKernel3D.h"
#include "cuda/LinearProcessingKernel3D.hpp"

namespace mgard_cuda {

#define KERNELS(D, T)                                                          \
  template void lpk_reo_2_3d<D, T>(                                            \
      Handle<D, T> & handle, SIZE nr, SIZE nc, SIZE nf_c, SIZE nc_c, T *ddist_c,   \
      T *dratio_c, T *dv1, SIZE lddv11, SIZE lddv12, T *dv2, SIZE lddv21,         \
      SIZE lddv22, T *dw, SIZE lddw1, SIZE lddw2, int queue_idx, int config);\
  template class Lpk2Reo3D<Handle<D, T>, D, T, CUDA>;

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