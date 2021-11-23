/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */
#include "cuda/CommonInternal.h"
 
#include "cuda/DataRefactoring/Correction/IterativeProcessingKernel3D.h"
#include "cuda/DataRefactoring/Correction/IterativeProcessingKernel3D.hpp"

namespace mgard_cuda {

#define KERNELS(D, T)                                                          \
  template void ipk_3_3d<D, T>(                                                \
      Handle<D, T> & handle, SIZE nr_c, SIZE nc_c, SIZE nf_c, T *am, T *bm,       \
      T *ddist_r, T *dv, SIZE lddv1, SIZE lddv2, int queue_idx, int config);\
  template class Ipk3Reo3D<D, T, CUDA>;

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