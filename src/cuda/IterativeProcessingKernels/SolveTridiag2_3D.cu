/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */
#include "cuda/CommonInternal.h"
 
#include "cuda/DataRefactoring/Correction/IterativeProcessingKernel3D.h"
#include "cuda/DataRefactoring/Correction/IterativeProcessingKernel3D.hpp"

namespace mgard_x {

#define KERNELS(D, T)                                                          \
  template void ipk_2_3d<D, T>(                                                \
      Handle<D, T> & handle, SIZE nr, SIZE nc_c, SIZE nf_c, T *am, T *bm,         \
      T *ddist_c, T *dv, SIZE lddv1, SIZE lddv2, int queue_idx, int config);\
  template class Ipk2Reo3D<D, T, CUDA>;

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

} // namespace mgard_x