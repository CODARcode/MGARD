/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */
#include "cuda/CommonInternal.h"
 
#include "cuda/DataRefactoring/Correction/IterativeProcessingKernel3D_AMR.h"
#include "cuda/DataRefactoring/Correction/IterativeProcessingKernel3D_AMR.hpp"

namespace mgard_x {

#define KERNELS(D, T)                                                          \
  template void ipk_1_3d_amr<D, T>(                                                \
      Handle<D, T> & handle, int nr, int nc, int nf_c, T *am, T *bm,           \
      T *ddist_f, T *dv, int lddv1, int lddv2, \
      bool retrieve, int block_size,  \
      T * fv, int ldfv1, int ldfv2, T * bv, int ldbv1, int ldbv2, \
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

} // namespace mgard_x