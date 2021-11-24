/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */
#include "cuda/CommonInternal.h"
 
#include "cuda/DataRefactoring/Correction/LinearProcessingKernel.h"
#include "cuda/DataRefactoring/Correction/LinearProcessingKernel.hpp"

namespace mgard_x {

#define KERNELS(D, T)                                                          \
  template void lpk_reo_2<D, T>(                                               \
      Handle<D, T> & handle, SIZE *shape_h, SIZE *shape_c_h, SIZE *shape_d,       \
      SIZE *shape_c_d, SIZE *ldvs, SIZE *ldws, DIM processed_n,                   \
      DIM *processed_dims_h, DIM *processed_dims_d, DIM curr_dim_r,            \
      DIM curr_dim_c, DIM curr_dim_f, T *ddist_c, T *dratio_c, T *dv1,         \
      LENGTH lddv11, LENGTH lddv12, T *dv2, LENGTH lddv21, LENGTH lddv22, T *dw,           \
      LENGTH lddw1, LENGTH lddw2, int queue_idx, int config);

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