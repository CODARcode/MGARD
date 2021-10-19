/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#include "cuda/CommonInternal.h"
 
#include "cuda/LevelwiseProcessingKernel.h"
#include "cuda/LevelwiseProcessingKernel.hpp"

namespace mgard_cuda {

#define KERNELS(D, T)                                                          \
  template void lwpk<D, T, COPY>(Handle<D, T> & handle, SIZE *shape_h,          \
                                 SIZE *shape_d, T *dv, SIZE *ldvs, T *dwork,     \
                                 SIZE *ldws, int queue_idx);\
  template class LwpkReo<Handle<D, T>, D, T, COPY, CUDA>;

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