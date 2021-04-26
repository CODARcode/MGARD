/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/LevelwiseProcessingKernel.h"
#include "cuda/LevelwiseProcessingKernel.hpp"

namespace mgard_cuda {

#define KERNELS(D, T)                                                          \
  template void lwpk<D, T, ADD>(                                               \
      Handle<D, T> & handle, thrust::device_vector<int> shape, T * dv,         \
      thrust::device_vector<int> ldvs, T * dwork,                              \
      thrust::device_vector<int> ldws, int queue_idx);

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

#define KERNELS(D, T)                                                          \
  template void lwpk<D, T, ADD>(Handle<D, T> & handle, int *shape_h,           \
                                int *shape_d, T *dv, int *ldvs, T *dwork,      \
                                int *ldws, int queue_idx);

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