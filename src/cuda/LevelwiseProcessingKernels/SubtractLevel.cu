/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/LevelwiseProcessingKernel.h"
#include "cuda/LevelwiseProcessingKernel.hpp"

namespace mgard_cuda {

#define KERNELS(T, D)                                                          \
  template void lwpk<T, D, SUBTRACT>(                                          \
      Handle<T, D> & handle, thrust::device_vector<int> shape, T * dv,         \
      thrust::device_vector<int> ldvs, T * dwork,                              \
      thrust::device_vector<int> ldws, int queue_idx);

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

#define KERNELS(T, D)                                                          \
  template void lwpk<T, D, SUBTRACT>(Handle<T, D> & handle, int *shape_h,      \
                                     int *shape_d, T *dv, int *ldvs, T *dwork, \
                                     int *ldws, int queue_idx);

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