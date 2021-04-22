/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/LinearQuantization.h"
#include "cuda/LinearQuantization.hpp"

namespace mgard_cuda {

#define KERNELS(T, D)                                                          \
  template void levelwise_linear_dequantize<T, D>(                             \
      Handle<T, D> & handle, int *shapes, int l_target, quant_meta<T> m,       \
      int *dv, int *ldvs, T *dwork, int *ldws, size_t outlier_count,           \
      unsigned int *outlier_idx, int *outliers, int queue_idx);

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