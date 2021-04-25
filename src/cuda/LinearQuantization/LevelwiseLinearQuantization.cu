/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/LinearQuantization.h"
#include "cuda/LinearQuantization.hpp"

namespace mgard_cuda {

#define KERNELS(D, T)                                                          \
  template void levelwise_linear_quantize<D, T>(                               \
      Handle<D, T> & handle, int *shapes, int l_target, quant_meta<T> m,       \
      T *dv, int *ldvs,\ 
         int *dwork,                                                           \
      int *ldws,\ 
         bool prep_huffmam,                                                    \
      int *shape, size_t *outlier_count, unsigned int *outlier_idx,            \
      int *outliers, int queue_idx);

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