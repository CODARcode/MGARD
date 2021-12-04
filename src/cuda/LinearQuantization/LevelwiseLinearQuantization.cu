/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/CommonInternal.h"

#include "cuda/LinearQuantization.h"
#include "cuda/LinearQuantization.hpp"

namespace mgard_cuda {

#define KERNELS(D, T)                                                          \
  template void levelwise_linear_quantize<D, T>(                               \
      Handle<D, T> & handle, SIZE * shapes, SIZE l_target, T * volumes,        \
      SIZE ldvolumes, Metadata & m, T * dv, SIZE * ldvs,\ 
         int *dwork,                                                           \
      SIZE *ldws,\ 
         bool prep_huffmam,                                                    \
      SIZE *shape, LENGTH *outlier_count, LENGTH *outlier_idx,                 \
      QUANTIZED_INT *outliers, int queue_idx);

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