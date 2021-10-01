/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#include "cuda/Testing/ReorderToolsCPU.h"
#include "cuda/Testing/ReorderToolsCPU.hpp"


namespace mgard {
    #define KERNELS(D, T)                                                          \
      template void ReorderCPU<D, T>(TensorMeshHierarchy<D, T> &hierarchy, T * input, T * output); \
      template void ReverseReorderCPU<D, T>(TensorMeshHierarchy<D, T> &hierarchy, T * input, T * output);

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
}