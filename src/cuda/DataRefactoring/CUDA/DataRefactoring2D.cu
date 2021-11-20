/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */


#include "cuda/DataRefactoring.hpp"

#include <iostream>

#include <chrono>
namespace mgard_cuda {

#define KERNELS(D, T)                                        \
  template void decompose<D, T, CUDA>(Handle<D, T> & handle, \
                                      SubArray<D, T, CUDA>& v, \
                                      SIZE l_target, int queue_idx);  \
  template void recompose<D, T, CUDA>(Handle<D, T> & handle, \
                                      SubArray<D, T, CUDA>& v, \
                                      SIZE l_target, int queue_idx);

KERNELS(2, double)
KERNELS(2, float)
#undef KERNELS

} // namespace mgard_cuda
