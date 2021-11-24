/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */


#include "cuda/DataRefactoring.hpp"

#include <iostream>

#include <chrono>
namespace mgard_x {

#define KERNELS(D, T)                                        \
  template void decompose<D, T, CUDA>(Handle<D, T> & handle, \
                                      SubArray<D, T, CUDA>& v, \
                                      SIZE l_target, int queue_idx);  \
  template void recompose<D, T, CUDA>(Handle<D, T> & handle, \
                                      SubArray<D, T, CUDA>& v, \
                                      SIZE l_target, int queue_idx);

KERNELS(1, double)
KERNELS(1, float)
#undef KERNELS

} // namespace mgard_x
