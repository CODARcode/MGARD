/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGRAD_CUDA_DATA_REFACTORING_AMR
#define MGRAD_CUDA_DATA_REFACTORING_AMR

#include "CommonInternal.h"
#include "MemoryManagement.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace mgard_cuda {

template <uint32_t D, typename T>
void decompose_amr(Handle<D, T> &handle, T *dv, std::vector<int> ldvs,
               int l_target);

template <uint32_t D, typename T>
void recompose_amr(Handle<D, T> &handle, T *dv, std::vector<int> ldvs,
               int l_target);

} // namespace mgard_cuda

#endif