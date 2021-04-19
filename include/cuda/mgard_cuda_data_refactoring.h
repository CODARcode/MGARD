/* 
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_DATA_REFACTORING
#define MGRAD_CUDA_DATA_REFACTORING

#include "mgard_cuda_common_internal.h"
#include "mgard_cuda_memory_management.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace mgard_cuda {

template <typename T, int D>
void refactor_reo(mgard_cuda_handle<T, D> &handle, T *dv, std::vector<int> ldvs, int l_target);

template <typename T, int D>
void recompose_reo(mgard_cuda_handle<T, D> &handle, T *dv, std::vector<int> ldvs, int l_target);

}

#endif