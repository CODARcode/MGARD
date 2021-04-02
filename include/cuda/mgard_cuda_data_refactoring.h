/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/mgard_cuda_helper.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace mgard_cuda {

template <typename T, int D>
void refactor_reo(mgard_cuda_handle<T, D> &handle, T *dv,
                  thrust::device_vector<int> lds, int l_target);

template <typename T, int D>
void recompose_reo(mgard_cuda_handle<T, D> &handle, T *dv,
                   thrust::device_vector<int> lds, int l_target);

} // namespace mgard_cuda