/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_LEVELWISE_PROCESSING_KERNEL
#define MGRAD_CUDA_LEVELWISE_PROCESSING_KERNEL

#include "Common.h"
#include "CommonInternal.h"

namespace mgard_cuda {

template <uint32_t D, typename T, int OP>
void lwpk(Handle<D, T> &handle, thrust::device_vector<int> shape, T *dv,
          thrust::device_vector<int> ldvs, T *dwork,
          thrust::device_vector<int> ldws, int queue_idx);

template <uint32_t D, typename T, int OP>
void lwpk(Handle<D, T> &handle, int *shape_h, int *shape_d, T *dv, int *ldvs,
          T *dwork, int *ldws, int queue_idx);

} // namespace mgard_cuda

#endif