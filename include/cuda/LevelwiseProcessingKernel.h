/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_LEVELWISE_PROCESSING_KERNEL
#define MGRAD_CUDA_LEVELWISE_PROCESSING_KERNEL

#include "Common.h"

namespace mgard_cuda {

template <DIM D, typename T, int OP>
void lwpk(Handle<D, T> &handle, thrust::device_vector<SIZE> shape, T *dv,
          thrust::device_vector<SIZE> ldvs, T *dwork,
          thrust::device_vector<SIZE> ldws, int queue_idx);

template <DIM D, typename T, int OP>
void lwpk(Handle<D, T> &handle, SIZE *shape_h, SIZE *shape_d, T *dv, SIZE *ldvs,
          T *dwork, SIZE *ldws, int queue_idx);

} // namespace mgard_cuda

#endif