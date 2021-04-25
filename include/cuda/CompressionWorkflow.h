/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_COMPRESSION_WORKFLOW
#define MGRAD_CUDA_COMPRESSION_WORKFLOW

#include "Common.h"

namespace mgard_cuda {

template <typename T, uint32_t D>
Array<unsigned char, 1>
refactor_qz_cuda(Handle<T, D> &handle, Array<T, D> &in_array,
                 enum error_bound_type type, T tol, T s);

template <typename T, uint32_t D>
Array<T, D> recompose_udq_cuda(Handle<T, D> &handle,
                               Array<unsigned char, 1> &compressed_array);

} // namespace mgard_cuda

#endif