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

template <uint32_t D, typename T>
Array<1, unsigned char>
refactor_qz_cuda(Handle<D, T> &handle, Array<D, T> &in_array,
                 enum error_bound_type type, T tol, T s);

template <uint32_t D, typename T>
Array<D, T> recompose_udq_cuda(Handle<D, T> &handle,
                               Array<1, unsigned char> &compressed_array);

} // namespace mgard_cuda

#endif