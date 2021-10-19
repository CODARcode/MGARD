/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGRAD_CUDA_COMPRESSION_WORKFLOW
#define MGRAD_CUDA_COMPRESSION_WORKFLOW

#include "Common.h"

namespace mgard_cuda {

template <DIM D, typename T, typename DeviceType>
Array<1, unsigned char, DeviceType> compress(Handle<D, T> &handle, Array<D, T, DeviceType> &in_array,
                                 enum error_bound_type type, T tol, T s);

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType> decompress(Handle<D, T> &handle,
                       Array<1, unsigned char, DeviceType> &compressed_array);

} // namespace mgard_cuda

#endif