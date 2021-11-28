/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_COMPRESSION_WORKFLOW
#define MGARD_X_COMPRESSION_WORKFLOW


#include "RuntimeX/RuntimeXPublic.h"
#include "Handle.h"

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
Array<1, unsigned char, DeviceType> compress(Handle<D, T, DeviceType> &handle, Array<D, T, DeviceType> &in_array,
                                 enum error_bound_type type, T tol, T s);

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType> decompress(Handle<D, T, DeviceType> &handle,
                       Array<1, unsigned char, DeviceType> &compressed_array);

} // namespace mgard_x

#endif