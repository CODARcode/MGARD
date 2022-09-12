/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_COMPRESSION_LOW_LEVEL_H
#define MGARD_X_COMPRESSION_LOW_LEVEL_H

#include "../Hierarchy/Hierarchy.hpp"
#include "../RuntimeX/RuntimeXPublic.h"
#include "CompressionLowLevelWorkspace.hpp"

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
Array<1, unsigned char, DeviceType>
compress(Hierarchy<D, T, DeviceType> &hierarchy,
         Array<D, T, DeviceType> &in_array, enum error_bound_type type, T tol,
         T s, T &norm, Config config,
         CompressionLowLevelWorkspace<D, T, DeviceType> &workspace);

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType>
decompress(Hierarchy<D, T, DeviceType> &hierarchy,
           Array<1, unsigned char, DeviceType> &compressed_array,
           enum error_bound_type type, T tol, T s, T norm, Config config,
           CompressionLowLevelWorkspace<D, T, DeviceType> &workspace);

} // namespace mgard_x

#endif