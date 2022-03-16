/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_COMPRESSION_WORKFLOW_H
#define MGARD_X_COMPRESSION_WORKFLOW_H

#include "Hierarchy.h"
#include "RuntimeX/RuntimeXPublic.h"

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
Array<1, unsigned char, DeviceType>
compress(Hierarchy<D, T, DeviceType> &hierarchy,
         Array<D, T, DeviceType> &in_array, enum error_bound_type type, T tol,
         T s, T &norm, Config config);

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType>
decompress(Hierarchy<D, T, DeviceType> &hierarchy,
           Array<1, unsigned char, DeviceType> &compressed_array,
           enum error_bound_type type, T tol, T s, T norm, Config config);

} // namespace mgard_x

#endif