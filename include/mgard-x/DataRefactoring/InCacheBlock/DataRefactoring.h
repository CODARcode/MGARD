/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_IN_CACHE_DATA_REFACTORING
#define MGARD_X_IN_CACHE_DATA_REFACTORING

// #include "Common.h"
#include "../../Hierarchy/Hierarchy.h"
#include "../../RuntimeX/RuntimeXPublic.h"

namespace mgard_x {

namespace data_refactoring {

namespace in_cache_block {

template <DIM D, typename T, typename DeviceType>
void decompose(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, SubArray<D, T, DeviceType> w,
               SubArray<D, T, DeviceType> b, int start_level, int stop_level,
               int queue_idx);

template <DIM D, typename T, typename DeviceType>
void recompose(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, SubArray<D, T, DeviceType> w,
               SubArray<D, T, DeviceType> b, int start_level, int stop_level,
               int queue_idx);

} // namespace in_cache_block

} // namespace data_refactoring

} // namespace mgard_x

#endif