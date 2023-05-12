/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../Hierarchy/Hierarchy.h"
#include "../../RuntimeX/RuntimeX.h"

#include "Autocorrelation8x8x8.hpp"
#include "DataRefactoring.h"
#include "MultiDimension8x8x8.hpp"

#include <iostream>

#ifndef MGARD_X_IN_CACHE_BLOCK_DATA_REFACTORING_HPP
#define MGARD_X_IN_CACHE_BLOCK_DATA_REFACTORING_HPP

namespace mgard_x {

namespace data_refactoring {

namespace in_cache_block {

template <DIM D, typename T, typename DeviceType>
void decompose(SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> coarse,
               SubArray<1, T, DeviceType> coeff, int queue_idx) {
  if constexpr (D <= 3) {
    DeviceLauncher<DeviceType>::Execute(
        MultiDimension8x8x8Kernel<D, T, DECOMPOSE, DeviceType>(v, coarse,
                                                               coeff),
        queue_idx);

    // Array<D, T, DeviceType> ac_x({(v.shape(0)-1)/8+1, (v.shape(1)-1)/8+1,
    // (v.shape(2)-1)/8+1}, false, false); Array<D, T, DeviceType>
    // ac_y({(v.shape(0)-1)/8+1, (v.shape(1)-1)/8+1, (v.shape(2)-1)/8+1}, false,
    // false); Array<D, T, DeviceType> ac_z({(v.shape(0)-1)/8+1,
    // (v.shape(1)-1)/8+1, (v.shape(2)-1)/8+1}, false, false);

    // DeviceLauncher<DeviceType>::Execute(
    //     Autocorrelation8x8x8Kernel<D, T, DECOMPOSE, DeviceType>(v,
    //     SubArray(ac_x), SubArray(ac_y),
    //                                                                SubArray(ac_z),
    //                                                                   1),
    //                                                                   queue_idx);

    // PrintSubarray("ac_x", SubArray<2, T, DeviceType>({ac_x.shape(0),
    // ac_x.shape(1)}, ac_x.data())); PrintSubarray("ac_y", SubArray<2, T,
    // DeviceType>({ac_x.shape(0), ac_y.shape(1)}, ac_y.data()));
    // PrintSubarray("ac_z", SubArray<2, T, DeviceType>({ac_z.shape(0),
    // ac_z.shape(1)}, ac_z.data()));
  }
}

template <DIM D, typename T, typename DeviceType>
void recompose(SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> coarse,
               SubArray<1, T, DeviceType> coeff, int queue_idx) {

  if constexpr (D <= 3) {
  }
}

} // namespace in_cache_block

} // namespace data_refactoring

} // namespace mgard_x

#endif