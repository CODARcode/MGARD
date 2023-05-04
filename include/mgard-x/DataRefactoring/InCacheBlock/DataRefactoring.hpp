/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../Hierarchy/Hierarchy.h"
#include "../../RuntimeX/RuntimeX.h"

#include "DataRefactoring.h"
#include "DataRefactoringKernel.hpp"

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
        DataRefactoringKernel<D, T, 8, 8, 8, DECOMPOSE, DeviceType>(v, coarse,
                                                                    coeff),
        queue_idx);
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