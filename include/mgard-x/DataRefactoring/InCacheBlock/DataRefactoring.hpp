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
void decompose(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, SubArray<D, T, DeviceType> w,
               SubArray<D, T, DeviceType> b, int start_level, int stop_level,
               int queue_idx) {

  if (start_level < 0 || start_level > hierarchy.l_target()) {
    std::cout << log::log_err << "decompose: start_level out of bound.\n";
    exit(-1);
  }

  if (stop_level < 0 || stop_level > hierarchy.l_target()) {
    std::cout << log::log_err << "decompose: stop_level out of bound.\n";
    exit(-1);
  }

  if constexpr (D <= 3) {
    DeviceLauncher<DeviceType>::Execute(
        DataRefactoringKernel<D, T, 8, 8, 8, DECOMPOSE, DeviceType>(v),
        queue_idx);
  }
}

template <DIM D, typename T, typename DeviceType>
void recompose(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, SubArray<D, T, DeviceType> w,
               SubArray<D, T, DeviceType> b, int start_level, int stop_level,
               int queue_idx) {

  if (stop_level < 0 || stop_level > hierarchy.l_target()) {
    std::cout << log::log_err << "recompose: stop_level out of bound.\n";
    exit(-1);
  }

  if (start_level < 0 || start_level > hierarchy.l_target()) {
    std::cout << log::log_err << "recompose: start_level out of bound.\n";
    exit(-1);
  }

  if constexpr (D <= 3) {
  }
}

} // namespace in_cache_block

} // namespace data_refactoring

} // namespace mgard_x

#endif