/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.hpp"
#include "../../../RuntimeX/RuntimeX.h"

#include "../DataRefactoring.h"

#include "LevelMaxKernel.hpp"

#ifndef MGARD_X_DATA_REFACTORING_LEVEL_MAX
#define MGARD_X_DATA_REFACTORING_LEVEL_MAX

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void LevelMax(SIZE l_target,
               SubArray<1, SIZE, DeviceType> ranges,
               SubArray<D, T, DeviceType> v,
               SubArray<1, T, DeviceType> level_max, int queue_idx) {

  LevelMaxKernel<D, T, DeviceType>().Execute(
      l_target, ranges, v, level_max, queue_idx);

  if (multidim_refactoring_debug_print) {
    PrintSubarray("LevelMax", level_max);
  }
}

} // namespace mgard_x

#endif