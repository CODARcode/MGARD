/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.hpp"
#include "../../../RuntimeX/RuntimeX.h"

#include "../DataRefactoring.h"

#include "EarlyFeatureDetectorKernel.hpp"

#ifndef MGARD_X_DATA_REFACTORING_EARLY_FEATURE_DETECTOR
#define MGARD_X_DATA_REFACTORING_EARLY_FEATURE_DETECTOR

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void EarlyFeatureDetector(SubArray<D, T, DeviceType> v, T current_error, T iso_value,
                          SubArray<D, SIZE, DeviceType> feature_flag_coarser,
                          SubArray<D, SIZE, DeviceType> feature_flag, int queue_idx) {

  EarlyFeatureDetectorKernel<D, T, DeviceType>().Execute(
      v, current_error, iso_value, feature_flag_coarser, feature_flag, queue_idx);

  if (multidim_refactoring_debug_print) {
    PrintSubarray("feature_flag", feature_flag);
  }
}

} // namespace mgard_x

#endif