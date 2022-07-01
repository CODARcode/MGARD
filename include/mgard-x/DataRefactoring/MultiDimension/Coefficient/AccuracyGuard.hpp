/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.hpp"
#include "../../../RuntimeX/RuntimeX.h"

#include "../DataRefactoring.h"

#include "AccuracyGuardKernel.hpp"

#ifndef MGARD_X_DATA_REFACTORING_ACCURACY_GUARD
#define MGARD_X_DATA_REFACTORING_ACCURACY_GUARD

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void AccuracyGuard(SIZE total_level, SIZE current_level, 
                   SubArray<1, T, DeviceType> error_impact_budget,
                   SubArray<1, T, DeviceType> previous_error_impact,
                   SubArray<1, T, DeviceType> current_error_impact,
                   SubArray<D+1, T, DeviceType> max_abs_coefficient,
                   SubArray<D, SIZE, DeviceType> refinement_flag,
                   int queue_idx) {

  AccuracyGuardKernel<D, T, DeviceType>().Execute(
                        total_level, current_level, error_impact_budget, previous_error_impact,
                        current_error_impact, max_abs_coefficient, refinement_flag, queue_idx);

  if (multidim_refactoring_debug_print) {
    PrintSubarray("refinement_flag", refinement_flag);
  }
}

} // namespace mgard_x

#endif