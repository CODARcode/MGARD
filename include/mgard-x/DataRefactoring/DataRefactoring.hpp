/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "MultiDimension/DataRefactoring.h"
#include "SingleDimension/DataRefactoring.h"
#include "DataRefactoringWorkspace.hpp"

#ifndef MGARD_X_DATA_REFACTORING_HPP
#define MGARD_X_DATA_REFACTORING_HPP
namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void Decompose(Hierarchy<D, T, DeviceType> &hierarchy,
         Array<D, T, DeviceType> &in_array, Config config,
         DataRefactoringWorkspace<D, T, DeviceType> &workspace,
         int queue_idx) {
  SubArray in_subarray(in_array);
  if (config.decomposition == decomposition_type::MultiDim) {
    decompose<D, T, DeviceType>(hierarchy, in_subarray,
                                workspace.refactoring_w_subarray,
                                workspace.refactoring_b_subarray, 0, queue_idx);
  } else if (config.decomposition == decomposition_type::SingleDim) {
    decompose_single<D, T, DeviceType>(hierarchy, in_subarray, 0, queue_idx);
  }
}

template <DIM D, typename T, typename DeviceType>
void Recompose(Hierarchy<D, T, DeviceType> &hierarchy,
         Array<D, T, DeviceType> &in_array, Config config,
         DataRefactoringWorkspace<D, T, DeviceType> &workspace,
         int queue_idx) {
  SubArray in_subarray(in_array);
  if (config.decomposition == decomposition_type::MultiDim) {
    recompose<D, T, DeviceType>(
        hierarchy, in_subarray, workspace.refactoring_w_subarray,
        workspace.refactoring_b_subarray, hierarchy.l_target(), queue_idx);
  } else if (config.decomposition == decomposition_type::SingleDim) {
    recompose_single<D, T, DeviceType>(hierarchy, in_subarray,
                                       hierarchy.l_target(), queue_idx);
  }
}
}

#endif