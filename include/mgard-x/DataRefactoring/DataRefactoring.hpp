/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "DataRefactoringWorkspace.hpp"
#include "MultiDimension/DataRefactoring.h"
#include "SingleDimension/DataRefactoring.h"

#ifndef MGARD_X_DATA_REFACTORING_HPP
#define MGARD_X_DATA_REFACTORING_HPP
namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void Decompose(Hierarchy<D, T, DeviceType> &hierarchy,
               Array<D, T, DeviceType> &in_array, Config config,
               DataRefactoringWorkspace<D, T, DeviceType> &workspace,
               int queue_idx) {
  Timer timer;
  if (log::level & log::TIME)
    timer.start();
  SubArray in_subarray(in_array);
  if (config.decomposition == decomposition_type::MultiDim) {
    decompose<D, T, DeviceType>(hierarchy, in_subarray,
                                workspace.refactoring_w_subarray,
                                workspace.refactoring_b_subarray, 0, queue_idx);
  } else if (config.decomposition == decomposition_type::SingleDim) {
    decompose_single<D, T, DeviceType>(hierarchy, in_subarray, 0, queue_idx);
  }
  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    timer.print("Decomposition");
    log::time("Decomposition throughput: " +
              std::to_string((double)(hierarchy.total_num_elems() * sizeof(T)) /
                             timer.get() / 1e9) +
              " GB/s");
    timer.clear();
  }
}

template <DIM D, typename T, typename DeviceType>
void Recompose(Hierarchy<D, T, DeviceType> &hierarchy,
               Array<D, T, DeviceType> &in_array, Config config,
               DataRefactoringWorkspace<D, T, DeviceType> &workspace,
               int queue_idx) {
  Timer timer;
  if (log::level & log::TIME)
    timer.start();
  SubArray in_subarray(in_array);
  if (config.decomposition == decomposition_type::MultiDim) {
    recompose<D, T, DeviceType>(
        hierarchy, in_subarray, workspace.refactoring_w_subarray,
        workspace.refactoring_b_subarray, hierarchy.l_target(), queue_idx);
  } else if (config.decomposition == decomposition_type::SingleDim) {
    recompose_single<D, T, DeviceType>(hierarchy, in_subarray,
                                       hierarchy.l_target(), queue_idx);
  }
  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    timer.print("Recomposition");
    log::time("Recomposition throughput: " +
              std::to_string((double)(hierarchy.total_num_elems() * sizeof(T)) /
                             timer.get() / 1e9) +
              " GB/s");
    timer.clear();
  }
}
} // namespace mgard_x

#endif