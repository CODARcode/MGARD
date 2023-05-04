/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "LevelwiseDataRefactorInterface.hpp"
// #include "DataRefactoringWorkspace.hpp"
#include "InCacheBlock/DataRefactoring.h"
#include "MultiDimension/DataRefactoring.h"
#include "SingleDimension/DataRefactoring.h"

#ifndef MGARD_X_LEVELWISE_DATA_REFACTOROR_HPP
#define MGARD_X_LEVELWISE_DATA_REFACTOROR_HPP
namespace mgard_x {

namespace data_refactoring {

template <DIM D, typename T, typename DeviceType>
class LevelwiseDataRefactor
    : public LevelwiseDataRefactorInterface<D, T, DeviceType> {
public:
  LevelwiseDataRefactor(Hierarchy<D, T, DeviceType> hierarchy, Config config)
      : hierarchy(hierarchy), config(config) {
    std::vector<SIZE> workspace_shape =
        hierarchy.level_shape(hierarchy.l_target());
    for (DIM d = 0; d < D; d++)
      workspace_shape[d] = ((workspace_shape[d] - 1) / 8 + 1) * 5;
    w_array = Array<D, T, DeviceType>(workspace_shape);
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    size_t size = 0;
    return size;
  }

  void Decompose(Array<D, T, DeviceType> &data,
                 Array<1, T, DeviceType> &decomposed_data, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME)
      timer.start();
    SubArray<D, T, DeviceType> data_subarray(data);
    SubArray<D, T, DeviceType> w_subarray(w_array);
    SubArray<1, T, DeviceType> decomposed_data_subarray(decomposed_data);

    in_cache_block::decompose<D, T, DeviceType>(
        data_subarray, w_subarray, decomposed_data_subarray, queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Decomposition");
      log::time(
          "Decomposition throughput: " +
          std::to_string((double)(hierarchy.total_num_elems() * sizeof(T)) /
                         timer.get() / 1e9) +
          " GB/s");
      timer.clear();
    }
  }
  void Recompose(Array<D, T, DeviceType> &data,
                 Array<1, T, DeviceType> &decomposed_data, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME)
      timer.start();
    SubArray<D, T, DeviceType> data_subarray(data);
    SubArray<D, T, DeviceType> w_subarray(w_array);
    SubArray<1, T, DeviceType> decomposed_data_subarray(decomposed_data);

    in_cache_block::recompose<D, T, DeviceType>(
        data_subarray, w_subarray, decomposed_data_subarray, queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Recomposition");
      log::time(
          "Recomposition throughput: " +
          std::to_string((double)(hierarchy.total_num_elems() * sizeof(T)) /
                         timer.get() / 1e9) +
          " GB/s");
      timer.clear();
    }
  }

  Hierarchy<D, T, DeviceType> hierarchy;
  Config config;
  Array<D, T, DeviceType> w_array;
};

} // namespace data_refactoring

} // namespace mgard_x

#endif