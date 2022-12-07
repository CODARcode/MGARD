/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "DataRefactorInterface.hpp"
// #include "DataRefactoringWorkspace.hpp"
#include "MultiDimension/DataRefactoring.h"
#include "SingleDimension/DataRefactoring.h"

#ifndef MGARD_X_DATA_REFACTOROR_HPP
#define MGARD_X_DATA_REFACTOROR_HPP
namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
class DataRefactor : public DataRefactorInterface<D, T, DeviceType> {
public:
  DataRefactor(Hierarchy<D, T, DeviceType> hierarchy, Config config)
      : hierarchy(hierarchy), config(config) {
    std::vector<SIZE> workspace_shape =
        hierarchy.level_shape(hierarchy.l_target());
    for (DIM d = 0; d < D; d++)
      workspace_shape[d] += 2;
    w_array = Array<D, T, DeviceType>(workspace_shape);
    if (D > 3) {
      b_array = Array<D, T, DeviceType>(workspace_shape);
    }
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {

    Array<1, T, DeviceType> array_with_pitch({1});
    size_t pitch_size = array_with_pitch.ld(0) * sizeof(T);

    size_t size = 0;
    size += sizeof(T);
    size_t workspace_size = 1;
    for (DIM d = 0; d < D; d++) {
      if (d == D - 1) {
        workspace_size *=
            roundup((size_t)(shape[d] + 2) * sizeof(T), pitch_size);
      } else {
        workspace_size *= shape[d] + 2;
      }
    }
    size += workspace_size;
    if (D > 3) {
      size += workspace_size;
    }
    return size;
  }

  void Decompose(Array<D, T, DeviceType> &data, int start_level, int stop_level,
                 int queue_idx) {
    Timer timer;
    if (log::level & log::TIME)
      timer.start();
    SubArray<D, T, DeviceType> data_subarray(data);
    SubArray<D, T, DeviceType> w_subarray(w_array);
    SubArray<D, T, DeviceType> b_subarray;
    if (D > 3) {
      b_subarray = SubArray<D, T, DeviceType>(b_array);
    }

    if (config.decomposition == decomposition_type::MultiDim) {
      decompose<D, T, DeviceType>(hierarchy, data_subarray, w_subarray,
                                  b_subarray, start_level, stop_level,
                                  queue_idx);
    } else if (config.decomposition == decomposition_type::SingleDim) {
      decompose_single<D, T, DeviceType>(hierarchy, data_subarray, start_level,
                                         stop_level, queue_idx);
    }
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
  void Recompose(Array<D, T, DeviceType> &data, int start_level, int stop_level,
                 int queue_idx) {
    Timer timer;
    if (log::level & log::TIME)
      timer.start();
    SubArray<D, T, DeviceType> data_subarray(data);
    SubArray<D, T, DeviceType> w_subarray(w_array);
    SubArray<D, T, DeviceType> b_subarray;
    if (D > 3)
      b_subarray = SubArray<D, T, DeviceType>(b_array);
    if (config.decomposition == decomposition_type::MultiDim) {
      recompose<D, T, DeviceType>(hierarchy, data_subarray, w_subarray,
                                  b_subarray, start_level, stop_level,
                                  queue_idx);
    } else if (config.decomposition == decomposition_type::SingleDim) {
      recompose_single<D, T, DeviceType>(hierarchy, data_subarray, start_level,
                                         stop_level, queue_idx);
    }
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

  void Decompose(Array<D, T, DeviceType> &data, int queue_idx) {
    Decompose(data, hierarchy.l_target(), 0, queue_idx);
  }

  void Recompose(Array<D, T, DeviceType> &data, int queue_idx) {
    Recompose(data, 0, hierarchy.l_target(), queue_idx);
  }

  Hierarchy<D, T, DeviceType> hierarchy;
  Config config;
  Array<D, T, DeviceType> w_array;
  Array<D, T, DeviceType> b_array;
};

} // namespace mgard_x

#endif