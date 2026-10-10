/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include "DataRefactorInterface.hpp"
#include "InCacheBlock/DataRefactoring.h"
#include "MultiDimension/DataRefactoring.h"
#include "SingleDimension/DataRefactoring.h"

#ifndef MGARD_X_160_DATA_REFACTOROR_HPP
#define MGARD_X_160_DATA_REFACTOROR_HPP
namespace mgard_x_160 {

namespace data_refactoring {

template <DIM D, typename T, typename DeviceType>
class DataRefactor : public DataRefactorInterface<D, T, DeviceType> {
public:
  DataRefactor() : initialized(false) {}
  DataRefactor(Hierarchy<D, T, DeviceType> &hierarchy, Config config)
      : initialized(true), hierarchy(&hierarchy), config(config) {
    std::vector<SIZE> workspace_shape =
        hierarchy.level_shape(hierarchy.l_target());
    for (DIM d = 0; d < D; d++)
      workspace_shape[d] += 2;
    w_array = Array<D, T, DeviceType>(workspace_shape);
    if (D > 3) {
      b_array = Array<D, T, DeviceType>(workspace_shape);
    }
  }

  void Adapt(Hierarchy<D, T, DeviceType> &hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->config = config;
    std::vector<SIZE> workspace_shape =
        hierarchy.level_shape(hierarchy.l_target());
    for (DIM d = 0; d < D; d++)
      workspace_shape[d] += 2;
    w_array.resize(workspace_shape, queue_idx);
    if (D > 3) {
      b_array.resize(workspace_shape, queue_idx);
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

  void Decompose(SubArray<D, T, DeviceType> data, int start_level,
                 int stop_level, bool orthogonal_projection, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    SubArray<D, T, DeviceType> w_subarray(w_array);
    SubArray<D, T, DeviceType> b_subarray;
    if (D > 3) {
      b_subarray = SubArray<D, T, DeviceType>(b_array);
    }

    if (config.decomposition == decomposition_type::MultiDim ||
        config.decomposition == decomposition_type::Hybrid) {
      multi_dimension::decompose<D, T, DeviceType>(
          *hierarchy, data, w_subarray, b_subarray, start_level, stop_level,
          orthogonal_projection, queue_idx);
    } else if (config.decomposition == decomposition_type::SingleDim) {
      single_dimension::decompose<D, T, DeviceType>(
          *hierarchy, data, start_level, stop_level, queue_idx);
    }
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Global Decomposition",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }
  }
  void Recompose(SubArray<D, T, DeviceType> data, int start_level,
                 int stop_level, bool orthogonal_projection, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    SubArray<D, T, DeviceType> w_subarray(w_array);
    SubArray<D, T, DeviceType> b_subarray;
    if (D > 3)
      b_subarray = SubArray<D, T, DeviceType>(b_array);
    if (config.decomposition == decomposition_type::MultiDim) {
      multi_dimension::recompose<D, T, DeviceType>(
          *hierarchy, data, w_subarray, b_subarray, start_level, stop_level,
          orthogonal_projection, queue_idx);
    } else if (config.decomposition == decomposition_type::SingleDim) {
      single_dimension::recompose<D, T, DeviceType>(
          *hierarchy, data, start_level, stop_level, queue_idx);
    }
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Global Recomposition",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }
  }

  void Decompose(SubArray<D, T, DeviceType> data, bool orthogonal_projection,
                 int queue_idx) {
    Decompose(data, hierarchy->l_target(), 0, orthogonal_projection, queue_idx);
  }

  void Recompose(SubArray<D, T, DeviceType> data, bool orthogonal_projection,
                 int queue_idx) {
    Recompose(data, 0, hierarchy->l_target(), orthogonal_projection, queue_idx);
  }

  bool initialized;
  Hierarchy<D, T, DeviceType> *hierarchy;
  Config config;
  Array<D, T, DeviceType> w_array;
  Array<D, T, DeviceType> b_array;
};

} // namespace data_refactoring

} // namespace mgard_x_160

#endif