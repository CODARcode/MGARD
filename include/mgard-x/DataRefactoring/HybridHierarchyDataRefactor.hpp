/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "DataRefactor.hpp"
#include "HybridHierarchyDataRefactorInterface.hpp"
// #include "DataRefactoringWorkspace.hpp"
#include "../Linearization/LevelLinearizer.hpp"
#include "InCacheBlock/DataRefactoring.h"
#include "MultiDimension/DataRefactoring.h"
#include "SingleDimension/DataRefactoring.h"

#ifndef MGARD_X_HYBRID_HIERARCHY_DATA_REFACTOROR_HPP
#define MGARD_X_HYBRID_HIERARCHY_DATA_REFACTOROR_HPP
namespace mgard_x {

namespace data_refactoring {

template <DIM D, typename T, typename DeviceType>
class HybridHierarchyDataRefactor
    : public HybridHierarchyDataRefactorInterface<D, T, DeviceType> {
public:
  HybridHierarchyDataRefactor() : initialized(false) {}
  HybridHierarchyDataRefactor(Hierarchy<D, T, DeviceType> &hierarchy,
                              Config config)
      : initialized(true), hierarchy(&hierarchy), config(config),
        global_refactor(hierarchy, config) {

    coarse_shape = hierarchy.level_shape(hierarchy.l_target());
    // If we do at least one level of local refactoring
    if (config.num_local_refactoring_level > 0) {
      for (int l = 0; l < config.num_local_refactoring_level; l++) {
        SIZE last_level_size = 1, curr_level_size = 1;

        // std::cout << coarse_shape[0] << " " << coarse_shape[1] << " "
        //           << coarse_shape[2] << "\n";
        for (DIM d = 0; d < D; d++) {
          coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 8;
          last_level_size *= coarse_shape[d];
          coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 5;
          curr_level_size *= coarse_shape[d];
        }

        // std::cout << coarse_shape[0] << " " << coarse_shape[1] << " "
        //           << coarse_shape[2] << "\n";
        coarse_shapes.push_back(coarse_shape);
        coarse_num_elems.push_back(last_level_size);
        if (l == 0) {
          coarse_array = Array<D, T, DeviceType>(coarse_shape);
        }
        local_coeff_size.push_back(last_level_size - curr_level_size);
        // std::cout << local_coeff_size[local_coeff_size.size() - 1] << "\n";
      }
    }

    global_hierarchy = Hierarchy<D, T, DeviceType>(coarse_shape, config);
    global_refactor = DataRefactor(global_hierarchy, config);
  }

  void Adapt(Hierarchy<D, T, DeviceType> &hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->config = config;
    coarse_shape = hierarchy.level_shape(hierarchy.l_target());
    coarse_shapes.clear();
    coarse_num_elems.clear();
    local_coeff_size.clear();
    // If we do at least one level of local refactoring
    if (config.num_local_refactoring_level > 0) {
      for (int l = 0; l < config.num_local_refactoring_level; l++) {
        SIZE last_level_size = 1, curr_level_size = 1;

        // std::cout << coarse_shape[0] << " " << coarse_shape[1] << " "
        //           << coarse_shape[2] << "\n";
        for (DIM d = 0; d < D; d++) {
          coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 8;
          last_level_size *= coarse_shape[d];
          coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 5;
          curr_level_size *= coarse_shape[d];
        }

        // std::cout << coarse_shape[0] << " " << coarse_shape[1] << " "
        //           << coarse_shape[2] << "\n";
        coarse_shapes.push_back(coarse_shape);
        coarse_num_elems.push_back(last_level_size);
        if (l == 0) {
          coarse_array.resize(coarse_shape, queue_idx);
        }
        local_coeff_size.push_back(last_level_size - curr_level_size);
        // std::cout << local_coeff_size[local_coeff_size.size() - 1] << "\n";
      }
    }

    global_hierarchy = Hierarchy<D, T, DeviceType>(coarse_shape, config);
    global_refactor = DataRefactor(global_hierarchy, config);
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    size_t size = 0;
    return size;
  }

  size_t DecomposedDataSize() {
    size_t coeff_size = 0;
    // local
    for (int l = 0; l < config.num_local_refactoring_level; l++) {
      coeff_size += local_coeff_size[l];
    }
    // global
    coeff_size += global_hierarchy.total_num_elems();
    return coeff_size;
  }

  void Decompose(SubArray<D, T, DeviceType> data,
                 SubArray<1, T, DeviceType> decomposed_data, int queue_idx) {

    // PrintSubarray("data", data);

    if (config.num_local_refactoring_level > 0) {
      Timer timer;
      SubArray<D, T, DeviceType> coarse_data(coarse_array);
      SIZE accumulated_local_coeff_size = 0;
      for (int l = 0; l < config.num_local_refactoring_level; l++) {
        if (log::level & log::TIME)
          timer.start();
        accumulated_local_coeff_size += local_coeff_size[l];
        SubArray<1, T, DeviceType> local_coeff(
            {local_coeff_size[l]},
            decomposed_data(decomposed_data.shape(0) -
                            accumulated_local_coeff_size));
        // std::cout << "accumulated_local_coeff_size: "
        //           << accumulated_local_coeff_size << "\n";
        in_cache_block::decompose<D, T, DeviceType>(data, coarse_data,
                                                    local_coeff, queue_idx);

        // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        // PrintSubarray("local_coeff_subarray", local_coeff_subarray);
        // PrintSubarray("coarse_subarray", coarse_subarray);
        SubArray<D, T, DeviceType> tmp = coarse_data;
        if (l + 1 < config.num_local_refactoring_level) {
          coarse_data =
              SubArray<D, T, DeviceType>(coarse_shapes[l + 1], data.data());
        }
        data = tmp;
        if (log::level & log::TIME) {
          DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
          timer.end();
          timer.print("Local Decomposition");
          log::time("Decomposition throughput: " +
                    std::to_string((double)(coarse_num_elems[l] * sizeof(T)) /
                                   timer.get() / 1e9) +
                    " GB/s");
          timer.clear();
        }

        // bool check = true;
        // VerifySubArray("coarse", coarse_data, !check, check);
      }
    }

    // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // PrintSubarray("before data", data);

    // PrintSubarray("coarse_data", SubArray<1, T, DeviceType>({10},
    // data.data()));

    // Array<D, T, DeviceType> global_data(coarse_shape, coarse_array.data());
    SubArray<D, T, DeviceType> global_coeff_subarray(
        {global_hierarchy.level_shape(global_hierarchy.l_target())},
        decomposed_data((IDX)0));
    global_refactor.Decompose(data, queue_idx);

    // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // PrintSubarray("after data", data);

    multi_dimension::CopyND(data, global_coeff_subarray, queue_idx);
  }
  void Recompose(SubArray<D, T, DeviceType> data,
                 SubArray<1, T, DeviceType> decomposed_data, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME)
      timer.start();
    SubArray<D, T, DeviceType> data_subarray(data);
    SubArray<D, T, DeviceType> w_subarray(coarse_array);
    SubArray<1, T, DeviceType> decomposed_data_subarray(decomposed_data);

    in_cache_block::recompose<D, T, DeviceType>(
        data_subarray, w_subarray, decomposed_data_subarray, queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Recomposition");
      log::time(
          "Recomposition throughput: " +
          std::to_string((double)(hierarchy->total_num_elems() * sizeof(T)) /
                         timer.get() / 1e9) +
          " GB/s");
      timer.clear();
    }
  }

  bool initialized;
  Hierarchy<D, T, DeviceType> *hierarchy;
  Hierarchy<D, T, DeviceType> global_hierarchy;
  Config config;
  std::vector<SIZE> coarse_shape;
  std::vector<SIZE> coarse_num_elems;
  DataRefactor<D, T, DeviceType> global_refactor;
  Array<D, T, DeviceType> coarse_array;
  std::vector<std::vector<SIZE>> coarse_shapes;
  std::vector<SIZE> local_coeff_size;
};

} // namespace data_refactoring

} // namespace mgard_x

#endif