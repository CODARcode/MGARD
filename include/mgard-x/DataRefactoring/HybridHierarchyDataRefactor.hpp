/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include "../Hierarchy/HybridHierarchy.h"
#include "../RuntimeX/Utilities/Exceptions.h"
#include "BlockLocalHierarchyDataRefactor.hpp"
#include "DataRefactor.hpp"
#include "HybridHierarchyDataRefactorInterface.hpp"
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
  HybridHierarchyDataRefactor(HybridHierarchy<D, T, DeviceType> &hierarchy,
                              Config config)
      : initialized(true), hierarchy(&hierarchy), config(config) {
    this->L = config.num_local_refactoring_level;
    this->M = config.num_global_refactoring_level;
  }

  void Adapt(HybridHierarchy<D, T, DeviceType> &hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->config = config;
    this->L = config.num_local_refactoring_level;
    this->M = config.num_global_refactoring_level;

    // Adaptive intialization for local and global
    if (this->L == 0 && this->M == 0) {
      throw ProcessingException("Both L and M cannot be zero");
    }
    hierarchy.check_config(config);

    if (this->L > 0) {
      local_refactor.Adapt(hierarchy, config, queue_idx);
    }

    if (this->M > 0) {
      // The global hierarchy is built (once) by the HybridHierarchy, on the
      // coarsest local grid or, without a local stage, on the input grid.
      global_refactor.Adapt(
          hierarchy.global_hierarchy(),
          HybridHierarchy<D, T, DeviceType>::global_config(this->M), queue_idx);
    }
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape,
                                        Config config) {
    size_t size = 0;

    SIZE L = config.num_local_refactoring_level;
    SIZE M = config.num_global_refactoring_level;

    if (L > 0) {
      size += BlockLocalHierarchyDataRefactor<
          D, T, DeviceType>::EstimateMemoryFootprint(shape);
      if (M > 0) {
        // Calculate Coarest shape from local
        std::vector<SIZE> coarest_shape = shape;
        for (SIZE l = 0; l < config.num_local_refactoring_level; l++) {
          for (DIM d = 0; d < D; d++) {
            coarest_shape[d] = ((coarest_shape[d] - 1) / 8 + 1) * 5;
          }
        }
        size += DataRefactor<D, T, DeviceType>::EstimateMemoryFootprint(
            coarest_shape);
      }
    } else {
      size += DataRefactor<D, T, DeviceType>::EstimateMemoryFootprint(shape);
    }
    return size;
  }

  size_t DecomposedDataSize() {
    if (this->L > 0) {
      return local_refactor.DecomposedDataSize();
    }

    return hierarchy->total_num_elems();
  }

  // Global-stage decomposition over the coarsest region at the front of
  // decomposed_data (in-place). Factored out so the fused
  // decompose+quantize path can run it separately from the local stage.
  void DecomposeGlobal(SubArray<1, T, DeviceType> decomposed_data,
                       int queue_idx, bool orthogonal_projection = true) {
    std::vector<SIZE> global_shape = hierarchy->global_shape();
    SubArray<D, T, DeviceType> global_input_data(global_shape,
                                                 decomposed_data.data());
    for (DIM d = 0; d < D; d++) {
      global_input_data.setLd(d, global_shape[d]);
    }
    global_input_data.project(D - 3, D - 2, D - 1);

    global_refactor.Decompose(global_input_data, orthogonal_projection,
                              queue_idx);
  }

  // Global-stage recomposition over the coarsest region at the front of
  // decomposed_data (in-place). Factored out so the fused
  // dequantize+recompose path can run it separately from the local stage.
  void RecomposeGlobal(SubArray<1, T, DeviceType> decomposed_data,
                       int queue_idx, bool orthogonal_projection = true) {
    std::vector<SIZE> global_shape = hierarchy->global_shape();
    SubArray<D, T, DeviceType> global_input_data(global_shape,
                                                 decomposed_data.data());
    for (DIM d = 0; d < D; d++) {
      global_input_data.setLd(d, global_shape[d]);
    }
    global_input_data.project(D - 3, D - 2, D - 1);

    global_refactor.Recompose(global_input_data, orthogonal_projection,
                              queue_idx);
  }

  // Need revise further to exclude copy time
  void Decompose(SubArray<D, T, DeviceType> data,
                 SubArray<1, T, DeviceType> decomposed_data, int queue_idx,
                 bool orthogonal_projection = true) {
    if (this->L == 0 && this->M == 0) {
      throw ProcessingException("Both L and M cannot be zero");
    }
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    if (this->L == 0) {
      // Pure Global (In-Place)
      std::vector<SIZE> original_shape = hierarchy->input_shape();
      SubArray<D, T, DeviceType> global_input_data(original_shape,
                                                   decomposed_data.data());
      for (DIM d = 0; d < D; d++) {
        global_input_data.setLd(d, original_shape[d]);
      }
      global_input_data.project(D - 3, D - 2, D - 1);

      multi_dimension::CopyND(data, global_input_data, queue_idx);

      global_refactor.Decompose(global_input_data, orthogonal_projection,
                                queue_idx);
    } else if (this->M == 0) {
      // Pure Local
      local_refactor.Decompose(data, decomposed_data, queue_idx,
                               orthogonal_projection);
    } else {
      // Local decomposition
      local_refactor.Decompose(data, decomposed_data, queue_idx,
                               orthogonal_projection);

      // Global decomposition
      DecomposeGlobal(decomposed_data, queue_idx, orthogonal_projection);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Hybrid Decomposition",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }
  }

  // Need revise further to exclude copy time
  void Recompose(SubArray<D, T, DeviceType> data,
                 SubArray<1, T, DeviceType> decomposed_data, int queue_idx,
                 bool orthogonal_projection = true) {
    if (this->L == 0 && this->M == 0) {
      throw ProcessingException("Both L and M cannot be zero");
    }
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    if (this->L == 0) {
      // Pure Global (In-Place)
      std::vector<SIZE> original_shape = hierarchy->input_shape();
      SubArray<D, T, DeviceType> global_input_data(original_shape,
                                                   decomposed_data.data());
      for (DIM d = 0; d < D; d++) {
        global_input_data.setLd(d, original_shape[d]);
      }
      global_input_data.project(D - 3, D - 2, D - 1);

      global_refactor.Recompose(global_input_data, orthogonal_projection,
                                queue_idx);

      // Copy back to data
      multi_dimension::CopyND(global_input_data, data, queue_idx);
    } else if (this->M == 0) {
      // Pure Local
      local_refactor.Recompose(data, decomposed_data, queue_idx,
                               orthogonal_projection);
    } else {
      std::vector<SIZE> local_coarest_shape =
          local_refactor.coarse_shapes[this->L - 1];
      SubArray<D, T, DeviceType> global_input_data({local_coarest_shape},
                                                   decomposed_data.data());
      for (DIM d = 0; d < D; d++) {
        global_input_data.setLd(d, local_coarest_shape[d]);
      }
      global_input_data.project(D - 3, D - 2, D - 1);

      // Global recomposition
      global_refactor.Recompose(global_input_data, orthogonal_projection,
                                queue_idx);

      // Local recomposition
      local_refactor.Recompose(data, decomposed_data, queue_idx,
                               orthogonal_projection);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Hybrid Recomposition",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }
  }

  bool initialized;
  HybridHierarchy<D, T, DeviceType> *hierarchy;
  Config config;

  SIZE L; // Number of local levels
  SIZE M; // Number of global levels

  BlockLocalHierarchyDataRefactor<D, T, DeviceType> local_refactor;
  DataRefactor<D, T, DeviceType> global_refactor;
};

} // namespace data_refactoring

} // namespace mgard_x

#endif
