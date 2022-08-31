/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_COMPRESSION_LOW_LEVEL_WORKSPACE_HPP
#define MGARD_X_COMPRESSION_LOW_LEVEL_WORKSPACE_H

#include "../Hierarchy/Hierarchy.hpp"
#include "../RuntimeX/RuntimeXPublic.h"

namespace mgard_x {

template<DIM D, typename T, typename DeviceType>
class CompressionLowLevelWorkspace {
public:
  CompressionLowLevelWorkspace() {
    // By defualt it is not pre-allocated
    pre_allocated = false;
  }
  
  void allocate(Hierarchy<D, T, DeviceType>& hierarchy, Config &config, double estimated_outlier_ratio) {
    // Do pre-allocation
    quantizers_array = Array<1, T, DeviceType>({hierarchy.l_target() + 1});
    quantized_array = Array<D, QUANTIZED_INT, DeviceType>(hierarchy.level_shape(hierarchy.l_target()), false, false);
    outlier_count_array = Array<1, LENGTH, DeviceType>({1}, false, false);
    outlier_idx_array = Array<1, LENGTH, DeviceType>({(SIZE)(hierarchy.total_num_elems()*estimated_outlier_ratio)});
    outliers_array = Array<1, QUANTIZED_INT, DeviceType>({(SIZE)(hierarchy.total_num_elems()*estimated_outlier_ratio)});
    // Reset the outlier count to zero
    outlier_count_array.memset(0);
    outlier_idx_array.memset(0);
    outliers_array.memset(0);
    if (config.reorder != 0) {
      quantized_linearized_array = Array<1, QUANTIZED_INT, DeviceType>({(SIZE)hierarchy.total_num_elems()});
    }
    pre_allocated = true;
  }

  void move(const CompressionLowLevelWorkspace<D, T, DeviceType> &workspace) {
    // Move instead of copy
    quantizers_array = std::move(workspace.quantizers_array);
    quantized_array = std::move(workspace.quantized_array);
    outlier_count_array = std::move(workspace.outlier_count_array);
    outlier_idx_array = std::move(workspace.outlier_idx_array);
    outliers_array = std::move(workspace.outliers_array);
    quantized_linearized_array = std::move(workspace.quantized_linearized_array);
  }

  void move(CompressionLowLevelWorkspace<D, T, DeviceType> &&workspace) {
    // Move instead of copy
    quantizers_array = std::move(workspace.quantizers_array);
    quantized_array = std::move(workspace.quantized_array);
    outlier_count_array = std::move(workspace.outlier_count_array);
    outlier_idx_array = std::move(workspace.outlier_idx_array);
    outliers_array = std::move(workspace.outliers_array);
    quantized_linearized_array = std::move(workspace.quantized_linearized_array);
  }

  CompressionLowLevelWorkspace(Hierarchy<D, T, DeviceType>& hierarchy, Config &config, double estimated_outlier_ratio = 0.3) {
    allocate(hierarchy, config, estimated_outlier_ratio);
  }

  CompressionLowLevelWorkspace(const CompressionLowLevelWorkspace<D, T, DeviceType> &workspace) {
    move(std::move(workspace));
  }

  CompressionLowLevelWorkspace &operator=(const CompressionLowLevelWorkspace<D, T, DeviceType> &workspace) {
    move(std::move(workspace));
  }

  CompressionLowLevelWorkspace(CompressionLowLevelWorkspace<D, T, DeviceType> &&workspace) {
    move(std::move(workspace));
  }

  CompressionLowLevelWorkspace &operator=(CompressionLowLevelWorkspace<D, T, DeviceType> &&workspace) {
    move(std::move(workspace));
  }

  bool pre_allocated;
  Array<1, T, DeviceType> quantizers_array;
  Array<D, QUANTIZED_INT, DeviceType> quantized_array;
  Array<1, LENGTH, DeviceType> outlier_count_array;
  Array<1, LENGTH, DeviceType> outlier_idx_array;
  Array<1, QUANTIZED_INT, DeviceType> outliers_array;
  Array<1, QUANTIZED_INT, DeviceType> quantized_linearized_array;
};

} // namespace mgard_x

#endif