/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_DATA_REFACTORING_WORKSPACE_HPP
#define MGARD_X_DATA_REFACTORING_WORKSPACE_HPP

#include "../Hierarchy/Hierarchy.h"
#include "../RuntimeX/RuntimeXPublic.h"

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
class DataRefactoringWorkspace {
public:
  DataRefactoringWorkspace() {
    // By defualt it is not pre-allocated
    pre_allocated = false;
  }

  void initialize_subarray() {
    refactoring_w_subarray = SubArray(refactoring_w_array);
    if (D > 3) {
      refactoring_b_subarray = SubArray(refactoring_b_array);
    }
  }

  size_t estimate_size(std::vector<SIZE> shape) {

    Array<1, T, DeviceType> array_with_pitch({1});
    size_t pitch_size = array_with_pitch.ld(0) * sizeof(T);

    size_t size = 0;
    size += sizeof(T);
    size_t workspace_size = 1;
    for (DIM d = 0; d < D; d++) {
      if (d == D - 1) {
        workspace_size *= roundup((shape[d] + 2) * sizeof(T), pitch_size);
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

  void allocate(Hierarchy<D, T, DeviceType> &hierarchy) {
    std::vector<SIZE> workspace_shape =
        hierarchy.level_shape(hierarchy.l_target());
    for (DIM d = 0; d < D; d++)
      workspace_shape[d] += 2;
    refactoring_w_array = Array<D, T, DeviceType>(workspace_shape);
    if (D > 3) {
      refactoring_b_array = Array<D, T, DeviceType>(workspace_shape);
    }
    initialize_subarray();

    pre_allocated = true;
  }

  void move(const DataRefactoringWorkspace<D, T, DeviceType> &workspace) {
    // Move instead of copy
    refactoring_w_array = std::move(workspace.refactoring_w_array);
    if (D > 3) {
      refactoring_b_array = std::move(workspace.refactoring_b_array);
    }
    initialize_subarray();
  }

  void move(DataRefactoringWorkspace<D, T, DeviceType> &&workspace) {
    // Move instead of copy
    refactoring_w_array = std::move(workspace.refactoring_w_array);
    if (D > 3) {
      refactoring_b_array = std::move(workspace.refactoring_b_array);
    }
    initialize_subarray();
  }

  DataRefactoringWorkspace(Hierarchy<D, T, DeviceType> &hierarchy) {
    allocate(hierarchy);
  }

  DataRefactoringWorkspace(
      const DataRefactoringWorkspace<D, T, DeviceType> &workspace) {
    move(std::move(workspace));
  }

  DataRefactoringWorkspace &
  operator=(const DataRefactoringWorkspace<D, T, DeviceType> &workspace) {
    move(std::move(workspace));
  }

  DataRefactoringWorkspace(
      DataRefactoringWorkspace<D, T, DeviceType> &&workspace) {
    move(std::move(workspace));
  }

  DataRefactoringWorkspace &
  operator=(DataRefactoringWorkspace<D, T, DeviceType> &&workspace) {
    move(std::move(workspace));
  }

  bool pre_allocated;

  Array<D, T, DeviceType> refactoring_w_array;
  Array<D, T, DeviceType> refactoring_b_array;

  SubArray<D, T, DeviceType> refactoring_w_subarray;
  SubArray<D, T, DeviceType> refactoring_b_subarray;
};

} // namespace mgard_x

#endif