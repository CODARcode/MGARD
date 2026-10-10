/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_HYBRID_HIERARCHY_DATA_REFACTOR_INTERFACE_HPP
#define MGARD_X_160_HYBRID_HIERARCHY_DATA_REFACTOR_INTERFACE_HPP
namespace mgard_x_160 {

namespace data_refactoring {

template <DIM D, typename T, typename DeviceType>
class HybridHierarchyDataRefactorInterface {
  virtual void Decompose(SubArray<D, T, DeviceType> data,
                         SubArray<1, T, DeviceType> decomposed_data,
                         int queue_idx, bool orthogonal_projection = true) = 0;
  virtual void Recompose(SubArray<D, T, DeviceType> data,
                         SubArray<1, T, DeviceType> decomposed_data,
                         int queue_idx, bool orthogonal_projection = true) = 0;
};

} // namespace data_refactoring

} // namespace mgard_x_160

#endif
