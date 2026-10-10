/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_DATA_REFACTOR_INTERFACE_HPP
#define MGARD_X_160_DATA_REFACTOR_INTERFACE_HPP
namespace mgard_x_160 {

namespace data_refactoring {

template <DIM D, typename T, typename DeviceType> class DataRefactorInterface {
  virtual void Decompose(SubArray<D, T, DeviceType> data, int start_level,
                         int stop_level, bool orthogonal_projection,
                         int queue_idx) = 0;
  virtual void Recompose(SubArray<D, T, DeviceType> data, int start_level,
                         int stop_level, bool orthogonal_projection,
                         int queue_idx) = 0;
  virtual void Decompose(SubArray<D, T, DeviceType> data,
                         bool orthogonal_projection, int queue_idx) = 0;
  virtual void Recompose(SubArray<D, T, DeviceType> data,
                         bool orthogonal_projection, int queue_idx) = 0;
};

} // namespace data_refactoring

} // namespace mgard_x_160

#endif