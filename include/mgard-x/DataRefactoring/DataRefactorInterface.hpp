/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_DATA_REFACTOR_INTERFACE_HPP
#define MGARD_X_DATA_REFACTOR_INTERFACE_HPP
namespace mgard_x {
template <DIM D, typename T, typename DeviceType> class DataRefactorInterface {
  virtual void Decompose(Array<D, T, DeviceType> &data, int stop_level,
                         int queue_idx) = 0;
  virtual void Recompose(Array<D, T, DeviceType> &data, int stop_level,
                         int queue_idx) = 0;
  virtual void Decompose(Array<D, T, DeviceType> &data, int queue_idx) = 0;
  virtual void Recompose(Array<D, T, DeviceType> &data, int queue_idx) = 0;
};
} // namespace mgard_x

#endif