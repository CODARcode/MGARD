/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_NORM_CALCULATOR_HPP
#define MGARD_X_NORM_CALCULATOR_HPP

namespace mgard_x {
template <DIM D, typename T, typename DeviceType>
T norm_calculator(Array<D, T, DeviceType> &original_array,
                  SubArray<1, T, DeviceType> workspace_subarray,
                  SubArray<1, T, DeviceType> norm_subarray, T s,
                  bool normalize_coordinates) {
  Timer timer;
  if (log::level & log::TIME)
    timer.start();
  SIZE total_elems = 1;
  for (DIM d = 0; d < D; d++)
    total_elems *= original_array.shape(d);

  T norm = 0;
  SubArray<1, T, DeviceType> temp_subarray;
  if (!original_array.isPitched()) { // zero copy
    log::info("Use zero copy when calculating norm");
    temp_subarray =
        SubArray<1, T, DeviceType>({total_elems}, original_array.data());
  } else { // need to linearized
    log::info("Explicit copy used when calculating norm");
    temp_subarray = workspace_subarray;
    SIZE linearized_width = 1;
    for (DIM d = 0; d < D - 1; d++)
      linearized_width *= original_array.shape(d);
    MemoryManager<DeviceType>::CopyND(
        temp_subarray.data(), original_array.shape(D - 1),
        original_array.data(), original_array.ld(D - 1),
        original_array.shape(D - 1), linearized_width, 0);
  }
  DeviceRuntime<DeviceType>::SyncQueue(0);
  if (s == std::numeric_limits<T>::infinity()) {
    Array<1, Byte, DeviceType> abs_max_workspace;
    DeviceCollective<DeviceType>::AbsMax(
        total_elems, temp_subarray, norm_subarray, abs_max_workspace, false, 0);
    DeviceCollective<DeviceType>::AbsMax(
        total_elems, temp_subarray, norm_subarray, abs_max_workspace, true, 0);
    MemoryManager<DeviceType>::Copy1D(&norm, norm_subarray.data(), 1, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    // Avoiding issue with norm == 0
    if (norm == 0)
      norm = std::numeric_limits<T>::epsilon();
    log::info("L_inf norm: " + std::to_string(norm));
  } else {
    Array<1, Byte, DeviceType> square_sum_workspace;
    DeviceCollective<DeviceType>::SquareSum(total_elems, temp_subarray,
                                            norm_subarray, square_sum_workspace,
                                            false, 0);
    DeviceCollective<DeviceType>::SquareSum(total_elems, temp_subarray,
                                            norm_subarray, square_sum_workspace,
                                            true, 0);
    MemoryManager<DeviceType>::Copy1D(&norm, norm_subarray.data(), 1, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    if (!normalize_coordinates) {
      norm = std::sqrt(norm);
    } else {
      norm = std::sqrt(norm / total_elems);
    }
    // Avoiding issue with norm == 0
    if (norm == 0)
      norm = std::numeric_limits<T>::epsilon();
    log::info("L_2 norm: " + std::to_string(norm));
  }
  if (log::level & log::TIME) {
    timer.end();
    timer.print("Calculate norm");
    timer.clear();
  }
  return norm;
}

} // namespace mgard_x

#endif