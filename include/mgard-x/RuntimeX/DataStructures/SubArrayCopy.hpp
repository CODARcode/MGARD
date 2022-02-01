/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_SUBARRAY_COPY_HPP
#define MGARD_X_SUBARRAY_COPY_HPP

namespace mgard_x {

template <typename DeviceType1, typename DeviceType2>
class CompatibleDeviceType {
  using DeviceType = std::conditional<
      std::is_same<DeviceType1, Serial>::value &&
          std::is_same<DeviceType2, Serial>::value,
      Serial,
      std::conditional<
          std::is_same<DeviceType1, CUDA>::value ||
              std::is_same<DeviceType2, CUDA>::value,
          CUDA,
          std::conditional<
              std::is_same<DeviceType1, HIP>::value ||
                  std::is_same<DeviceType2, HIP>::value,
              HIP,
              std::conditional<std::is_same<DeviceType1, KOKKOS>::value ||
                                   std::is_same<DeviceType2, KOKKOS>::value,
                               KOKKOS, None>>>>;
};

template <typename SubArrayType1, typename SubArrayType2>
void SubArrayCopy1D(SubArrayType1 subArray1, SubArrayType2 subArray2,
                    SIZE count, int queue_idx) {
  using DeviceType = typename CompatibleDeviceType<
      typename SubArrayType1::DevType,
      typename SubArrayType2::DevType>::DeviceType;
  MemoryManager<DeviceType>::Copy1D(subArray1.data(), subArray2.data(), count,
                                    queue_idx);
}

template <typename SubArrayType1, typename SubArrayType2>
void SubArrayCopyND(SubArrayType1 subArray1, SubArrayType2 subArray2,
                    SIZE count1, SIZE count2, int queue_idx) {
  using DeviceType = typename CompatibleDeviceType<
      typename SubArrayType1::DevType,
      typename SubArrayType2::DevType>::DeviceType;
  MemoryManager<DeviceType>::CopyND(subArray1.data(), subArray1.getLd(0),
                                    subArray2.data(), subArray2.getLd(0),
                                    count1, count2, queue_idx);
}

} // namespace mgard_x
#endif