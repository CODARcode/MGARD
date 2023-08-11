/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_OUTLIER_SEPARATOR_TEMPLATE_HPP
#define MGARD_X_OUTLIER_SEPARATOR_TEMPLATE_HPP
#include "../../RuntimeX/RuntimeX.h"

#define MGARDX_SEPARATE_OUTLIER 1
#define MGARDX_RESTORE_OUTLIER 2

namespace mgard_x {

template <typename T, OPTION OP, typename DeviceType>
class OutlierSeparatorFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT OutlierSeparatorFunctor() {}
  MGARDX_CONT
  OutlierSeparatorFunctor(SubArray<1, T, DeviceType> v, SIZE dict_size,
                          SubArray<1, ATOMIC_IDX, DeviceType> outlier_count,
                          SubArray<1, ATOMIC_IDX, DeviceType> outlier_index,
                          SubArray<1, T, DeviceType> outlier_value)
      : v(v), dict_size(dict_size), outlier_count(outlier_count),
        outlier_index(outlier_index), outlier_value(outlier_value) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE id = FunctorBase<DeviceType>::GetBlockIdX() *
                  FunctorBase<DeviceType>::GetBlockDimX() +
              FunctorBase<DeviceType>::GetThreadIdX();
    if (id < v.shape(0)) {
      T value = *v(id);
      if constexpr (OP == MGARDX_SEPARATE_OUTLIER) {
        // printf("%d %lld %d\n", value, dict_size, value < 0 || value >=
        // dict_size);
        if (value < 0 || value >= dict_size) {
          ATOMIC_IDX outlier_write_index =
              Atomic<ATOMIC_IDX, AtomicGlobalMemory, AtomicDeviceScope,
                     DeviceType>::Add(outlier_count((IDX)0), (ATOMIC_IDX)1);
          if (outlier_write_index < outlier_index.shape(0)) {
            *outlier_index(outlier_write_index) = id;
            *outlier_value(outlier_write_index) = value;
            *v(id) = 0;
          }
        }
      } else if constexpr (OP == MGARDX_RESTORE_OUTLIER) {
        if (id < outlier_value.shape(0)) {
          ATOMIC_IDX index = *outlier_index(id);
          QUANTIZED_INT value = *outlier_value(id);
          *v(index) = value;
        }
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, T, DeviceType> v;
  SIZE dict_size;
  SubArray<1, ATOMIC_IDX, DeviceType> outlier_count;
  SubArray<1, ATOMIC_IDX, DeviceType> outlier_index;
  SubArray<1, T, DeviceType> outlier_value;
};

template <typename T, OPTION OP, typename DeviceType>
class OutlierSeparatorKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "OutlierSeparator";
  MGARDX_CONT
  OutlierSeparatorKernel(SubArray<1, T, DeviceType> v, SIZE dict_size,
                         SubArray<1, ATOMIC_IDX, DeviceType> outlier_count,
                         SubArray<1, ATOMIC_IDX, DeviceType> outlier_index,
                         SubArray<1, T, DeviceType> outlier_value)
      : v(v), dict_size(dict_size), outlier_count(outlier_count),
        outlier_index(outlier_index), outlier_value(outlier_value) {}

  MGARDX_CONT
  Task<OutlierSeparatorFunctor<T, OP, DeviceType>> GenTask(int queue_idx) {
    using FunctorType = OutlierSeparatorFunctor<T, OP, DeviceType>;
    FunctorType functor(v, dict_size, outlier_count, outlier_index,
                        outlier_value);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = (v.shape(0) - 1) / tbx + 1;
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, T, DeviceType> v;
  SIZE dict_size;
  SubArray<1, ATOMIC_IDX, DeviceType> outlier_count;
  SubArray<1, ATOMIC_IDX, DeviceType> outlier_index;
  SubArray<1, T, DeviceType> outlier_value;
};

} // namespace mgard_x

#endif