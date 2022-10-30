/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_DICTIONARY_SHIFT_TEMPLATE_HPP
#define MGARD_X_DICTIONARY_SHIFT_TEMPLATE_HPP
#include "../../RuntimeX/RuntimeX.h"

#define MGARDX_SHIFT_DICT 1
#define MGARDX_RESTORE_DICT 2

namespace mgard_x {

template <typename T, OPTION OP, typename DeviceType>
class DictionaryShiftFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT DictionaryShiftFunctor() {}
  MGARDX_CONT DictionaryShiftFunctor(SubArray<1, T, DeviceType> v,
                                     SIZE dict_size)
      : v(v), dict_size(dict_size) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE id = FunctorBase<DeviceType>::GetBlockIdX() *
                  FunctorBase<DeviceType>::GetBlockDimX() +
              FunctorBase<DeviceType>::GetThreadIdX();
    if (id < v.shape(0)) {
      T value = *v(id);
      if constexpr (OP == MGARDX_SHIFT_DICT) {
        *v(id) += dict_size / 2;
      } else if constexpr (OP == MGARDX_RESTORE_DICT) {
        *v(id) -= dict_size / 2;
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, T, DeviceType> v;
  SIZE dict_size;
};

template <typename T, OPTION OP, typename DeviceType>
class DictionaryShiftKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "DictionaryShift";
  MGARDX_CONT
  DictionaryShiftKernel(SubArray<1, T, DeviceType> v, SIZE dict_size)
      : v(v), dict_size(dict_size) {}

  MGARDX_CONT
  Task<DictionaryShiftFunctor<T, OP, DeviceType>> GenTask(int queue_idx) {
    using FunctorType = DictionaryShiftFunctor<T, OP, DeviceType>;
    FunctorType functor(v, dict_size);
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
};

} // namespace mgard_x

#endif